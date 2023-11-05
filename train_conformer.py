"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

General training script.
"""
import os
import sys
import time
import random
import subprocess, shlex
import datetime
import yaml
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
import wandb

from core import data
from core import utils
import models
import runners


@record
def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=str, default="other", help="wandb project name"
    )
    parser.add_argument(
        "--wandb_id", type=str, default="", help="wandb username"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yml", help="experiment config"
    )
    parser.add_argument(
        "--train", default=False, action="store_true", help="dont run in debug mode"
    )
    parser.add_argument(
        "--overfit", type=int, default=-1, help="number of examples to overfit to"
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="do not prepend debug to output dirs",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="which GPU to use")
    parser.add_argument(
        "--n_gpu_per_node", type=int, default=1, help="num gpus per node"
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="num nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="rank amongst nodes")
    parser.add_argument(
        "--use_dataparallel",
        default=False,
        action="store_true",
        help="use DataParallel",
    )
    parser.add_argument(
        "--use_ddp",
        default=False,
        action="store_true",
        help="use DistributedDataParallel",
    )
    parser.add_argument(
        "--detect_anomaly", default=False, action="store_true", help="detect nans"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader num workers"
    )
    parser.add_argument(
        "--use_amp",
        default=False,
        action="store_true",
        help="automatic mixed precision",
    )

    opt = parser.parse_args()

    if opt.use_ddp:
        opt.world_size = opt.n_gpu_per_node * opt.n_nodes
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=5400)
        )
        dist.barrier()

    train(opt)

    return


def train(opt):
    config, config_dict = utils.load_config(opt.config, return_dict=True)

    if config.train.home_dir == '':
        config.train.home_dir = os.path.dirname(os.getcwd())

    if opt.use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{rank}"
        assert dist.is_available() and dist.is_initialized(), (
            dist.is_available(),
            dist.is_initialized(),
        )
    elif not opt.no_cuda and torch.cuda.is_available():
        torch.cuda.init()
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        device = f"cuda:{opt.gpu_id}"
    else:
        device = "cpu"

    # Init wandb and logging for process 0
    log_dir = ""
    if not opt.use_ddp or rank == 0:
        if opt.train:
            wandb.init(
                project=opt.project,
                entity=opt.wandb_id,
                job_type="train",
                config=config_dict,
            )
        else:
            config.train.eval_freq = 10
            if not os.path.isdir("wandb_debug"):
                subprocess.run(shlex.split("mkdir wandb_debug"))
            wandb.init(
                project=opt.project,
                entity=opt.wandb_id,
                job_type="debug",
                config=config_dict,
                dir="./wandb_debug",
            )
        print(
            f'Beginning; run name "{wandb.run.name}", run id "{wandb.run.id}", device {device}'
        )
        print(opt)

        if opt.train:
            log_dir = (
                f"{config.train.home_dir}/training_logs/{opt.project}/{wandb.run.name}"
            )
        else:
            log_dir = f"{config.train.home_dir}/training_logs/{opt.project}/debug_{wandb.run.name}"
        if not os.path.isdir(log_dir):
            if not os.path.isdir(log_dir + "results"):
                subprocess.run(["mkdir", "-p", log_dir + "/results"])
            if not os.path.isdir(log_dir + "checkpoints"):
                subprocess.run(["mkdir", "-p", log_dir + "/checkpoints"])
            subprocess.run(
                f"cp -r {os.getcwd()}/*.py {os.getcwd()}/*.sbatch {os.getcwd()}/core/ {os.getcwd()}/configs/ {log_dir}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            raise Exception(f"Logging directory {log_dir} already exists.")

    # Set seeds
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.deterministic = True
    # nonrandom selection of CUDNN convolution, maybe slower
    torch.backends.cudnn.benchmark = False

    # Set up datasets
    def get_dataloader(mode):
        dataset = data.Dataset(
            pdb_path=os.path.join(config.train.home_dir, config.data.pdb_path),
            fixed_size=config.data.fixed_size,
            mode=mode,
            overfit=opt.overfit,
            short_epoch=not opt.train,
            se3_data_augment=config.data.se3_data_augment,
        )
        if mode == "train":
            bs = config.train.batch_size
        elif mode == "eval":
            bs = 1

        if opt.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=opt.world_size, rank=rank
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                num_workers=opt.num_workers,
                pin_memory="cuda" in device,
                shuffle=False,
                sampler=sampler,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                num_workers=opt.num_workers,
                pin_memory="cuda" in device,
                shuffle=True,
            )
        return dataset, dataloader

    dataset, dataloader = get_dataloader("train")
    eval_dataset, eval_dataloader = get_dataloader("eval")

    # Set up model and optimizers
    model = models.Protpardelle(config, device)
    if opt.use_dataparallel:
        model = torch.nn.DataParallel(model)
    if opt.use_ddp:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )
    start_epoch = checkpoint_epoch = config.train.checkpoint[1]
    if checkpoint_epoch > 0:
        training_state = torch.load(
            f"{config.train.checkpoint[0]}/checkpoints/epoch{checkpoint_epoch}_training_state.pth",
            map_location=device,
        )
        model.load_state_dict(training_state["model_state_dict"])
    model.train()
    model.to(device)

    runner = runners.ProtpardelleRunner(
        config,
        model,
        dataset,
        eval_dataloader,
        log_dir,
        device,
    )
    if checkpoint_epoch > 0:
        for _ in range(int(len(dataset) / config.train.batch_size * checkpoint_epoch)):
            runner.scheduler.step()
        runner.optimizer.load_state_dict(training_state["optim_state_dict"])
    runner.train_init()
    total_steps = 0
    if not opt.use_ddp or rank == 0:
        start_time = time.time()
    with torch.autograd.set_detect_anomaly(
        True
    ) if opt.detect_anomaly else nullcontext():
        for epoch in range(start_epoch + 1, config.train.max_epochs + 1):
            if opt.use_ddp:
                dist.barrier()
                dataloader.sampler.set_epoch(epoch)
                eval_dataloader.sampler.set_epoch(epoch)

            for inputs in tqdm(
                dataloader, desc=f"epoch {epoch}/{config.train.max_epochs}"
            ):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["step"] = total_steps
                log_dict = runner.train_step(inputs)
                log_dict["learning_rate"] = runner.scheduler.get_last_lr()[0]
                if not opt.use_ddp or rank == 0:
                    wandb.log(log_dict)
                total_steps += 1

            with torch.no_grad():  # per epoch
                # Run eval and save checkpoint
                if opt.use_ddp:
                    dist.barrier()
                if not opt.use_ddp or rank == 0:
                    wandb.log(runner.epoch_eval(start_time))
                if (
                    epoch % config.train.checkpoint_freq == 0
                    or epoch in config.train.checkpoints
                ):
                    if not opt.use_ddp or rank == 0:
                        runner.model.eval()
                        torch.save(
                            runner.model,
                            f"{log_dir}/checkpoints/epoch{epoch}_model.pth",
                        )
                        torch.save(
                            {
                                "model_state_dict": runner.model.state_dict(),
                                "optim_state_dict": runner.optimizer.state_dict(),
                            },
                            f"{log_dir}/checkpoints/epoch{epoch}_training_state.pth",
                        )

                        runner.model.train()

    if not opt.use_ddp or rank == 0:
        wandb.finish()
        subprocess.run(shlex.split(f"cp -r {wandb.run.dir} {log_dir}"))
        print(
            f'Training finished. (run name "{wandb.run.name}", run id "{wandb.run.id}")'
        )
    if opt.use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
