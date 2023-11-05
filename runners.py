"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Model runners for training.
"""
import functools
import os
import time

from einops import repeat, rearrange
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from core import data

from core import utils
from core import protein_mpnn
from core import residue_constants
import diffusion
import evaluation
import models
import modules


def mean(x):
    if len(x) == 0:
        return 0.0
    return sum(x) / len(x)


class ProtpardelleRunner(object):
    def __init__(
        self,
        config,
        model,
        train_dataset,
        eval_dataloader,
        save_dir,
        device,
        scaler=None,
        load_eval_models=True,
    ):
        super().__init__()
        self.config = config

        if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            self.model = model.module
        else:
            self.model = model
        self.forward = model

        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()
        self.dataset = train_dataset
        self.eval_dataloader = eval_dataloader
        self.save_dir = save_dir
        self.device = device
        self.scaler = scaler

        self.next_eval_time = config.train.eval_freq

        if load_eval_models:
            self.mpnn_model = protein_mpnn.get_mpnn_model(
                path_to_model_weights=config.train.home_dir
                + "/ProteinMPNN/vanilla_model_weights",
                device=device,
            )
            self.struct_pred_model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1"
            ).to(device)
            self.struct_pred_model.esm = self.struct_pred_model.esm.half()
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

        self.sigma_data = self.model.sigma_data

    def get_optimizer_and_scheduler(self):
        params_to_train = [(n, p) for n, p in self.model.named_parameters()]
        if self.model.task == "seqdes":
            params_to_train = [
                (n, p) for n, p in params_to_train if "struct_model" not in n
            ]
        params_to_train = [p for n, p in params_to_train]
        optimizer = torch.optim.Adam(
            params_to_train,
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = modules.LinearWarmupCosineDecay(
            optimizer,
            self.config.train.lr,
            warmup_steps=self.config.train.warmup_steps,
            decay_steps=self.config.train.decay_steps,
        )
        return optimizer, scheduler

    def train_init(self):
        print(f"total params: {sum(p.numel() for p in self.model.parameters())}")
        print(
            f"trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def compute_loss(
        self,
        inputs,
        time=None,
        seq_time=None,
        is_training=False,
        return_aux=False,
        compute_sigma_data=False,
    ):
        seq_mask = inputs["seq_mask"]
        coords = inputs["coords_in"]
        aatype = inputs["aatype"]
        aatype_oh = F.one_hot(aatype, self.config.data.n_aatype_tokens).float()
        atom_mask = inputs["atom_mask"]
        conformer = inputs["conformer"]
        device = coords.device
        bs = coords.shape[0]
        noising_atom_mask = None

        struct_crop_cond = None
        if self.config.train.crop_conditional:
            coords, crop_cond_mask = data.make_crop_cond_mask_and_recenter_coords(
                atom_mask, coords, **vars(self.config.train.crop_cond)
            )
            struct_crop_cond = coords * crop_cond_mask[..., None]

        # Estimate sigma_data: run with a large batch size; this block will compute and terminate. a bit hacky
        if compute_sigma_data:
            masked_coords = data.get_masked_coords_array(coords, atom_mask)
            print("for", coords.shape[0], "examples:")
            print("mean:", masked_coords.mean())
            print("std:", masked_coords.std())
            raise Exception("Done computing sigma_data.")

        # Noise data
        if time is None:
            time = torch.rand(bs).clamp(min=1e-9, max=1 - 1e-9).to(device)
        noise_level = self.model.training_noise_schedule(time)

        noised_coords = diffusion.noise_coords(
            coords,
            noise_level,
            # noising_mask=noising_atom_mask,
            dummy_fill_masked_atoms=self.config.model.dummy_fill_masked_atoms,
            atom_mask=atom_mask,
        )

        if self.config.model.task == "backbone":
            bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
            bb_atom_mask = utils.atom37_mask_from_aatype(bb_seq, seq_mask)
            noised_coords *= bb_atom_mask[..., None]

        # Forward pass
        model_inputs = {
            "noisy_coords": noised_coords,
            "noise_level": noise_level,
            "seq_mask": seq_mask,
            "residue_index": inputs["residue_index"],
            "struct_crop_cond": struct_crop_cond,
            "conformer": conformer
        }
        forward_fn = self.forward if is_training else self.model
        struct_self_cond, seq_self_cond = None, None

        if hasattr(self.config.model, "debug_mpnn") and self.config.model.debug_mpnn:
            if (
                np.random.uniform() < self.config.train.self_cond_train_prob
                and self.config.model.mpnn_model.use_self_conditioning
            ):
                with torch.no_grad():
                    _, _, _, seq_self_cond = forward_fn(
                        **model_inputs,
                    )
            _, pred_seq_logprobs, _, _ = forward_fn(
                **model_inputs,
                seq_self_cond=seq_self_cond,
            )
        else:

            if is_training:
                conformer_cond_prob = 0.9
            else:
                conformer_cond_prob = 1.0



            if np.random.uniform() < self.config.train.self_cond_train_prob:
                with torch.no_grad():
                    _, _, struct_self_cond, seq_self_cond = forward_fn(**model_inputs, conformer_cond_prob=conformer_cond_prob)
            denoised_coords, pred_seq_logprobs, _, _ = forward_fn(
                **model_inputs,
                struct_self_cond=struct_self_cond,
                seq_self_cond=seq_self_cond,
                conformer_cond_prob=conformer_cond_prob
            )

        loss = 0.0
        aux = {}

        # Compute structure loss
        if self.config.model.task in ["backbone", "allatom", "codesign"]:
            if self.config.model.task == "backbone":
                struct_loss_mask = torch.ones_like(coords) * bb_atom_mask[..., None]
            else:
                struct_loss_mask = torch.ones_like(coords) * atom_mask[..., None]
            loss_weight = (noise_level**2 + self.sigma_data**2) / (
                (noise_level * self.sigma_data) ** 2
            )
            struct_loss = utils.masked_mse(
                coords, denoised_coords, struct_loss_mask, loss_weight
            )
            loss += struct_loss
            aux["struct_loss"] = struct_loss.mean().detach().cpu().item()

        # Compute mpnn loss
        if self.config.model.task in ["seqdes", "codesign"]:
            alpha = self.config.model.mpnn_model.label_smoothing
            target_oh = (1 - alpha) * aatype_oh + alpha / self.model.n_tokens
            seq_loss_mask = seq_mask
            mpnn_loss = utils.masked_cross_entropy(
                pred_seq_logprobs, target_oh, seq_loss_mask
            )
            loss += mpnn_loss
            aux["mpnn_loss"] = mpnn_loss.mean().detach().cpu().item()

        aux["train_loss"] = loss.mean().detach().cpu().item()
        if return_aux:
            return loss.mean(), aux
        return loss.mean()

    def train_step(self, inputs):
        self.model.zero_grad()

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss, log_dict = self.compute_loss(
                    inputs, is_training=True, return_aux=True
                )
                self.scaler.scale(loss).backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip_val
                    )
                except Exception:
                    pass
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            loss, log_dict = self.compute_loss(
                inputs, is_training=True, return_aux=True
            )
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip_val
                )
            except Exception:
                pass
            self.optimizer.step()
            self.scheduler.step()

        return log_dict

    def epoch_eval(self, start_time):
        log_dict = {}
        self.model.eval()

        time_elapsed = time.time() - start_time
        if time_elapsed > self.next_eval_time:
            with torch.no_grad():
                gly_idx = residue_constants.restype_order["G"]

                # # Validation set metrics
                # noise_level_ts = self.config.train.eval_loss_t
                # eval_metrics = {}
                # eval_losses = {}
                # # Assumes batchsize 1
                # for inputs in self.eval_dataloader:
                #     if np.random.uniform() > self.config.train.subsample_eval_set:
                #         continue
                #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
                #     for t in noise_level_ts:
                #         batched_t = t * torch.ones(inputs["coords_in"].shape[0]).to(
                #             self.model.device
                #         )
                #         eval_losses.setdefault(str(t), []).append(
                #             self.compute_loss(inputs, time=batched_t, is_training=False)
                #             .detach()
                #             .cpu()
                #             .item()
                #         )

                # for k, v in eval_metrics.items():
                #     eval_metrics[k] = mean(v)

                # for k, v in eval_losses.items():
                #     eval_metrics[f"{k}_eval_loss"] = mean(v)
                # eval_metrics["avg_eval_loss"] = mean(
                #     [eval_metrics[f"{str(t)}_eval_loss"] for t in noise_level_ts]
                # )
                eval_metrics = {}

                # # Sampling metrics
                sampling_metrics = {}

                if self.model.task == "allatom":
                    # try:
                    #     aa_metrics, aa_aux = evaluation.evaluate_allatom_generation(
                    #         self.model,
                    #         n_samples=self.config.train.n_eval_samples,
                    #         sample_length_range=self.config.train.sample_length_range,
                    #         struct_pred_model=self.struct_pred_model,
                    #         tokenizer=self.tokenizer,
                    #     )
                    #     sampling_metrics = {**sampling_metrics, **aa_metrics}

                    #     # Save some samples and the ESMFold predictions
                    #     (
                    #         trimmed_coords,
                    #         trimmed_aatype,
                    #         trimmed_atom_mask,
                    #         pred_coords,
                    #         best_idx,
                    #     ) = aa_aux
                    #     rand_samp_idx = (
                    #         best_idx + 1
                    #     ) % self.config.train.n_eval_samples
                    #     for i, idx in enumerate([best_idx, rand_samp_idx]):
                    #         if torch.isnan(trimmed_coords[idx]).sum() == 0:
                    #             utils.write_coords_to_pdb(
                    #                 trimmed_coords[idx],
                    #                 f"{self.save_dir}/results/time{int(time_elapsed)+i}_allatom_samp{idx}.pdb",
                    #                 batched=False,
                    #                 aatype=trimmed_aatype[idx],
                    #                 atom_mask=trimmed_atom_mask[idx],
                    #                 conect=True,
                    #             )
                    #         if torch.isnan(pred_coords[idx]).sum() == 0:
                    #             utils.write_coords_to_pdb(
                    #                 pred_coords[idx],
                    #                 f"{self.save_dir}/results/time{int(time_elapsed)+i}_allatom_pred{idx}.pdb",
                    #                 batched=False,
                    #                 aatype=trimmed_aatype[idx],
                    #                 atom_mask=trimmed_atom_mask[idx],
                    #                 conect=True,
                    #             )

                    # except RuntimeError as e:
                    #     print(f"Skipping allatom eval, due to error {e}...")
                    pass


                if self.model.task == "backbone":

                    print("-------EVALUATION: NORMAL------>")

                    bb_metrics, bb_aux = evaluation.evaluate_backbone_generation(
                        self.model,
                        n_samples=self.config.train.n_eval_samples,
                        sample_length_range=self.config.train.sample_length_range,
                        mpnn_model=self.mpnn_model,
                        struct_pred_model=self.struct_pred_model,
                        tokenizer=self.tokenizer,
                    )
                    sampling_metrics = {**sampling_metrics, **bb_metrics}
                    # Save some samples
                    (
                        sampled_coords,
                        seq_mask,
                        best_idx,
                        pred_coords,
                        designed_seqs,
                    ) = bb_aux
                    rand_samp_idx = (best_idx + 1) % self.config.train.n_eval_samples
                    for i, idx in enumerate([best_idx, rand_samp_idx]):
                        if torch.isnan(sampled_coords[idx]).sum() == 0:
                            gly_aatype = seq_mask[idx, seq_mask[idx] == 1] * gly_idx
                            utils.write_coords_to_pdb(
                                sampled_coords[idx],
                                f"{self.save_dir}/results/time{int(time_elapsed)+i}_bb_samp{idx}.pdb",
                                batched=False,
                                aatype=gly_aatype,
                            )
                        if torch.isnan(pred_coords[idx]).sum() == 0:
                            designed_seq = utils.seq_to_aatype(designed_seqs[idx])
                            utils.write_coords_to_pdb(
                                pred_coords[idx],
                                f"{self.save_dir}/results/time{int(time_elapsed)+i}_bb_pred{idx}.pdb",
                                batched=False,
                                aatype=designed_seq,
                            )



                    print("-----Evaluation: Conformations - Train")
                    conformer_tm_scores_train = []
                    conformer_rmsd_scores_train = []
                    for inputs in list(self.eval_dataloader_train)[:10]: 
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        truth_path = f"{self.eval_dataloader_train.dataset.pdb_path}/dompdb/{int(inputs['dataset_id'])}"
                        truth = utils.load_feats_from_pdb(truth_path)
                        sampled_coords, seq_mask, tm_score, rmsd_score = evaluation.evaluate_backbone_conformer(inputs, self.model, truth)
                        conformer_tm_scores_train.append(tm_score)
                        conformer_rmsd_scores_train.append(rmsd_score)
                        evaluation.conformer_save_pdbs(self.save_dir, inputs['dataset_id'], "train", sampled_coords, seq_mask, tm_score, rmsd_score, truth_path, time_elapsed, gly_idx)
                    sampling_metrics['conformer_tm_score_mean_train'] = np.mean(conformer_tm_scores_train)
                    sampling_metrics['conformer_rmsd_score_mean_train'] = np.mean(conformer_rmsd_scores_train)


                    print("------Evaluation: Conformations - Test")
                    conformer_tm_scores_eval = []
                    conformer_rmsd_scores_eval = []
                    for inputs in list(self.eval_dataloader)[:10]: 
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        truth_path = f"{self.eval_dataloader.dataset.pdb_path}/dompdb/{int(inputs['dataset_id'])}"
                        truth = utils.load_feats_from_pdb(truth_path)
                        sampled_coords, seq_mask, tm_score, rmsd_score = evaluation.evaluate_backbone_conformer(inputs, self.model, truth)
                        conformer_tm_scores_eval.append(tm_score)
                        conformer_rmsd_scores_eval.append(rmsd_score)
                        evaluation.conformer_save_pdbs(self.save_dir, inputs['dataset_id'], "eval", sampled_coords, seq_mask, tm_score, rmsd_score, truth_path, time_elapsed, gly_idx)
                    sampling_metrics['conformer_tm_score_mean_eval'] = np.mean(conformer_tm_scores_eval)
                    sampling_metrics['conformer_rmsd_score_mean_eval'] = np.mean(conformer_rmsd_scores_eval)



                log_dict = {**eval_metrics, **sampling_metrics}

            self.next_eval_time = time.time() - start_time + self.config.train.eval_freq

        self.model.train()
        return log_dict
