import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import itertools
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from sklearn.metrics import accuracy_score
from semilearn.core.criterions import calculate_ece
import torch.nn as nn
from .schedulers import make_scheduler
from .text_enhancer import generate_enhanced_prompts
from .utils import SASTThresholdingHook
from semilearn.nets.clip.clip import tokenize
from semilearn.core.utils import get_data_loader


class AnchorDataset(Dataset):
    def __init__(self, base_dataset, indices, pseudo_labels):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, i):
        original_idx = self.indices[i]
        sample = self.base_dataset[original_idx]

        if 'x_ulb_w' in sample:
            img = sample['x_ulb_w']
        elif 'x' in sample:
            img = sample['x']
        else:
            for v in sample.values():
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    img = v
                    break

        return {
            'x_lb': img,
            'y_lb': torch.tensor(self.pseudo_labels[i], dtype=torch.long),
            'idx_lb': original_idx
        }

    def __len__(self):
        return len(self.indices)


@ALGORITHMS.register("SAST_tl")
class SASTTL(AlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):

        self.use_hard_label = args.hard_label
        self.dist_align = args.dist_align
        self.dist_uniform = args.dist_uniform
        self.ema_p = args.ema_p
        self.per_class = args.per_class
        self.queue_size = args.queue_size
        self.warm_up_iter = args.warm_up_iter
        self.model_dir = args.model_dir
        self.base_thresh = args.base_thresh
        self.sigma = args.sigma

        self.consistency_penalty = args.consistency_penalty
        self.image_prototypes = None
        self.text_prototypes = None
        self.anchor_loader = None

        super().__init__(args, net_builder, tb_log, logger)

        self.seen_classnames = self.dataset_dict.get('seen_classes', [])
        self.unseen_classnames = self.dataset_dict.get('unseen_classes', [])

        self.label_to_idx = {c: i for i, c in enumerate(self.classnames)}

        self.all_indices = [self.label_to_idx[c] for c in self.classnames if c in self.label_to_idx]
        self.seen_indices = [self.label_to_idx[c] for c in self.seen_classnames if c in self.label_to_idx]
        self.unseen_indices = [self.label_to_idx[c] for c in self.unseen_classnames if c in self.label_to_idx]

        if len(self.seen_indices) == 0 or len(self.unseen_indices) == 0:
            self.print_fn("[Warning] Seen or Unseen indices list is empty! Check split.json or dataset loading.")

        self.print_fn(f"[SAST-TL] Setup: {len(self.seen_indices)} Seen Classes, {len(self.unseen_indices)} Unseen Classes.")

        self.num_data = self.args.ulb_dest_len
        self.ce_loss = nn.CrossEntropyLoss().cuda(self.gpu)

        self.guid_gold = {}

        self.queue_pseudo_labels = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_ptr = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)
        self.queue_pseudo_labels_text = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_pseudo_labels_image = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_counts = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)

    def set_optimizer(self):
        self.print_fn("[SAST-TL] Freezing backbone parameters to prevent overfitting on Seen classes...")

        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if any(k in name for k in ["prompt", "adapter", "projector", "classifier"]):
                param.requires_grad = True

        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = False
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'logit_scale'):
            self.model.module.logit_scale.requires_grad = False

        self.optimizer, _ = super().set_optimizer()

        class ConfigAdapter:
            def __init__(self, args, num_train_iter):
                self.SCHEDULER = "cosine"
                self.WARMUP_EPOCHS = 1
                self.ITER_PER_EPOCH = args.warm_up_iter
                self.EPOCHS = 50

        scheduler_config = ConfigAdapter(self.args, self.num_train_iter)
        self.scheduler = make_scheduler(self.optimizer, scheduler_config)

        return self.optimizer, self.scheduler

    def _load_per_class_prompts(self, path):
        if self.args.v11_attr_path is not None and os.path.exists(self.args.v11_attr_path):
            self.print_fn("[attr prompt] prompt augmentation enabled.")

            enhanced_prompt_map = generate_enhanced_prompts(
                attr_path=self.args.v11_attr_path,
                classnames=self.classnames
            )
            if enhanced_prompt_map:
                try:
                    first_key = list(enhanced_prompt_map.keys())[0]
                    first_val = enhanced_prompt_map[first_key]
                    if len(first_val) > 100:
                        first_val = first_val[:100] + "..."
                    self.print_fn(f"[attr prompt validation] prompts have successfully generated.")
                    self.print_fn(f"[attr prompt] example: '{first_key}' -> '{first_val}'")
                except Exception as e:
                    self.print_fn(f"[attr prompt validation] cannot print prompt: {e}")
            self.print_fn("[attr prompt] prompts have been injected")
            return enhanced_prompt_map

        else:
            if self.args.v11_attr_path:
                self.print_fn(f"[attr prompt] Warning: v11_attr_path set to {self.args.v11_attr_path} but not found.")
            return super()._load_per_class_prompts(path)

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(SASTThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p, per_class=self.args.per_class, base_thresh=self.args.base_thresh), "MaskingHook")
        self.register_hook(DistAlignHook(num_classes=self.num_classes, align_scale=self.args.align_scale), "DistAlignHook")
        if self.consistency_penalty > 0:
            self._compute_text_prototypes()

    @torch.no_grad()
    def refresh_anchors_tl(self, epoch=0):
        per_class_num = (len(self.dataset_dict["train_lb"]) + len(self.dataset_dict["train_ulb"])) // self.num_classes
        k_min = max(8, per_class_num // 100)
        k_max = max(16, int(per_class_num * self.args.anchor_num_end))
        ramp_epochs = max(1, int(self.epochs * 0.7))
        progress = min(1.0, epoch / ramp_epochs)

        current_k = int(k_min + (k_max - k_min) * progress)

        self.print_fn(f"\n[Epoch {epoch}] Refreshing Anchors for UNSEEN classes...")
        self.print_fn(f"  > Curriculum Strategy: K grows from {k_min} to {k_max}. Current K = {current_k} (Progress: {progress:.1%})")

        self.model.eval()

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()

        temp_loader = DataLoader(
            self.dataset_dict["train_ulb"],
            batch_size=self.args.eval_batch_size,
            shuffle=False, num_workers=self.args.num_workers, drop_last=False
        )

        all_probs = []
        all_indices = []

        for data in temp_loader:
            x = data.get('x_ulb_w', data.get('x')).cuda(self.gpu)
            idx = data['idx_ulb'].cuda(self.gpu)

            outputs = self.model(x)
            logits_text = outputs['logits_text']

            if self.image_prototypes is not None:
                feat = outputs['feat']
                norm_feat = F.normalize(feat, dim=-1)
                img_proto = self.image_prototypes.to(device=norm_feat.device, dtype=norm_feat.dtype)
                logits_image = logit_scale * norm_feat @ img_proto.t()

                pred_text = torch.argmax(logits_text, dim=-1)
                pred_image = torch.argmax(logits_image, dim=-1)

                rel_text, rel_image = self.get_switching_scores(idx, pred_text, pred_image)

                fusion_out = self._dynamic_fusion_logic(logits_text, logits_image, rel_text, rel_image)
                logits = fusion_out['fused_logits']
            else:
                logits = logits_text

            logits[:, self.seen_indices] = -1e4
            probs = self.compute_prob(logits)

            all_probs.append(probs.cpu())
            all_indices.append(idx.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        all_max_probs, all_preds = torch.max(all_probs, dim=1)

        anchor_indices = []
        anchor_labels = []
        selected_count = 0
        class_counts = {}

        class_anchor_stats = {}
        if hasattr(self.dataset_dict["train_ulb"], 'targets'):
            all_targets = torch.tensor(self.dataset_dict["train_ulb"].targets)
        else:
            all_targets = None

        for c in self.unseen_indices:
            candidate_mask = (all_preds == c)

            count_c = candidate_mask.sum().item()
            if count_c == 0:
                self.print_fn(f"  Warning: Unseen Class {c} has 0 candidates. Skipping.")
                continue

            c_confs = all_max_probs[candidate_mask]
            c_indices = all_indices[candidate_mask]

            k_c = min(current_k, count_c)

            if k_c > 0:
                topk_vals, topk_rel_idx = torch.topk(c_confs, k=k_c)
                best_indices = c_indices[topk_rel_idx]

                accuracy = None
                if all_targets is not None:
                    selected_targets = all_targets[best_indices]
                    correct = (selected_targets == c).sum().item()
                    accuracy = correct / k_c if k_c > 0 else 0.0

                anchor_indices.extend(best_indices.tolist())
                anchor_labels.extend([c] * k_c)

                selected_count += k_c
                class_counts[c] = k_c

                class_anchor_stats[c] = {
                    'count': k_c,
                    'accuracy': accuracy
                }

        self.print_fn(f"\n  > Per-class Anchor Statistics:")
        for c, stats in class_anchor_stats.items():
            acc_str = f"{stats['accuracy']:.1%}" if stats['accuracy'] is not None else "N/A"
            self.print_fn(f"    - Class {c}: {stats['count']} anchors, Accuracy: {acc_str}")

        if class_anchor_stats:
            total_anchors = sum(stats['count'] for stats in class_anchor_stats.values())
            if all_targets is not None:
                avg_accuracy = sum(stats['accuracy'] for stats in class_anchor_stats.values() if stats['accuracy'] is not None) / len(class_anchor_stats)
                self.print_fn(f"  > Total: {total_anchors} anchors, Average Accuracy: {avg_accuracy:.1%}")

        if len(anchor_indices) > 0:
            raw_dataset = self.dataset_dict["train_ulb"]
            if hasattr(raw_dataset, 'dataset'):
                real_dataset = raw_dataset.dataset
            else:
                real_dataset = raw_dataset

            anchor_ds = AnchorDataset(real_dataset, anchor_indices, anchor_labels)

            self.anchor_loader = DataLoader(
                anchor_ds,
                batch_size=min(len(anchor_ds), self.args.batch_size),
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True if len(anchor_ds) >= self.args.batch_size else False
            )
        else:
            self.print_fn("CRITICAL WARNING: No anchors selected! Model might collapse.")

        self.model.train()

    def train_step(self, idx_ulb, x_ulb_w, x_ulb_s, y_ulb,
                   x_lb_seen=None, y_lb_seen=None,
                   x_lb_anchor=None, y_lb_anchor=None,
                   epoch=0):

        DA_hook = self.hooks_dict.get('DistAlignHook')
        if DA_hook:
            DA_hook.update(self, current_epoch=epoch)

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()

        target_img_proto = None
        if self.image_prototypes is not None:
            target_img_proto = self.image_prototypes.detach().to(self.gpu)

        with self.amp_cm():
            total_sup_loss = torch.tensor(0.0).cuda(self.gpu)

            if x_lb_seen is not None:
                x_lb_seen = x_lb_seen.cuda(self.gpu)
                y_lb_seen = y_lb_seen.cuda(self.gpu)

                out_seen = self.model(x_lb_seen)
                logits_seen = out_seen['logits_text']

                if DA_hook:
                    logits_seen = DA_hook.adjust_logits(logits_seen, y_lb_seen)

                loss_seen = self.ce_loss(logits_seen.float(), y_lb_seen)
                total_sup_loss += loss_seen

            if x_lb_anchor is not None:
                x_lb_anchor = x_lb_anchor.cuda(self.gpu)
                y_lb_anchor = y_lb_anchor.cuda(self.gpu)

                out_anchor = self.model(x_lb_anchor)
                logits_anchor = out_anchor['logits_text']

                if DA_hook:
                    logits_anchor = DA_hook.adjust_logits(logits_anchor, y_lb_anchor)

                loss_anchor = self.ce_loss(logits_anchor.float(), y_lb_anchor)
                total_sup_loss += loss_anchor

            with torch.no_grad():
                out_ulb_w = self.model(x_ulb_w)
                logits_text_ulb_w = out_ulb_w['logits_text']
                norm_feats_w = F.normalize(out_ulb_w['feat'], dim=-1)

                if target_img_proto is not None:
                    img_proto_cast = target_img_proto.to(dtype=norm_feats_w.dtype)
                    logits_image_ulb_w = logit_scale * norm_feats_w @ img_proto_cast.t()
                    pred_image_w = torch.argmax(logits_image_ulb_w, dim=-1)
                else:
                    logits_image_ulb_w = logits_text_ulb_w
                    pred_image_w = torch.zeros_like(y_ulb)

                pred_text_w = torch.argmax(logits_text_ulb_w, dim=-1)

                if self.image_prototypes is not None:
                    rel_text_w, rel_image_w = self.get_switching_scores(idx_ulb, pred_text_w, pred_image_w)
                    fusion_out_w = self._dynamic_fusion_logic(logits_text_ulb_w, logits_image_ulb_w, rel_text_w, rel_image_w)
                    logits_ensemble_ulb_w = fusion_out_w['fused_logits']
                else:
                    logits_ensemble_ulb_w = logits_text_ulb_w
                    rel_text_w, rel_image_w = None, None

                logits_ensemble_ulb_w[:, self.seen_indices] = -1e4
                teacher_targets = torch.argmax(logits_ensemble_ulb_w, dim=-1)

                self.update_bank(idx_ulb, teacher_targets,
                                 torch.max(self.compute_prob(logits_ensemble_ulb_w), dim=-1)[0],
                                 logits_ensemble_ulb_w, pred_text_w, pred_image_w)

            easy_mask = self.call_hook("masking", "MaskingHook",
                                       logits_x_ulb=logits_ensemble_ulb_w,
                                       sigma=self.sigma, softmax_x_ulb=True)

            out_ulb_s = self.model(x_ulb_s)
            logits_text_ulb_s = out_ulb_s['logits_text']

            if target_img_proto is not None:
                norm_feats_s = F.normalize(out_ulb_s['feat'], dim=-1)
                img_proto_detached = target_img_proto.detach().to(dtype=norm_feats_s.dtype)
                logits_image_ulb_s = logit_scale * norm_feats_s @ img_proto_detached.t()

                fusion_out_s = self._dynamic_fusion_logic(
                    logits_text_ulb_s, logits_image_ulb_s,
                    rel_text_w, rel_image_w)
                logits_ensemble_ulb_s = fusion_out_s['fused_logits']
            else:
                logits_ensemble_ulb_s = logits_text_ulb_s

            easy_unsup_loss = torch.tensor(0.0, device=self.gpu)

            if DA_hook:
                logits_ensemble_ulb_s = DA_hook.adjust_logits(logits_ensemble_ulb_s, teacher_targets)

            if easy_mask.sum() > 0:
                targets_easy = teacher_targets[easy_mask]
                easy_unsup_loss = self.consistency_loss(logits_ensemble_ulb_s[easy_mask], targets_easy, 'ce')

            total_loss = total_sup_loss + easy_unsup_loss
            ece = calculate_ece(logits_ensemble_ulb_w.detach().cpu(), y_ulb.detach().cpu())

            log_dict = self.process_log_dict(sup_loss=total_sup_loss.item(),
                                             easy_unsup_loss=easy_unsup_loss.item(),
                                             total_loss=total_loss.item(),
                                             easy_ratio=easy_mask.float().mean().item(),
                                             ece=ece)

            return self.process_out_dict(loss=total_loss, feat={'x_ulb_w': out_ulb_w['feat']}), log_dict

    @torch.no_grad()
    def _update_image_prototypes(self):
        per_class_num = (len(self.dataset_dict["train_lb"]) + len(self.dataset_dict["train_ulb"])) // self.num_classes
        k_min = max(8, per_class_num // 100)
        k_max = max(16, int(per_class_num * self.args.anchor_num_end))
        ramp_epochs = max(1, int(self.epochs * 0.7))
        progress = min(1.0, self.epoch / ramp_epochs)
        current_k = int(k_min + (k_max - k_min) * progress)

        self.print_fn(f"Updating image prototypes (Seen:GT, Unseen:UL-Logic). Target K={current_k}. [Force Val Transform]")
        self.model.eval()

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()

        val_transform = self.dataset_dict['eval'].transform

        def get_root_dataset(dset):
            if hasattr(dset, 'dataset'):
                return get_root_dataset(dset.dataset)
            return dset

        lb_dset_root = get_root_dataset(self.dataset_dict["train_lb"])
        ulb_dset_root = get_root_dataset(self.dataset_dict["train_ulb"])

        old_lb_transform = getattr(lb_dset_root, 'transform', None)
        old_ulb_transform = getattr(ulb_dset_root, 'transform', None)

        if old_lb_transform is not None:
            lb_dset_root.transform = val_transform
        if old_ulb_transform is not None:
            ulb_dset_root.transform = val_transform

        try:
            feats_by_class = {c: [] for c in range(self.num_classes)}

            temp_lb_loader = DataLoader(
                self.dataset_dict["train_lb"], batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, drop_last=False
            )
            for data in temp_lb_loader:
                x = data['x_lb'].cuda(self.gpu)
                y = data['y_lb'].cuda(self.gpu)
                outputs = self.model(x, only_feat=True)
                feats = outputs['feat'] if isinstance(outputs, dict) else outputs
                for i in range(len(y)):
                    label = y[i].item()
                    if label in self.seen_indices:
                        feats_by_class[label].append(feats[i].detach())

            ulb_candidates = []

            temp_ulb_loader = DataLoader(
                self.dataset_dict["train_ulb"], batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, drop_last=False
            )

            for data in temp_ulb_loader:
                x = data.get('x_ulb_w', data.get('x')).cuda(self.gpu)
                idx_ulb = data['idx_ulb']

                outputs = self.model(x)
                feats_ulb = outputs['feat']
                logits_text = outputs['logits_text']

                if self.image_prototypes is not None:
                    norm_feat = F.normalize(feats_ulb, dim=-1)
                    img_proto = self.image_prototypes.to(device=norm_feat.device, dtype=norm_feat.dtype)
                    logits_image = logit_scale * norm_feat @ img_proto.t()

                    pred_text = torch.argmax(logits_text, dim=-1)
                    pred_image = torch.argmax(logits_image, dim=-1)
                    rel_text, rel_image = self.get_switching_scores(idx_ulb.cuda(self.gpu), pred_text, pred_image)

                    fusion_out = self._dynamic_fusion_logic(logits_text, logits_image, rel_text, rel_image)
                    logits = fusion_out['fused_logits']
                else:
                    logits = logits_text

                logits[:, self.seen_indices] = -1e4

                probs = self.compute_prob(logits)
                max_probs, preds = torch.max(probs, dim=-1)

                idx_gpu = idx_ulb.cuda(self.gpu)
                history_targets = self.queue_pseudo_labels[idx_gpu]
                counts = self.queue_counts[idx_gpu]
                history_len = counts.float() + 1e-9
                arange_tensor = torch.arange(self.queue_size, device=idx_gpu.device)
                valid_mask = arange_tensor.unsqueeze(0) < counts.unsqueeze(1)

                switches = ((history_targets != preds.unsqueeze(-1)) & valid_mask).float().sum(dim=-1)
                switch_freq_val = (switches / history_len)
                switch_freq_val = torch.where(counts < 2, torch.tensor(0.5, device=idx_gpu.device), switch_freq_val)

                for i in range(len(idx_ulb)):
                    label_item = preds[i].item()
                    if label_item in self.unseen_indices:
                        ulb_candidates.append({
                            'prob': max_probs[i].item(),
                            'switching_freq': switch_freq_val[i].item(),
                            'idx': idx_ulb[i].item(),
                            'label': preds[i].cpu(),
                            'feat': feats_ulb[i].detach().to(torch.float32)
                        })

            if ulb_candidates:
                switching_freqs_series = pd.Series([c['switching_freq'] for c in ulb_candidates])
                def rank_norm_negative(x_series): return (1.0 - x_series.rank(pct=True).fillna(0.5)).values
                norm_stability = rank_norm_negative(switching_freqs_series)
                for i, cand in enumerate(ulb_candidates):
                    cand['hybrid_score'] = norm_stability[i]

            candidates_by_class = {c: [] for c in self.unseen_indices}
            for cand in ulb_candidates:
                candidates_by_class[cand['label'].item()].append(cand)
            for c in self.unseen_indices:
                candidates_by_class[c].sort(key=lambda x: x['hybrid_score'], reverse=True)

            unseen_selected_count = 0

            text_proto_all = None
            if self.text_prototypes is not None:
                text_proto_all = self.text_prototypes.to(self.gpu).float()

            for c in self.unseen_indices:
                candidate_pool = candidates_by_class[c]
                if not candidate_pool: continue

                cand_feats_raw_stack = torch.stack([cand['feat'] for cand in candidate_pool]).to(self.gpu).float()

                target_device = cand_feats_raw_stack.device
                cand_feats_norm_stack = F.normalize(cand_feats_raw_stack, dim=1)
                cand_hybrid_scores = torch.tensor([cand['hybrid_score'] for cand in candidate_pool], device=target_device, dtype=torch.float32)
                valid_mask = torch.ones(len(candidate_pool), dtype=torch.bool, device=target_device)

                current_sum_vector = torch.zeros(cand_feats_raw_stack.shape[1], device=target_device, dtype=torch.float32)
                current_count = 0

                text_proto_c = text_proto_all[c].to(target_device) if text_proto_all is not None else None

                for _ in range(current_k):
                    if not valid_mask.any(): break

                    current_proto_refined = None
                    if current_count > 0:
                        current_proto_raw = current_sum_vector / current_count
                        current_proto_refined = F.normalize(current_proto_raw, dim=0)

                    if current_proto_refined is None:
                        final_scores = cand_hybrid_scores.clone()
                        final_scores[~valid_mask] = -float('inf')

                        if text_proto_c is not None:
                            sim_to_text = torch.mv(cand_feats_norm_stack, text_proto_c)
                            cold_start_penalty = self.consistency_penalty * (1.0 - sim_to_text)
                            if isinstance(cold_start_penalty, torch.Tensor):
                                cold_start_penalty = cold_start_penalty.to(final_scores.device)
                            final_scores -= cold_start_penalty

                        best_idx_tensor = torch.argmax(final_scores)
                    else:
                        consistency_penalties = torch.zeros_like(cand_hybrid_scores)
                        if text_proto_c is not None:
                            new_proto_ests = (current_proto_refined.unsqueeze(0) * current_count + cand_feats_norm_stack) / (current_count + 1)
                            new_proto_ests_norm = F.normalize(new_proto_ests, dim=1)
                            original_consistency = F.cosine_similarity(current_proto_refined.unsqueeze(0), text_proto_c.unsqueeze(0)).item()
                            new_consistencies = torch.mv(new_proto_ests_norm, text_proto_c)
                            diffs = original_consistency - new_consistencies
                            consistency_penalties = self.consistency_penalty * torch.clamp(diffs, min=0.0)

                        if isinstance(consistency_penalties, torch.Tensor):
                            consistency_penalties = consistency_penalties.to(target_device)

                        final_scores = cand_hybrid_scores - consistency_penalties
                        final_scores[~valid_mask] = -float('inf')
                        best_idx_tensor = torch.argmax(final_scores)

                    best_idx = best_idx_tensor.item()
                    best_feat_raw = cand_feats_raw_stack[best_idx]
                    current_sum_vector += best_feat_raw
                    current_count += 1
                    valid_mask[best_idx] = False

                    best_candidate = candidate_pool[best_idx]
                    feats_by_class[c].append(best_candidate['feat'])
                    unseen_selected_count += 1

            has_any_feat = any(len(l) > 0 for l in feats_by_class.values())
            if not has_any_feat:
                self.print_fn("Warning: No features collected for any class.")
                return

            feat_dim = 0
            for c in range(self.num_classes):
                if feats_by_class[c]:
                    feat_dim = feats_by_class[c][0].shape[0]
                    break

            prototypes = torch.zeros(self.num_classes, feat_dim, dtype=torch.float32).cuda(self.gpu)
            class_counts = torch.zeros(self.num_classes).cuda(self.gpu)

            for c in range(self.num_classes):
                if feats_by_class[c]:
                    class_feats = torch.stack(feats_by_class[c]).to(self.gpu)
                    prototypes[c] = class_feats.sum(dim=0)
                    class_counts[c] = len(class_feats)
                elif c in self.seen_indices:
                    if self.text_prototypes is not None:
                        prototypes[c] = self.text_prototypes[c].to(self.gpu)
                        class_counts[c] = 1
                elif c in self.unseen_indices:
                    if self.text_prototypes is not None:
                        prototypes[c] = self.text_prototypes[c].to(self.gpu)
                        class_counts[c] = 1

            prototypes /= class_counts.unsqueeze(1).clamp(min=1)
            self.image_prototypes = F.normalize(prototypes, dim=-1)

            self.print_fn(f"Image prototypes updated (using Val-Transform). Unseen selected: {unseen_selected_count}.")

        finally:
            if old_lb_transform is not None:
                lb_dset_root.transform = old_lb_transform
            if old_ulb_transform is not None:
                ulb_dset_root.transform = old_ulb_transform

        self.model.train()

    def train(self):
        """
        Main training loop for Transductive Zero-Shot Learning (TRZSL).
        Strategy:
        1. Epoch length is determined by the Labeled (Seen) dataset size.
        2. Unlabeled (Unseen) data and Anchors are cycled to match the length of Labeled data.
        """
        self.model.train()

        if self.text_prototypes is None:
            self._compute_text_prototypes()

        self.refresh_anchors_tl(epoch=0)

        self.call_hook("before_run")

        anchor_iter = itertools.cycle(self.anchor_loader)

        ulb_loader = self.loader_dict["train_ulb"]
        ulb_iter = itertools.cycle(ulb_loader)

        lb_loader = self.loader_dict["train_lb"]
        self.initialize_hooks_after_warmup()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            if self.it >= self.num_train_iter: break

            self.call_hook("before_train_epoch")

            if epoch > 0:
                self.refresh_anchors_tl(epoch=epoch)
                anchor_iter = itertools.cycle(self.anchor_loader)

            self._update_image_prototypes()

            for data_lb_seen in lb_loader:
                if self.it >= self.num_train_iter: break

                try:
                    data_lb_anchor = next(anchor_iter)
                except StopIteration:
                    anchor_iter = itertools.cycle(self.anchor_loader)
                    data_lb_anchor = next(anchor_iter)

                try:
                    data_ulb = next(ulb_iter)
                except StopIteration:
                    ulb_iter = itertools.cycle(ulb_loader)
                    data_ulb = next(ulb_iter)

                self.call_hook("before_train_step")

                args_seen = self.process_batch(input_args=['x_lb', 'y_lb'], **data_lb_seen)
                args_anchor = self.process_batch(input_args=['x_lb', 'y_lb'], **data_lb_anchor)
                args_ulb = self.process_batch(**data_ulb)

                self.out_dict, self.log_dict = self.train_step(
                    idx_ulb=args_ulb['idx_ulb'],
                    x_ulb_w=args_ulb['x_ulb_w'],
                    x_ulb_s=args_ulb['x_ulb_s'],
                    y_ulb=args_ulb['y_ulb'],

                    x_lb_seen=args_seen['x_lb'],
                    y_lb_seen=args_seen['y_lb'],

                    x_lb_anchor=args_anchor['x_lb'],
                    y_lb_anchor=args_anchor['y_lb'],
                    epoch=epoch
                )

                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def _compute_text_prototypes(self):
        if self.text_prototypes is None:
            self.print_fn("Computing High-Quality Text Prototypes...")
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

            if hasattr(model_to_use, 'get_learned_text_features'):
                self.print_fn("[MaPLe] Extracting learned text features.")
                self.text_prototypes = model_to_use.get_learned_text_features().detach()
            else:
                self.print_fn("[CLIP] Computing text prototypes via templates.")
                with torch.no_grad():
                    dataset = self.dataset_dict['train_lb'] if self.dataset_dict else None
                    if hasattr(dataset, 'dataset'): dataset = dataset.dataset
                    templates = getattr(dataset, 'templates', ["a photo of a {}."])

                    final_text_feats = []
                    for c in self.classnames:
                        texts = [t.format(c.replace('_', ' ')) for t in templates]
                        text_inputs = tokenize(texts).to(self.gpu)
                        text_features = model_to_use.encode_text(text_inputs)
                        mean_feature = F.normalize(text_features, dim=-1).mean(dim=0)
                        final_text_feats.append(F.normalize(mean_feature, dim=-1))

                    self.text_prototypes = torch.stack(final_text_feats)

            self.text_prototypes = self.text_prototypes.to(self.gpu)

    @torch.no_grad()
    def update_bank(self, ulb_idxs, pseudo_labels, y_max_probs, logits, text_labels, image_labels):

        if self.distributed and self.world_size > 1:
            ulb_idxs = concat_all_gather(ulb_idxs)
            pseudo_labels = concat_all_gather(pseudo_labels)
            logits = concat_all_gather(logits)
            text_labels = concat_all_gather(text_labels)
            image_labels = concat_all_gather(image_labels)

        ptr = self.queue_ptr[ulb_idxs]

        self.queue_pseudo_labels[ulb_idxs, ptr] = pseudo_labels
        self.queue_pseudo_labels_text[ulb_idxs, ptr] = text_labels
        self.queue_pseudo_labels_image[ulb_idxs, ptr] = image_labels
        self.queue_counts[ulb_idxs] = (self.queue_counts[ulb_idxs] + 1).clamp(max=self.queue_size)
        self.queue_ptr[ulb_idxs] = (ptr + 1) % self.queue_size

    @torch.no_grad()
    def get_switching_scores(self, ulb_idxs, current_text_pred, current_image_pred):

        text_history = self.queue_pseudo_labels_text[ulb_idxs]
        image_history = self.queue_pseudo_labels_image[ulb_idxs]
        counts = self.queue_counts[ulb_idxs]

        history_len = counts.float() + 1e-9

        arange_tensor = torch.arange(self.queue_size, device=ulb_idxs.device)

        valid_history_mask = arange_tensor.unsqueeze(0) < counts.unsqueeze(1)

        text_switches = ((text_history != current_text_pred.unsqueeze(-1)) & valid_history_mask).float().sum(dim=-1)
        image_switches = ((image_history != current_image_pred.unsqueeze(-1)) & valid_history_mask).float().sum(dim=-1)

        text_switch_freq = text_switches / history_len
        image_switch_freq = image_switches / history_len

        rel_text = (1.0 - text_switch_freq).clamp(min=0.0)
        rel_image = (1.0 - image_switch_freq).clamp(min=0.0)

        neutral_reliability = 0.5
        rel_text = torch.where(counts < 2, neutral_reliability, rel_text)
        rel_image = torch.where(counts < 2, neutral_reliability, rel_image)

        return rel_text, rel_image

    def _dynamic_fusion_logic(self, logits_text, logits_image, rel_text_hist, rel_image_hist):

        with torch.no_grad():
            mean_text = logits_text.mean(dim=-1, keepdim=True)
            std_text = logits_text.std(dim=-1, keepdim=True).clamp(min=1e-6)
            mean_image = logits_image.mean(dim=-1, keepdim=True)
            std_image = logits_image.std(dim=-1, keepdim=True).clamp(min=1e-6)

        logits_text_norm = (logits_text - mean_text) / std_text
        logits_image_norm = (logits_image - mean_image) / std_image
        with torch.no_grad():
            rel_text_base = rel_text_hist.float()
            rel_image_base = rel_image_hist.float()

            rel_text_logit = torch.log(rel_text_base + 1e-9)
            rel_image_logit = torch.log(rel_image_base + 1e-9)

            reliability_logits = torch.stack([rel_text_logit, rel_image_logit], dim=1)

            final_weights = F.softmax(reliability_logits, dim=1)

            w_text = final_weights[:, 0].unsqueeze(1)
            w_image = final_weights[:, 1].unsqueeze(1)

            log_w_text = w_text.squeeze(1)
            log_w_image = w_image.squeeze(1)

        fused_logits_norm = (w_text * logits_text_norm + w_image * logits_image_norm)

        fused_logits_rescaled = fused_logits_norm * (std_text)

        return {
            'fused_logits': fused_logits_rescaled,
            'w_text': log_w_text,
            'w_image': log_w_image
        }

    @torch.no_grad()
    def evaluate(self, eval_dest="test", out_key="logits", return_logits=False):
        self.model.eval()
        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        y_true_all = []
        preds_collection = {"text": [], "image": [], "fused": []}

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()
        use_image_branch = (self.image_prototypes is not None)

        total_loss = 0.0
        total_num = 0.0

        with torch.no_grad():
            for data in eval_loader:
                x = data["x_lb"]
                y = data["y_lb"]

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = self.model(x)

                logits_text = outputs['logits_text']
                feats = outputs['feat']
                norm_feats = F.normalize(feats, dim=-1)

                if use_image_branch:
                    target_img_proto = self.image_prototypes.to(device=norm_feats.device, dtype=norm_feats.dtype)
                    logits_image = logit_scale * norm_feats @ target_img_proto.t()
                else:
                    logits_image = logits_text.clone()

                probs_text = F.softmax(logits_text.float(), dim=-1)
                rel_text_instant, _ = torch.max(probs_text, dim=-1)

                probs_image = F.softmax(logits_image.float(), dim=-1)
                rel_image_instant, _ = torch.max(probs_image, dim=-1)

                fusion_output = self._dynamic_fusion_logic(
                    logits_text, logits_image, rel_text_instant, rel_image_instant
                )
                logits_fused = fusion_output['fused_logits']

                loss = F.cross_entropy(logits_fused, y, reduction="mean", ignore_index=-1)
                total_loss += loss.item() * num_batch

                y_true_all.extend(y.cpu().tolist())
                preds_collection["text"].extend(torch.argmax(logits_text, dim=-1).cpu().tolist())
                preds_collection["image"].extend(torch.argmax(logits_image, dim=-1).cpu().tolist())
                preds_collection["fused"].extend(torch.argmax(logits_fused, dim=-1).cpu().tolist())

        y_true_np = np.array(y_true_all)
        seen_mask = np.isin(y_true_np, self.seen_indices)
        unseen_mask = np.isin(y_true_np, self.unseen_indices)

        def calc_metrics(y_true, y_pred, mask=None):
            if mask is not None:
                if np.sum(mask) == 0: return 0.0
                y_t = y_true[mask]
                y_p = np.array(y_pred)[mask]
            else:
                y_t, y_p = y_true, y_pred

            acc = accuracy_score(y_t, y_p)
            return acc

        self.print_fn(f"\n[{eval_dest.upper()} EVAL] Epoch {self.epoch}")
        self.print_fn(f"{'Branch':<10} | {'Type':<8} | {'Acc':<8} | {'H-Mean':<8}")
        self.print_fn("-" * 55)

        results_summary = {}

        for branch_name in ["text", "image", "fused"]:
            preds = preds_collection[branch_name]
            ov_acc = calc_metrics(y_true_np, preds, None)
            s_acc = calc_metrics(y_true_np, preds, seen_mask)
            u_acc = calc_metrics(y_true_np, preds, unseen_mask)

            h_mean = (2 * s_acc * u_acc) / (s_acc + u_acc + 1e-9)

            self.print_fn(f"{branch_name:<10} | {'All':<8} | {ov_acc:.4f}   | -")
            self.print_fn(f"{branch_name:<10} | {'Seen':<8} | {s_acc:.4f}   | -")
            self.print_fn(f"{branch_name:<10} | {'Unseen':<8} | {u_acc:.4f}   | {h_mean:.4f}")
            self.print_fn("-" * 55)

            if branch_name == "fused":
                results_summary = {"seen_acc": s_acc, "unseen_acc": u_acc, "h_mean": h_mean, "overall_acc": ov_acc}

        if np.sum(unseen_mask) > 0:
            self.print_fn("\n[Diagnostic] Text Branch Error Analysis on UNSEEN samples:")

            y_true_unseen = y_true_np[unseen_mask]
            y_pred_text_unseen = np.array(preds_collection["text"])[unseen_mask]
            total_unseen = len(y_true_unseen)

            correct = (y_true_unseen == y_pred_text_unseen)
            num_correct = np.sum(correct)

            incorrect_mask = ~correct
            wrong_preds = y_pred_text_unseen[incorrect_mask]

            if len(wrong_preds) > 0:
                is_seen_pred = np.isin(wrong_preds, self.seen_indices)
                num_leak_to_seen = np.sum(is_seen_pred)

                num_confuse_unseen = len(wrong_preds) - num_leak_to_seen
            else:
                num_leak_to_seen = 0
                num_confuse_unseen = 0

            self.print_fn(f"  > Total Unseen Samples: {total_unseen}")
            self.print_fn(f"  > Correct:              {num_correct:4d} ({num_correct/total_unseen:.2%})")
            self.print_fn(f"  > Wrong -> Predicted as SEEN:   {num_leak_to_seen:4d} ({num_leak_to_seen/total_unseen:.2%}) [Bias Issue]")
            self.print_fn(f"  > Wrong -> Predicted as UNSEEN: {num_confuse_unseen:4d} ({num_confuse_unseen/total_unseen:.2%}) [Confusion Issue]")

            if num_leak_to_seen > num_confuse_unseen:
                self.print_fn("  >> CONCLUSION: Model suffers primarily from SEEN CLASS BIAS.")
            else:
                self.print_fn("  >> CONCLUSION: Model suffers primarily from UNSEEN CLASS CONFUSION.")

        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.restore()
        self.model.train()

        return {
            eval_dest + "/loss": total_loss / (total_num + 1e-9),
            eval_dest + "/top-1-acc": results_summary["h_mean"],
            eval_dest + "/seen_acc": results_summary["seen_acc"],
            eval_dest + "/unseen_acc": results_summary["unseen_acc"],
            eval_dest + "/h_mean": results_summary["h_mean"],
        }

    @torch.no_grad()
    def initialize_hooks_after_warmup(self):
        self.print_fn("Initializing hooks after warm-up...")

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['queue_ptr'] = self.queue_ptr.cpu()
        save_dict['queue_pseudo_labels'] = self.queue_pseudo_labels.cpu()
        save_dict['guid_gold'] = self.guid_gold
        save_dict['queue_pseudo_labels_text'] = self.queue_pseudo_labels_text.cpu()
        save_dict['queue_pseudo_labels_image'] = self.queue_pseudo_labels_image.cpu()
        save_dict['queue_counts'] = self.queue_counts.cpu()
        if self.image_prototypes is not None:
            save_dict['image_prototypes'] = self.image_prototypes.cpu()
        if self.text_prototypes is not None:
            save_dict['text_prototypes'] = self.text_prototypes.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_ptr = checkpoint['queue_ptr'].cuda(self.gpu)
        self.queue_pseudo_labels = checkpoint['queue_pseudo_labels'].cuda(self.gpu)
        self.guid_gold = checkpoint['guid_gold']
        if 'image_prototypes' in checkpoint and checkpoint['image_prototypes'] is not None:
            self.image_prototypes = checkpoint['image_prototypes'].cuda(self.gpu)
        if 'text_prototypes' in checkpoint and checkpoint['text_prototypes'] is not None:
            self.text_prototypes = checkpoint['text_prototypes'].cuda(self.gpu)
        if 'queue_pseudo_labels_text' in checkpoint:
            self.queue_pseudo_labels_text = checkpoint['queue_pseudo_labels_text'].cuda(self.gpu)
            self.queue_pseudo_labels_image = checkpoint['queue_pseudo_labels_image'].cuda(self.gpu)
        else:
            self.queue_pseudo_labels_text = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
            self.queue_pseudo_labels_image = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        if 'queue_counts' in checkpoint:
            self.queue_counts = checkpoint['queue_counts'].cuda(self.gpu)
        else:
            self.queue_counts = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--ema_p', float, 0.99),
            SSL_Argument('--per_class', str2bool, False),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--queue_size', int, 50),
            SSL_Argument('--model_dir', str, None),
            SSL_Argument('--warm_up_iter', int, 2048),

            SSL_Argument('--base_thresh', float, 0.95),
            SSL_Argument('--sigma', float, 1),

            SSL_Argument('--consistency_penalty', float, 0.1),

            SSL_Argument('--v11_attr_path', str, 'Semi-supervised-learning/semilearn/datasets/cv_datasets/attr/cub.json'),

            SSL_Argument('--align_scale', float, 3),
            SSL_Argument('--anchor_num_end', float, 0.5),
        ]

    def set_data_loader(self):
        if self.dataset_dict is None:
            return

        self.print_fn("Create train and test data loaders (TL Optimized: 1:1 Batch Size)")
        loader_dict = {}

        loader_dict["train_lb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_lb"],
            self.args.batch_size * len(self.dataset_dict["train_lb"]) // len(self.dataset_dict["train_ulb"]) + 1,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict["train_ulb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_ulb"],
            self.args.batch_size,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict['eval'] = get_data_loader(
            self.args,
            self.dataset_dict['eval'],
            self.args.eval_batch_size,
            data_sampler=None,
            num_workers=self.args.num_workers,
            drop_last=False)

        if self.dataset_dict["test"] is not None:
            loader_dict["test"] = get_data_loader(
                self.args,
                self.dataset_dict["test"],
                self.args.eval_batch_size,
                data_sampler=None,
                num_workers=self.args.num_workers,
                drop_last=False,
            )

        return loader_dict