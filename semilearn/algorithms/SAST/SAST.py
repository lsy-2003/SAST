


import torch
import os
import csv
import itertools
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from collections import Counter
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook, DistAlignHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from semilearn.core.criterions import calculate_ece
from semilearn.core.criterions import GCELoss
from semilearn.core.criterions import CELoss
import torch.nn as nn
from .schedulers import make_scheduler
from .text_enhancer import generate_enhanced_prompts
from .utils import SASTThresholdingHook
from semilearn.nets.clip.clip import tokenize

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

@ALGORITHMS.register("SAST")
class SAST(AlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None):

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

        self.anchor_loader = None

        super().__init__(args, net_builder, tb_log, logger)
        self.label_to_idx = {c: i for i, c in enumerate(self.classnames)}
        self.all_indices = [self.label_to_idx[c] for c in self.classnames if c in self.label_to_idx]

        self.num_data = self.args.ulb_dest_len

        self.ce_loss = nn.CrossEntropyLoss()

        self.guid_gold = {}
        self.queue_pseudo_labels = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_ptr = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)
        self.queue_pseudo_labels_text = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_pseudo_labels_image = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_counts = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)

        self.correctness_ = {}
        self.image_prototypes = None
        self.text_prototypes = None
        self.eval_w_history = []

    def set_optimizer(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if any(k in name for k in ["prompt", "adapter", "projector", "classifier", "head"]):
                param.requires_grad = True
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

        if self.args.v11_attr_path is not None:
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
            self.print_fn("[attr prompt] v11_attr_path didn't exist")
            return super()._load_per_class_prompts(path)

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(SASTThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p, per_class=self.args.per_class, base_thresh=self.args.base_thresh), "MaskingHook")
        self.register_hook(DistAlignHook(num_classes=self.num_classes, align_scale = self.args.align_scale), "DistAlignHook")

        if self.consistency_penalty > 0:
            self.print_fn("Pre-computing text prototypes (from base CLIP) for consistency check...")
            with torch.no_grad():
                text_inputs = torch.cat([tokenize(f"a photo of a {c.replace('_', ' ')}") for c in self.classnames]).to(self.gpu)
                model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

                if hasattr(model_to_use, 'encode_text'):
                    text_features = model_to_use.encode_text(text_inputs)
                    self.text_prototypes = F.normalize(text_features, dim=-1)
                    self.print_fn("Base text prototypes (for I-I consistency) cached.")
                else:
                    self.print_fn("Warning: Model does not have 'encode_text'. Cannot cache base text prototypes.")
                    self.text_prototypes = None


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

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, y_ulb,
                   x_lb_anchor=None, y_lb_anchor=None, epoch=0):
        DA_hook = self.hooks_dict.get('DistAlignHook')
        if DA_hook:
            DA_hook.update(self, current_epoch=epoch)

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()

        target_text_proto = self.text_prototypes.to(self.gpu)

        target_img_proto = None
        if self.image_prototypes is not None:
            target_img_proto = self.image_prototypes.detach().to(self.gpu)

        with self.amp_cm():




            out_lb = self.model(x_lb)
            feats_lb = out_lb['feat']
            logits_lb = out_lb['logits_text']

            if DA_hook:
                logits_lb = DA_hook.adjust_logits(logits_lb, y_lb)

            sup_loss = self.ce_loss(logits_lb.float(), y_lb)

            anchor_loss = torch.tensor(0.0, device=self.gpu)
            if x_lb_anchor is not None:
                x_lb_anchor = x_lb_anchor.cuda(self.gpu)
                y_lb_anchor = y_lb_anchor.cuda(self.gpu)

                out_anchor = self.model(x_lb_anchor)
                logits_anchor = out_anchor['logits_text']

                if DA_hook:
                    logits_anchor = DA_hook.adjust_logits(logits_anchor, y_lb_anchor)

                anchor_loss = self.ce_loss(logits_anchor.float(), y_lb_anchor)

            total_sup_loss = sup_loss + anchor_loss




            out_ulb_w = self.model(x_ulb_w)
            feats_ulb_w = out_ulb_w['feat']
            logits_text_ulb_w = out_ulb_w['logits_text']
            norm_feats_w = F.normalize(feats_ulb_w, dim=-1)


            if target_img_proto is not None:
                img_proto_cast = target_img_proto.to(dtype=norm_feats_w.dtype)
                logits_image_ulb_w = logit_scale * norm_feats_w @ img_proto_cast.t()
                pred_image_w = torch.argmax(logits_image_ulb_w, dim=-1)
            else:
                logits_image_ulb_w = logits_text_ulb_w
                pred_image_w = torch.zeros_like(y_ulb)

            with torch.no_grad():
                pred_text_w = torch.argmax(logits_text_ulb_w, dim=-1)


            with torch.no_grad():
                if self.image_prototypes is not None:
                    rel_text_w, rel_image_w = self.get_switching_scores(idx_ulb, pred_text_w, pred_image_w)
                    fusion_out_w = self._dynamic_fusion_logic(logits_text_ulb_w, logits_image_ulb_w,
                                                                rel_text_w, rel_image_w)
                    logits_ensemble_ulb_w = fusion_out_w['fused_logits']
                    log_w_text = fusion_out_w['w_text']
                else:
                    logits_ensemble_ulb_w = logits_text_ulb_w
                    rel_text_w = torch.ones_like(y_ulb, dtype=torch.float)
                    rel_image_w = torch.zeros_like(y_ulb, dtype=torch.float)
                    log_w_text = torch.ones_like(y_ulb, dtype=torch.float)

            self.update_bank(idx_ulb,
                             torch.argmax(logits_ensemble_ulb_w, dim=-1),
                             torch.max(self.compute_prob(logits_ensemble_ulb_w), dim=-1)[0],
                             logits_ensemble_ulb_w,
                             pred_text_w,
                             pred_image_w)

            easy_mask = self.call_hook("masking", "MaskingHook",
                                                       logits_x_ulb=logits_ensemble_ulb_w,
                                                       sigma=self.sigma,
                                                       softmax_x_ulb=True)




            out_ulb_s = self.model(x_ulb_s)
            feats_ulb_s = out_ulb_s['feat']
            logits_text_ulb_s = out_ulb_s['logits_text']
            norm_feats_s = F.normalize(feats_ulb_s, dim=-1)

            if target_img_proto is not None:
                img_proto_cast = target_img_proto.to(dtype=norm_feats_s.dtype)
                logits_image_ulb_s = logit_scale * norm_feats_s @ img_proto_cast.t()

                logits_ensemble_ulb_s = self._dynamic_fusion_logic(
                    logits_text_ulb_s, logits_image_ulb_s,
                    rel_text_w, rel_image_w)['fused_logits']
            else:
                logits_ensemble_ulb_s = logits_text_ulb_s


            teacher_targets_full_hard = torch.argmax(logits_ensemble_ulb_w.detach(), dim=-1)
            easy_unsup_loss = torch.tensor(0.0, device=self.gpu)

            if DA_hook:
                logits_ensemble_ulb_s = DA_hook.adjust_logits(logits_ensemble_ulb_s, teacher_targets_full_hard)

            if easy_mask.sum() > 0:
                teacher_targets_easy = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                                      logits=logits_ensemble_ulb_w[easy_mask].detach(),
                                                      use_hard_label=self.use_hard_label)
                easy_unsup_loss = self.consistency_loss(
                    logits_ensemble_ulb_s[easy_mask],
                    teacher_targets_easy,
                    'ce'
                )


            total_loss = total_sup_loss + easy_unsup_loss
            ece = calculate_ece(logits_ensemble_ulb_w.detach().cpu(), y_ulb.detach().cpu())

        log_dict = self.process_log_dict(
                sup_loss=total_sup_loss.item(),
                easy_unsup_loss=easy_unsup_loss.item(),
                total_loss=total_loss.item(),
                easy_ratio=easy_mask.float().mean().item(),
                ece=ece,
            )

        return self.process_out_dict(loss=total_loss, feat={'x_lb': feats_lb, 'x_ulb_w': feats_ulb_w, 'x_ulb_s': out_ulb_s['feat']}), log_dict

    @torch.no_grad()
    def initialize_hooks_after_warmup(self):
        self.print_fn("Initializing hooks after warm-up...")

    def train(self):
        self.model.train()

        if self.text_prototypes is None:
            self.print_fn("Initializing Text Prototypes at start of train()...")
            self._compute_text_prototypes()

        if self.text_prototypes is None:
            raise RuntimeError("Failed to compute text_prototypes! Model cannot train/eval.")

        self.refresh_anchors_ssl(epoch=0)

        self.call_hook("before_run")

        anchor_iter = itertools.cycle(self.anchor_loader) if self.anchor_loader else None
        self.initialize_hooks_after_warmup()
        self._initialize_prototypes_with_labeled_data()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            if self.it >= self.num_train_iter: break

            self.call_hook("before_train_epoch")

            if epoch > 0:
                self.refresh_anchors_ssl(epoch=epoch)
                if self.anchor_loader:
                    anchor_iter = itertools.cycle(self.anchor_loader)

            self._update_image_prototypes()

            for data_lb, data_ulb in zip(self.loader_dict["train_lb"], self.loader_dict["train_ulb"]):
                if self.it >= self.num_train_iter: break

                data_anchor = {}
                if anchor_iter:
                    try:
                        data_anchor = next(anchor_iter)
                    except StopIteration:
                        anchor_iter = itertools.cycle(self.anchor_loader)
                        data_anchor = next(anchor_iter)

                self.call_hook("before_train_step")

                input_args = self.process_batch(**data_lb, **data_ulb)

                if data_anchor:
                    args_anchor = self.process_batch(input_args=['x_lb', 'y_lb'], **data_anchor)
                    input_args['x_lb_anchor'] = args_anchor['x_lb']
                    input_args['y_lb_anchor'] = args_anchor['y_lb']

                input_args['epoch'] = epoch

                self.out_dict, self.log_dict = self.train_step(**input_args)

                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def _compute_text_prototypes(self):
        if self.text_prototypes is None:
            self.print_fn("Computing High-Quality Text Prototypes...")


            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

            if hasattr(model_to_use, 'get_learned_text_features'):
                self.print_fn("[MaPLe] Extracting learned text features directly from model.")
                self.text_prototypes = model_to_use.get_learned_text_features().detach()

            else:
                self.print_fn("[Standard CLIP/Wrapper] Computing text prototypes via templates (Ensemble).")

                if hasattr(model_to_use, 'text_encoding_model'):
                    self.print_fn("[Debug] Forcing text_encoding_model to GPU to prevent device mismatch.")
                    model_to_use.text_encoding_model.to(self.gpu)

                with torch.no_grad():
                    dataset = self.dataset_dict['train_lb'] if self.dataset_dict else None
                    if hasattr(dataset, 'dataset'):
                        dataset = dataset.dataset

                    templates = getattr(dataset, 'templates', ["a photo of a {}."])

                    final_text_feats = []
                    for c in self.classnames:
                        texts = [t.format(c.replace('_', ' ')) for t in templates]
                        text_inputs = tokenize(texts).to(self.gpu)
                        text_features = model_to_use.encode_text(text_inputs)
                        text_features = F.normalize(text_features, dim=-1)


                        mean_feature = text_features.mean(dim=0)
                        mean_feature = F.normalize(mean_feature, dim=-1)
                        final_text_feats.append(mean_feature)

                    self.text_prototypes = torch.stack(final_text_feats)


            self.text_prototypes = self.text_prototypes.to(self.gpu)

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
            self.print_fn("SUCCESS: Image prototypes loaded from checkpoint.")
        else:
            self.print_fn("WARNING: No image prototypes found in checkpoint.")

        if 'text_prototypes' in checkpoint and checkpoint['text_prototypes'] is not None:
            self.text_prototypes = checkpoint['text_prototypes'].cuda(self.gpu)
        if 'queue_pseudo_labels_text' in checkpoint:
            self.queue_pseudo_labels_text = checkpoint['queue_pseudo_labels_text'].cuda(self.gpu)
            self.queue_pseudo_labels_image = checkpoint['queue_pseudo_labels_image'].cuda(self.gpu)
        else:
            self.print_fn("V6.0 historical queues not found, re-initializing.")
            self.queue_pseudo_labels_text = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
            self.queue_pseudo_labels_image = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        if 'queue_counts' in checkpoint:
            self.queue_counts = checkpoint['queue_counts'].cuda(self.gpu)
        else:
            self.print_fn("V6.1 queue_counts not found, re-initializing.")
            self.queue_counts = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)
        self.print_fn("additional SAST parameters loaded")
        return checkpoint

    @torch.no_grad()
    def _initialize_prototypes_with_labeled_data(self):
        self.print_fn("Initializing image prototypes with Labeled Data ONLY...")
        self.model.eval()

        lb_feats_by_class = {c: [] for c in range(self.num_classes)}

        temp_lb_loader = DataLoader(
            self.dataset_dict["train_lb"], batch_size=self.args.eval_batch_size,
            shuffle=False, num_workers=self.args.num_workers, drop_last=False
        )

        feat_dim = -1

        for data in temp_lb_loader:
            x_lb, y_lb = data['x_lb'].cuda(self.gpu), data['y_lb'].cuda(self.gpu)

            outputs = self.model(x_lb, only_feat=True)
            feats = outputs['feat'] if isinstance(outputs, dict) else outputs

            if feat_dim == -1 and feats.shape[0] > 0:
                feat_dim = feats.shape[1]

            for i in range(len(y_lb)):
                label = y_lb[i].item()
                lb_feats_by_class[label].append(feats[i].detach())

        if feat_dim == -1 or not any(lb_feats_by_class.values()):
            self.print_fn("Warning: No labeled features found. Cannot initialize prototypes.")
            self.model.train()
            return

        prototypes = torch.zeros(self.num_classes, feat_dim, dtype=torch.float32).cuda(self.gpu)
        class_counts = torch.zeros(self.num_classes).cuda(self.gpu)

        for c in range(self.num_classes):
            if lb_feats_by_class[c]:
                class_feats = torch.stack(lb_feats_by_class[c]).to(self.gpu)
                prototypes[c] = class_feats.sum(dim=0)
                class_counts[c] = len(class_feats)

        prototypes /= class_counts.unsqueeze(1).clamp(min=1)

        self.image_prototypes = F.normalize(prototypes, dim=-1)

        self.print_fn(f"Image prototypes initialized with {len(temp_lb_loader.dataset)} LB samples.")
        self.model.train()

    @torch.no_grad()
    def _update_image_prototypes(self):
        per_class_num = (len(self.dataset_dict["train_lb"]) + len(self.dataset_dict["train_ulb"])) // self.num_classes
        k_min = max(8, per_class_num // 100)
        k_max = max(16, int(per_class_num * self.args.anchor_num_end))
        ramp_epochs = max(1, int(self.epochs * 0.7))
        progress = min(1.0, self.epoch / ramp_epochs)
        current_k = int(k_min + (k_max - k_min) * progress)

        self.print_fn(f"Updating image prototypes (Fusion + Curriculum). Target K={current_k} (Progress: {progress:.1%}). [Force Val Transform]")
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
            lb_feats_by_class = {c: [] for c in range(self.num_classes)}
            lb_class_counts = torch.zeros(self.num_classes, dtype=torch.float32)
            num_labeled_samples = len(self.dataset_dict["train_lb"])

            temp_lb_loader = DataLoader(
                self.dataset_dict["train_lb"], batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, drop_last=False
            )
            for data in temp_lb_loader:
                x_lb, y_lb = data['x_lb'].cuda(self.gpu), data['y_lb'].cuda(self.gpu)
                outputs = self.model(x_lb, only_feat=True)
                feats = outputs['feat']
                for i in range(len(y_lb)):
                    label = y_lb[i].item()
                    lb_feats_by_class[label].append(feats[i].detach())
                    lb_class_counts[label] += 1

            ulb_candidates = []
            if self.dataset_dict.get("train_ulb") is not None:
                temp_ulb_loader = DataLoader(
                    self.dataset_dict["train_ulb"], batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers, drop_last=False
                )
                for data in temp_ulb_loader:
                    idx_ulb = data['idx_ulb']
                    x = data.get('x_ulb_w', data.get('x')).cuda(self.gpu)

                    outputs = self.model(x)
                    feats = outputs['feat']
                    logits_text = outputs['logits_text']


                    if self.image_prototypes is not None:
                        norm_feat = F.normalize(feats, dim=-1)
                        img_proto = self.image_prototypes.to(device=norm_feat.device, dtype=norm_feat.dtype)
                        logits_image = logit_scale * norm_feat @ img_proto.t()

                        pred_text = torch.argmax(logits_text, dim=-1)
                        pred_image = torch.argmax(logits_image, dim=-1)
                        rel_text, rel_image = self.get_switching_scores(idx_ulb.cuda(self.gpu), pred_text, pred_image)

                        fusion_out = self._dynamic_fusion_logic(logits_text, logits_image, rel_text, rel_image)
                        logits = fusion_out['fused_logits']
                    else:
                        logits = logits_text

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
                        ulb_candidates.append({
                            'prob': max_probs[i].item(),
                            'switching_freq': switch_freq_val[i].item(),
                            'idx': idx_ulb[i].item(),
                            'label': preds[i].cpu(),
                            'feat': feats[i].detach().to(torch.float32)
                        })

            if not ulb_candidates:
                self.print_fn("No unlabeled candidates found, skipping prototype update.")
                return
            switching_freqs_series = pd.Series([c['switching_freq'] for c in ulb_candidates])
            def rank_norm_negative(x_series): return (1.0 - x_series.rank(pct=True).fillna(0.5)).values
            norm_stability = rank_norm_negative(switching_freqs_series)
            for i, cand in enumerate(ulb_candidates):
                cand['hybrid_score'] = norm_stability[i]

            candidates_by_class = {c: [] for c in range(self.num_classes)}
            for cand in ulb_candidates:
                candidates_by_class[cand['label'].item()].append(cand)
            for c in range(self.num_classes):
                candidates_by_class[c].sort(key=lambda x: x['hybrid_score'], reverse=True)

            self.print_fn(f"Performing DYNAMIC selection. Target: Top-{current_k} per class.")

            final_prototypes_feats_by_class = {c: lb_feats_by_class[c].copy() for c in range(self.num_classes)}
            selected_ulb_count = 0

            for c in range(self.num_classes):
                num_to_select = current_k

                candidate_pool = candidates_by_class[c]

                if not candidate_pool:
                    continue

                cand_feats_raw_stack = torch.stack([cand['feat'] for cand in candidate_pool]).to(self.gpu).float()

                cand_feats_norm_stack = F.normalize(cand_feats_raw_stack, dim=1).float()

                cand_hybrid_scores = torch.tensor([cand['hybrid_score'] for cand in candidate_pool], device=self.gpu)

                valid_mask = torch.ones(len(candidate_pool), dtype=torch.bool, device=self.gpu)

                initial_feats_gpu = [feat.to(self.gpu) for feat in final_prototypes_feats_by_class[c]]

                if initial_feats_gpu:
                    current_sum_vector = torch.sum(torch.stack(initial_feats_gpu), dim=0)
                    current_count = len(initial_feats_gpu)
                else:
                    feat_dim = cand_feats_raw_stack.shape[1]
                    current_sum_vector = torch.zeros(feat_dim, device=self.gpu)
                    current_count = 0

                text_proto_c = self.text_prototypes[c].to(self.gpu)

                for _ in range(num_to_select):
                    if not valid_mask.any():
                        break

                    current_proto_refined = None
                    if current_count > 0:
                        current_proto_raw = current_sum_vector / current_count
                        current_proto_refined = F.normalize(current_proto_raw, dim=0).float()

                    if current_proto_refined is None:
                        final_scores = cand_hybrid_scores.clone()
                        final_scores[~valid_mask] = -float('inf')
                        best_idx_tensor = torch.argmax(final_scores)
                        best_idx = best_idx_tensor.item()

                    else:
                        consistency_penalties = torch.zeros_like(cand_hybrid_scores)

                        if self.text_prototypes is not None:
                            new_proto_ests = (current_proto_refined.unsqueeze(0) * current_count + cand_feats_norm_stack) / (current_count + 1)
                            new_proto_ests_norm = F.normalize(new_proto_ests, dim=1).float()

                            original_consistency = F.cosine_similarity(current_proto_refined.unsqueeze(0), text_proto_c.unsqueeze(0)).item()
                            new_consistencies = torch.mv(new_proto_ests_norm, text_proto_c)

                            diffs = original_consistency - new_consistencies
                            consistency_penalties = self.consistency_penalty * torch.clamp(diffs, min=0.0)

                        if isinstance(consistency_penalties, torch.Tensor):
                            consistency_penalties = consistency_penalties.to(cand_hybrid_scores.device)

                        final_scores = cand_hybrid_scores - consistency_penalties
                        final_scores[~valid_mask] = -float('inf')

                        best_idx_tensor = torch.argmax(final_scores)
                        best_idx = best_idx_tensor.item()

                    best_feat_raw = cand_feats_raw_stack[best_idx]
                    current_sum_vector += best_feat_raw
                    current_count += 1
                    valid_mask[best_idx] = False

                    best_candidate = candidate_pool[best_idx]
                    final_prototypes_feats_by_class[c].append(best_candidate['feat'])
                    selected_ulb_count += 1

            self.print_fn(f"Prototypes: {num_labeled_samples} LB samples, {selected_ulb_count} ULB samples selected.")

            if not ulb_candidates and not any(lb_feats_by_class.values()):
                self.print_fn("No features available to determine feat_dim. Skipping prototype update.")
                return

            if ulb_candidates:
                feat_dim = ulb_candidates[0]['feat'].shape[0]
            else:
                first_class_with_feats = next(c for c, feats in lb_feats_by_class.items() if feats)
                feat_dim = lb_feats_by_class[first_class_with_feats][0].shape[0]

            prototypes = torch.zeros(self.num_classes, feat_dim, dtype=torch.float32).cuda(self.gpu)
            class_counts = torch.zeros(self.num_classes).cuda(self.gpu)

            for c in range(self.num_classes):
                if final_prototypes_feats_by_class[c]:
                    class_feats = torch.stack(final_prototypes_feats_by_class[c]).to(self.gpu)
                    prototypes[c] = class_feats.sum(dim=0)
                    class_counts[c] = len(class_feats)

            prototypes /= class_counts.unsqueeze(1).clamp(min=1)
            self.image_prototypes = F.normalize(prototypes, dim=-1)

            self.print_fn("Image prototypes updated (using Val-Transform).")

        finally:
            if old_lb_transform is not None:
                lb_dset_root.transform = old_lb_transform
            if old_ulb_transform is not None:
                ulb_dset_root.transform = old_ulb_transform

        self.model.train()

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

        if (self.it <= self.warm_up_iter):
            rel_text = torch.ones_like(rel_text)
            rel_image = torch.zeros_like(rel_image)

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

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        self.model.eval()

        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.apply_shadow()
        else:
            self.print_fn("Warning: EMA hook not found during evaluation. Using current model weights.")

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()

        has_prototypes = (self.image_prototypes is not None)
        use_image_branch = has_prototypes and (self.it > self.args.warm_up_iter or eval_dest != "train_ulb")

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
                    logits_image = logits_text

                probs_text = F.softmax(logits_text.float(), dim=-1)
                rel_text_instant, _ = torch.max(probs_text, dim=-1)

                probs_image = F.softmax(logits_image.float(), dim=-1)
                rel_image_instant, _ = torch.max(probs_image, dim=-1)

                fusion_output = self._dynamic_fusion_logic(
                    logits_text,
                    logits_image,
                    rel_text_instant,
                    rel_image_instant
                )
                if not self.model.training:
                    self.eval_w_history.extend(fusion_output['w_text'].detach().cpu().numpy().flatten().tolist())
                    logits = fusion_output['fused_logits']

                    loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
                    y_true.extend(y.cpu().tolist())
                    y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                    y_logits.append(logits.cpu().numpy())
                    total_loss += loss.item() * num_batch

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        F1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ece = calculate_ece(torch.tensor(y_logits), torch.tensor(y_true))

        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.restore()

        if eval_dest != "test":
            self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/top-1-acc": top1,
            eval_dest + "/balanced_acc": balanced_top1,
            eval_dest + "/precision": precision,
            eval_dest + "/recall": recall,
            eval_dest + "/F1": F1,
            eval_dest + "/ece": ece,
        }

        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits
        return eval_dict

    @torch.no_grad()
    def refresh_anchors_ssl(self, epoch=0):
        per_class_num = (len(self.dataset_dict["train_lb"]) + len(self.dataset_dict["train_ulb"])) // self.num_classes
        k_min = max(8, per_class_num // 100)
        k_max = max(16, int(per_class_num * self.args.anchor_num_end))

        ramp_epochs = max(1, int(self.epochs * 0.7))
        progress = min(1.0, epoch / ramp_epochs)
        current_k = int(k_min + (k_max - k_min) * progress)

        self.print_fn(f"\n[Epoch {epoch}] Refreshing Anchors (SSL)...")
        self.print_fn(f"  > Curriculum: K grows {k_min}->{k_max}. Current K={current_k} (Progress: {progress:.1%})")

        self.model.eval()

        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        logit_scale = model_to_use.logit_scale.exp()

        temp_loader = DataLoader(
            self.dataset_dict["train_ulb"], batch_size=self.args.eval_batch_size,
            shuffle=False, num_workers=self.args.num_workers, drop_last=False
        )

        all_probs = []
        all_indices = []

        for data in temp_loader:
            x = data.get('x_ulb_w', data.get('x')).cuda(self.gpu)
            idx = data['idx_ulb']

            outputs = self.model(x)
            logits_text = outputs['logits_text']


            if self.image_prototypes is not None:
                feat = outputs['feat']
                norm_feat = F.normalize(feat, dim=-1)
                img_proto = self.image_prototypes.to(device=norm_feat.device, dtype=norm_feat.dtype)
                logits_image = logit_scale * norm_feat @ img_proto.t()

                pred_text = torch.argmax(logits_text, dim=-1)
                pred_image = torch.argmax(logits_image, dim=-1)

                rel_text, rel_image = self.get_switching_scores(idx.cuda(self.gpu), pred_text, pred_image)

                fusion_out = self._dynamic_fusion_logic(logits_text, logits_image, rel_text, rel_image)
                logits = fusion_out['fused_logits']
            else:
                logits = logits_text

            probs = self.compute_prob(logits)

            all_probs.append(probs.cpu())
            all_indices.append(idx.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        all_max_probs, all_preds = torch.max(all_probs, dim=1)

        anchor_indices = []
        anchor_labels = []
        selected_count = 0

        class_stats = {}
        all_targets = None
        if hasattr(self.dataset_dict["train_ulb"], 'targets'):
            all_targets = torch.tensor(self.dataset_dict["train_ulb"].targets)

        for c in range(self.num_classes):
            candidate_mask = (all_preds == c)
            count_c = candidate_mask.sum().item()

            if count_c == 0: continue

            c_confs = all_max_probs[candidate_mask]
            c_indices = all_indices[candidate_mask]

            k_c = min(current_k, count_c)
            if k_c > 0:
                topk_vals, topk_rel_idx = torch.topk(c_confs, k=k_c)
                best_indices = c_indices[topk_rel_idx]

                acc = None
                if all_targets is not None:
                    selected_targets = all_targets[best_indices]
                    correct = (selected_targets == c).sum().item()
                    acc = correct / k_c

                anchor_indices.extend(best_indices.tolist())
                anchor_labels.extend([c] * k_c)
                selected_count += k_c
                class_stats[c] = {'count': k_c, 'acc': acc}

        if class_stats:
            valid_accs = [s['acc'] for s in class_stats.values() if s['acc'] is not None]
            avg_acc = sum(valid_accs)/len(valid_accs) if valid_accs else 0.0
            self.print_fn(f"  > Selected {selected_count} Anchors. Avg Accuracy: {avg_acc:.1%}")

        if len(anchor_indices) > 0:
            raw_dataset = self.dataset_dict["train_ulb"]
            if hasattr(raw_dataset, 'dataset'): real_dataset = raw_dataset.dataset
            else: real_dataset = raw_dataset

            anchor_ds = AnchorDataset(real_dataset, anchor_indices, anchor_labels)
            self.anchor_loader = DataLoader(
                anchor_ds, batch_size=self.args.batch_size, shuffle=True,
                num_workers=self.args.num_workers, drop_last=True
            )
        else:
            self.print_fn("Warning: No anchors selected.")

        self.model.train()

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

            SSL_Argument('--v11_attr_path', str, 'Semi-supervised-learning/semilearn/datasets/cv_datasets/attr/cub.json',
                         help='Path to the V11 enhanced attribute JSON file (e.g., cub.json)'),
            SSL_Argument('--align_scale', float, 3),
            SSL_Argument('--anchor_num_end', float, 0.33),
        ]