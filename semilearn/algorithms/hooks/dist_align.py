


import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import concat_all_gather


class DistAlignEMAHook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        self.m = 0.9

        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)

        self.p_model = None

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):

        self.update_p(algorithm, probs_x_ulb, probs_x_lb)


        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned


    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):

        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.update_p_target:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']


        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes, )) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes, ))/ self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target

class DistAlignQueueHook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, queue_length=128, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.queue_length = queue_length


        self.p_target_ptr, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target.mean(dim=0))

        self.p_model = torch.zeros(self.queue_length, self.num_classes, dtype=torch.float)
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        """
        Args:
            algorithm: base algorithm
            probs_x_ulb: unlabeled batch probs
        """


        self.update_p(algorithm, probs_x_ulb, probs_x_lb)


        probs_x_ulb_aligned = probs_x_ulb * (self.p_target.mean(dim=0) + 1e-6) / (self.p_model.mean(dim=0) + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):


        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
            if self.p_target_ptr is not None:
                self.p_target_ptr = self.p_target_ptr.to(probs_x_ulb.device)

        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
            self.p_model_ptr = self.p_model_ptr.to(probs_x_ulb.device)


        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.p_target_ptr is not None:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr)
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length

        if self.p_target_ptr is not None:
            assert probs_x_lb is not None
            p_target_ptr = int(self.p_target_ptr)
            self.p_target[p_target_ptr] = probs_x_lb.mean(dim=0)
            self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']


        p_target_ptr = None
        if p_target_type == 'uniform':
            p_target = torch.ones(self.queue_length, self.num_classes, dtype=torch.float) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.zeros((self.queue_length, self.num_classes), dtype=torch.float)
            p_target_ptr =  torch.zeros(1, dtype=torch.long)
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
            p_target = p_target.unsqueeze(0).repeat((self.queue_length, 1))

        return p_target_ptr, p_target

class DistAlignHook(Hook):

    def __init__(self, num_classes, align_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.align_scale = align_scale

        self.register_buffer('align_matrix', torch.zeros(num_classes, num_classes))
        self.register_buffer('align_mask', 1.0 - torch.eye(num_classes))

        self.last_update_epoch = -1

    def register_buffer(self, name, tensor):
        if not hasattr(self, name):
            setattr(self, name, tensor)

    @torch.no_grad()
    def update(self, algorithm, current_epoch):

        if current_epoch == self.last_update_epoch and self.align_matrix.sum() != 0:
            return

        class_ratio = torch.ones(self.num_classes, device=self.align_matrix.device)

        if hasattr(algorithm, 'queue_pseudo_labels'):

            all_labels = algorithm.queue_pseudo_labels.view(-1)
            counts = torch.bincount(all_labels, minlength=self.num_classes).float()

            if algorithm.all_indices:
                unseen_counts = counts[algorithm.all_indices]
                max_val = unseen_counts.max()
                if max_val > 0:
                    for idx in algorithm.all_indices:
                        class_ratio[idx] = counts[idx] / max_val
                else:
                    class_ratio[algorithm.all_indices] = 0.0

        diff = (1.0 - class_ratio.min()) * pow(self.align_scale, 2)

        ratio_term = (1.0 - class_ratio) / (1.0 + class_ratio + 1e-6)


        self.align_matrix = (ratio_term.unsqueeze(1) * diff)

        if self.align_mask.device != self.align_matrix.device:
            self.align_mask = self.align_mask.to(self.align_matrix.device)

        self.align_matrix = self.align_matrix * self.align_mask

        self.last_update_epoch = current_epoch

    def adjust_logits(self, logits, labels):
        if self.align_matrix.device != logits.device:
            self.align_matrix = self.align_matrix.to(logits.device)
            self.align_mask = self.align_mask.to(logits.device)

        batch_margins = self.align_matrix[labels]

        return logits + batch_margins