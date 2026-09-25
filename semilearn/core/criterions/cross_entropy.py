



import torch
import torch.nn as nn

from torch.nn import functional as F


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:

        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)

def gce_loss(logits, targets, q=0.7, reduction='none'):
    """
    Generalized Cross Entropy Loss.
    
    Paper: Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels (NeurIPS 2018)
    
    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        q: hyper-parameter q in (0, 1]. The closer to 0, the more it behaves like CE; 
           the closer to 1, the more like MAE. Default is 0.7.
        reduction: 'none' | 'mean' | 'sum'
    """
    pred = F.softmax(logits, dim=-1)

    if logits.shape == targets.shape:
        loss_per_class = (1 - pred ** q) / q
        loss = torch.sum(targets * loss_per_class, dim=1)

    else:
        pred_k = torch.gather(pred, 1, targets.view(-1, 1)).squeeze(1)
        loss = (1 - pred_k ** q) / q

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, mask):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q

        if mask is not None:

            loss = loss * mask

        return loss.mean()