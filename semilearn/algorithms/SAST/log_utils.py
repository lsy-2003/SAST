import torch
import torch.nn.functional as F
import pandas as pd
import os
from semilearn.core.hooks import Hook
from collections import defaultdict
import numpy as np

def probability_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def probability_kurtosis(probs):
    mean = probs.mean(dim=-1, keepdim=True)
    std = probs.std(dim=-1, keepdim=True)
    z = (probs - mean) / (std + 1e-9)
    return torch.mean(z**4, dim=-1)

def topk_prob_ratio(probs, k=3):
    topk_probs, _ = torch.topk(probs, k, dim=-1)
    rest_probs = probs.sum(dim=-1) - topk_probs.sum(dim=-1)
    return topk_probs.sum(dim=-1) / (rest_probs + 1e-9)


def logits_variance(logits):

    return torch.var(logits, dim=-1)

def logits_kurtosis(logits):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    z = (logits - mean) / (std + 1e-9)
    return torch.mean(z**4, dim=-1)

def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=-1)


def nn_consistency(features, prototypes, predicted_labels):
    if prototypes is None or features is None:
        return torch.zeros(len(features), device=features.device), torch.zeros(len(features), device=features.device)

    prototypes_casted = prototypes.to(features.dtype)
    similarities = F.normalize(features) @ F.normalize(prototypes_casted).T

    similarities = similarities.to(predicted_labels.device)

    intra_sim = similarities.gather(1, predicted_labels.unsqueeze(1)).squeeze(1)

    mask = torch.ones_like(similarities, dtype=torch.bool)
    mask.scatter_(1, predicted_labels.unsqueeze(1), False)

    inter_sims_all = torch.where(mask, similarities, torch.tensor(0.0, device=similarities.device))
    num_inter_classes = prototypes.shape[0] - 1

    if num_inter_classes > 0:
        inter_sim = inter_sims_all.sum(dim=-1) / num_inter_classes
    else:
        inter_sim = torch.zeros_like(intra_sim)

    return intra_sim, inter_sim

def feature_norm(features):
    return torch.norm(features, p=2, dim=-1)

def feature_sparsity(features):
    sorted_features, _ = torch.sort(features.abs(), dim=-1)
    n = features.shape[-1]
    arange_tensor = torch.arange(1, n+1, device=features.device, dtype=features.dtype)
    sum_sorted = torch.sum(sorted_features, dim=-1, keepdim=True)

    sum_sorted = torch.where(sum_sorted == 0, torch.tensor(1e-9, device=sum_sorted.device), sum_sorted)

    gini_term = torch.sum(sorted_features * arange_tensor, dim=-1, keepdim=True) / (n * sum_sorted)
    gini = 1 - 2 * gini_term.squeeze(-1)
    return gini

def prediction_stability(current_probs, historical_probs_list):
    if not historical_probs_list:
        return torch.ones(current_probs.shape[0], device=current_probs.device)

    stability_scores = []

    for hist_probs in historical_probs_list:
        hist_probs = hist_probs.to(current_probs.device)
        m = 0.5 * (current_probs + hist_probs)
        m_log = torch.log(m + 1e-9)

        kl_p_m = (current_probs * (torch.log(current_probs + 1e-9) - m_log)).sum(dim=-1)
        kl_q_m = (hist_probs * (torch.log(hist_probs + 1e-9) - m_log)).sum(dim=-1)

        js_div = 0.5 * (kl_p_m + kl_q_m)

        stability_scores.append(1.0 - js_div.clamp(min=0.0))

    return torch.mean(torch.stack(stability_scores), dim=0)

def class_switching_frequency(current_pred, historical_preds_list):
    if not historical_preds_list:
        return torch.tensor(0.0, device=current_pred.device)

    history_len = len(historical_preds_list)
    historical_preds_tensor = torch.stack(historical_preds_list).to(current_pred.device)

    switches = (historical_preds_tensor != current_pred).float().sum(dim=0)

    return switches / history_len


def expert_agreement(text_pred, image_pred):
    agreements = (text_pred == image_pred).float()

    return agreements

def confidence_variance(text_conf, image_conf):
    confs = [text_conf, image_conf]

    conf_tensor = torch.stack(confs, dim=0)
    return torch.var(conf_tensor, dim=0)


class EnhancedAnalysisHook(Hook):
    def __init__(self, model_dir, history_size=5):
        super().__init__()
        self.model_dir = model_dir
        self.history_size = history_size
        self.log_data = []

        self.text_probs_history = defaultdict(list)
        self.text_preds_history = defaultdict(list)
        self.image_probs_history = defaultdict(list)
        self.image_preds_history = defaultdict(list)

    @torch.no_grad()
    def log_metrics(self, alg, iteration, idx_ulb, y_ulb,
                      text_logits, image_logits,
                      feats_ulb_w, sample_type, w_text):
        text_prototypes = alg.text_prototypes
        image_prototypes = alg.image_prototypes

        mean_text = text_logits.mean(dim=-1, keepdim=True)
        std_text = text_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)
        mean_image = image_logits.mean(dim=-1, keepdim=True)
        std_image = image_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)

        logits_text_scaled = text_logits.float()
        logits_image_scaled = ((image_logits.float() - mean_image) / std_image) * std_text + mean_text

        text_probs = F.softmax(logits_text_scaled, dim=-1)
        image_probs = F.softmax(logits_image_scaled, dim=-1)

        text_pred = torch.argmax(text_probs, dim=-1)
        image_pred = torch.argmax(image_probs, dim=-1)

        text_conf, _ = torch.max(text_probs, dim=-1)
        image_conf, _ = torch.max(image_probs, dim=-1)

        m_text_entropy = probability_entropy(text_probs)
        m_image_entropy = probability_entropy(image_probs)
        m_text_kurtosis = probability_kurtosis(text_probs)
        m_image_kurtosis = probability_kurtosis(image_probs)
        m_text_topk_ratio = topk_prob_ratio(text_probs, k=3)
        m_image_topk_ratio = topk_prob_ratio(image_probs, k=3)
        m_text_logits_var = logits_variance(logits_text_scaled)
        m_image_logits_var = logits_variance(logits_image_scaled)
        m_text_energy = energy_score(logits_text_scaled)
        m_image_energy = energy_score(logits_image_scaled)

        m_feat_norm = feature_norm(feats_ulb_w)
        m_feat_sparsity = feature_sparsity(feats_ulb_w)

        m_nn_intra_sim_text, m_nn_inter_sim_text = nn_consistency(feats_ulb_w, text_prototypes, text_pred)
        m_nn_intra_sim_image, m_nn_inter_sim_image = nn_consistency(feats_ulb_w, image_prototypes, image_pred)

        idx_ulb_cpu = idx_ulb.cpu().tolist()
        text_probs_cpu = text_probs.cpu()
        text_pred_cpu = text_pred.cpu()
        image_probs_cpu = image_probs.cpu()
        image_pred_cpu = image_pred.cpu()

        m_text_stability_list = []
        m_text_switching_list = []
        m_image_stability_list = []
        m_image_switching_list = []

        for i, guid in enumerate(idx_ulb_cpu):

            hist_text_probs = self.text_probs_history[guid]
            hist_text_preds = self.text_preds_history[guid]
            m_text_stability_list.append(prediction_stability(text_probs_cpu[i].unsqueeze(0), hist_text_probs).item())
            m_text_switching_list.append(class_switching_frequency(text_pred_cpu[i].unsqueeze(0), hist_text_preds).item())
            hist_text_probs.append(text_probs_cpu[i]); hist_text_preds.append(text_pred_cpu[i])
            if len(hist_text_probs) > self.history_size:
                hist_text_probs.pop(0); hist_text_preds.pop(0)

            hist_image_probs = self.image_probs_history[guid]
            hist_image_preds = self.image_preds_history[guid]
            m_image_stability_list.append(prediction_stability(image_probs_cpu[i].unsqueeze(0), hist_image_probs).item())
            m_image_switching_list.append(class_switching_frequency(image_pred_cpu[i].unsqueeze(0), hist_image_preds).item())
            hist_image_probs.append(image_probs_cpu[i]); hist_image_preds.append(image_pred_cpu[i])
            if len(hist_image_probs) > self.history_size:
                hist_image_probs.pop(0); hist_image_preds.pop(0)

        for i in range(len(idx_ulb_cpu)):
            log_entry = {
                'iteration': iteration,
                'guid': idx_ulb_cpu[i],
                'true_label': y_ulb[i].item(),
                'sample_type': sample_type[i].item(),
                'w_text': w_text[i].item(),

                'text_correct': (text_pred[i] == y_ulb[i]).item(),
                'image_correct': (image_pred[i] == y_ulb[i]).item(),

                'text_conf': text_conf[i].item(),
                'text_entropy': m_text_entropy[i].item(),
                'text_kurtosis': m_text_kurtosis[i].item(),
                'text_topk_ratio': m_text_topk_ratio[i].item(),
                'text_logits_var': m_text_logits_var[i].item(),
                'text_energy': m_text_energy[i].item(),
                'text_nn_intra_sim': m_nn_intra_sim_text[i].item(),
                'text_nn_inter_sim': m_nn_inter_sim_text[i].item(),
                'text_stability': m_text_stability_list[i],
                'text_switching_freq': m_text_switching_list[i],

                'text_pred': text_pred[i].item(),
                'image_pred': image_pred[i].item(),

                'image_conf': image_conf[i].item(),
                'image_entropy': m_image_entropy[i].item(),
                'image_kurtosis': m_image_kurtosis[i].item(),
                'image_topk_ratio': m_image_topk_ratio[i].item(),
                'image_logits_var': m_image_logits_var[i].item(),
                'image_energy': m_image_energy[i].item(),
                'image_nn_intra_sim': m_nn_intra_sim_image[i].item(),
                'image_nn_inter_sim': m_nn_inter_sim_image[i].item(),
                'image_stability': m_image_stability_list[i],
                'image_switching_freq': m_image_switching_list[i],

                'feat_norm': m_feat_norm[i].item(),
                'feat_sparsity': m_feat_sparsity[i].item(),
            }
            self.log_data.append(log_entry)

    def after_run(self, alg):
        if not self.log_data:
            print("EnhancedAnalysisHook: No data to save.")
            return

        df = pd.DataFrame(self.log_data)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        save_path = os.path.join(self.model_dir, "metric_correlation_log.csv")
        df.to_csv(save_path, index=False)
        print(f"Enhanced metrics log saved to {save_path}")