

import torch
import torch.nn.functional as F
import pandas as pd
import os
from semilearn.core.hooks import Hook

class SASTThresholdingHook(Hook):

    def __init__(self, num_classes, base_thresh=0.95, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.per_class = per_class
        self.momentum = momentum
        self.base_thresh = base_thresh


        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(self.base_thresh)
        else:
            self.prob_max_mu_t = torch.ones(self.num_classes) * self.base_thresh


    @torch.no_grad()
    def update(self, probs_x_ulb):


        self.prob_max_mu_t = self.prob_max_mu_t.to(probs_x_ulb.device)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)

        if not self.per_class:
            prob_max_mu_batch = torch.mean(max_probs)
            self.prob_max_mu_t = self.momentum * self.prob_max_mu_t + (1 - self.momentum) * prob_max_mu_batch
        else:

            pass



    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, sigma, softmax_x_ulb=True, *args, **kwargs):

        if softmax_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()


        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)



        mu_p = torch.max(self.prob_max_mu_t, torch.tensor(self.base_thresh, device=self.prob_max_mu_t.device))
        high_conf_mask = max_probs.ge(mu_p)

        return high_conf_mask


class PrototypeAnalysisHook(Hook):
    def __init__(self, model_dir, num_classes, *args, **kwargs):
        super().__init__()
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.stats_log_data = []
        self.selected_prototypes = []
        self.classifier_log_data = []
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    @torch.no_grad()
    def log_selection_stats(self, algorithm, iteration, class_id, all_selected_feats_cpu, text_proto_c):
        if not all_selected_feats_cpu:
            return


        all_feats = torch.stack(all_selected_feats_cpu).cpu().float()
        text_proto_c = text_proto_c.cpu().float()

        all_feats_norm = F.normalize(all_feats, dim=-1)
        text_proto_norm = F.normalize(text_proto_c, dim=-1)

        consistencies = all_feats_norm @ text_proto_norm
        avg_consistency = consistencies.mean().item()

        avg_intra_similarity = 0.0
        if len(all_feats) > 1:
            sim_matrix = all_feats_norm @ all_feats_norm.T

            upper_triangle_indices = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            if upper_triangle_indices.any():
                avg_intra_similarity = sim_matrix[upper_triangle_indices].mean().item()

        self.stats_log_data.append({
            'iteration': iteration,
            'class_id': class_id,
            'num_samples': len(all_feats),
            'avg_consistency': avg_consistency,
            'avg_intra_similarity': avg_intra_similarity,
        })

    def log_selected_prototypes(self, algorithm, iteration, class_id, selected_candidates):
        for cand in selected_candidates:

            label_item = cand['label'].item() if isinstance(cand['label'], torch.Tensor) else cand['label']
            self.selected_prototypes.append({
                'iteration': iteration,
                'class_id': class_id,
                'idx_ulb': cand['idx'],
                'pseudo_label': label_item,
                'hybrid_score': cand['hybrid_score']
            })

    @torch.no_grad()
    def log_classifier_similarity(self, algorithm, iteration, image_prototypes, text_prototypes):
        if image_prototypes is None or text_prototypes is None:
            return

        img_p = F.normalize(image_prototypes.cpu().float(), dim=-1)
        txt_p = F.normalize(text_prototypes.cpu().float(), dim=-1)


        classifier_sim_matrix = img_p @ txt_p.T

        avg_diag_sim = torch.diag(classifier_sim_matrix).mean().item()

        mask_off_diag = ~torch.eye(self.num_classes, dtype=torch.bool)
        avg_off_diag_sim = classifier_sim_matrix[mask_off_diag].mean().item()

        self.classifier_log_data.append({
            'iteration': iteration,
            'avg_diag_sim': avg_diag_sim,
            'avg_off_diag_sim': avg_off_diag_sim,
        })

    def save_log(self, algorithm):
        if self.stats_log_data:
            log_df = pd.DataFrame(self.stats_log_data)
            save_path = os.path.join(self.model_dir, "prototype_stats_log.csv")
            log_df.to_csv(save_path, index=False)
            algorithm.print_fn(f"prototype log saved to: {save_path}")

        if self.classifier_log_data:
            log_df = pd.DataFrame(self.classifier_log_data)
            save_path = os.path.join(self.model_dir, "classifier_similarity_log.csv")
            log_df.to_csv(save_path, index=False)
            algorithm.print_fn(f"classifier sim log saved to: {save_path}")

        if self.selected_prototypes:
            df_selected = pd.DataFrame(self.selected_prototypes)
            log_path = os.path.join(self.model_dir, "selected_prototypes_log.csv")
            df_selected.to_csv(log_path, index=False)
            algorithm.print_fn(f"Selected prototypes log saved to {log_path}")