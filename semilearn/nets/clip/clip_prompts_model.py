import torch
import torch.nn as nn
import json
from . import clip
from .pt_maple import CustomCLIP as MapleCLIP

class CLIPTopKPromptsWrapper(nn.Module):
    def __init__(self, num_classes, classnames, backbone='ViT-B/16',
                 prompts_path=None, k=2, alpha=0.5, **kwargs):
        super().__init__()

        assert prompts_path is not None, "Hybrid model requires prompts_path for the Top-K part."
        assert k > 0, "k must be a positive integer."
        assert 0.0 <= alpha <= 1.0, "alpha (weighting factor) must be between 0 and 1."

        self.k = k
        self.alpha = alpha
        self.num_classes = num_classes
        self.classnames = classnames

        maple_cfg = {
            'N_CTX': 2,
            'CTX_INIT': 'a photo of',
            'PROMPT_DEPTH': 9,
        }
        design_details = {"trainer": 'MaPLe', "vision_depth": maple_cfg['PROMPT_DEPTH'], "language_depth": maple_cfg['PROMPT_DEPTH'], "maple_length": maple_cfg['N_CTX']}
        clip_model_state_dict = clip.load(backbone, device='cpu')[0].state_dict()
        clip_model_for_maple = clip.build_model(clip_model_state_dict, design_details)
        self.maple_model = MapleCLIP(maple_cfg, classnames, clip_model_for_maple)

        self.base_clip_model, _ = clip.load(backbone, device='cpu', jit=False)
        self.text_features_per_class = self._load_and_encode_prompts(prompts_path)

        self.logit_scale = self.maple_model.logit_scale.exp
        self.num_features = self.base_clip_model.visual.output_dim

        for param in self.parameters():
            param.requires_grad = False
        print("Unfreezing parameters of MaPLe's prompt_learner for training...")
        for param in self.maple_model.prompt_learner.parameters():
            param.requires_grad = True

    def _load_and_encode_prompts(self, path):
        with open(path, 'r') as f:
            prompts_data = json.load(f)

        text_features_per_class = []
        with torch.no_grad():
            for classname in self.classnames:
                all_prompts = prompts_data[classname]['descriptive'] + prompts_data[classname]['distinctive']
                full_prompts = [f"a photo of {p}" for p in all_prompts]
                tokens = clip.tokenize(full_prompts).cpu()
                text_features = self.base_clip_model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features_per_class.append(text_features)
        return text_features_per_class

    def forward(self, x, only_feat=False, **kwargs):
        device = x.device
        self.maple_model.to(device)
        self.base_clip_model.to(device)

        text_features_maple, image_features = self.maple_model(x)

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        if only_feat:
            return image_features.to(torch.float32)

        text_features_maple_norm = text_features_maple / text_features_maple.norm(dim=-1, keepdim=True)
        logits_maple = self.logit_scale() * image_features_norm @ text_features_maple_norm.t()

        batch_size = image_features.shape[0]
        target_text_features = []
        for j in range(self.num_classes):
            class_prompts_embeds = self.text_features_per_class[j].to(device)
            sim = image_features_norm @ class_prompts_embeds.T
            _, topk_indices = torch.topk(sim, self.k, dim=1)
            topk_embeds = class_prompts_embeds[topk_indices]
            dynamic_class_feature = topk_embeds.mean(dim=1)
            target_text_features.append(dynamic_class_feature)

        target_text_features_tensor = torch.stack(target_text_features, dim=0).permute(1, 0, 2)
        logits_topk = self.logit_scale() * torch.einsum('bd,bnd->bn', image_features_norm, target_text_features_tensor)

        final_logits = self.alpha * logits_maple + (1.0 - self.alpha) * logits_topk

        return {'logits': final_logits, 'feat': image_features.to(torch.float32)}

    def group_matcher(self, *args, **kwargs):
        return {}

def clip_topk_prompts_vit_b_16(pretrained=True, **kwargs):
    assert 'classnames' in kwargs, "Hybrid model requires classnames."
    assert 'prompts_path' in kwargs, "Hybrid model requires prompts_path."
    kwargs.setdefault('backbone', 'ViT-B/16')
    return CLIPTopKPromptsWrapper(**kwargs)

def clip_topk_prompts_vit_b_32(pretrained=True, **kwargs):
    assert 'classnames' in kwargs, "Hybrid model requires classnames."
    assert 'prompts_path' in kwargs, "Hybrid model requires prompts_path."
    kwargs.setdefault('backbone', 'ViT-B/32')
    return CLIPTopKPromptsWrapper(**kwargs)