

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import clip

class CLIPVisionWrapper(nn.Module):
    def __init__(self, backbone, num_classes):
        super(CLIPVisionWrapper, self).__init__()
        self.backbone = backbone
        self.num_features = backbone.output_dim

        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.fc(x)

        feat = self.backbone(x)

        if only_feat:
            return feat.to(torch.float32)

        logits = self.fc(feat.to(torch.float32))

        return {'logits': logits, 'feat': feat.to(torch.float32)}

    def group_matcher(self, coarse=False, prefix=''):
        if hasattr(self.backbone, 'transformer'):
            matcher = {
                'stem': r'^{}backbone.conv1|^{}backbone.class_embedding|^{}backbone.positional_embedding|^{}backbone.ln_pre'.format(prefix, prefix, prefix, prefix),
                'blocks': r'^{}backbone.transformer.resblocks.(\d+)'.format(prefix),
            }

        else:
            matcher = {
                'stem': r'^{}backbone.conv1|^{}backbone.bn1|^{}backbone.conv2|^{}backbone.bn2|^{}backbone.conv3|^{}backbone.bn3'.format(prefix, prefix, prefix, prefix, prefix, prefix),
                'blocks': [
                    (r'^{}backbone.layer(\d+)'.format(prefix), None),
                    (r'^{}backbone.attnpool'.format(prefix), (len(self.backbone.layer4) + 1,)),
                ]
            }
        return matcher

    def no_weight_decay(self):
        nwd_params = set()
        if hasattr(self.backbone, 'positional_embedding'):
            nwd_params.add('backbone.positional_embedding')
        if hasattr(self.backbone, 'class_embedding'):
            nwd_params.add('backbone.class_embedding')

        for n, p in self.named_parameters():
             if n in nwd_params or 'bias' in n or 'ln_' in n or 'bn' in n:
                nwd_params.add(n)
        return nwd_params


def create_clip_model(arch='ViT-B/16', pretrained=True, num_classes=10, **kwargs):
    if not pretrained:
        raise NotImplementedError("Creating a CLIP model without pretrained weights is not supported.")

    model, _ = clip.load(arch, device='cpu', jit=False)
    vision_backbone = model.visual

    return CLIPVisionWrapper(backbone=vision_backbone, num_classes=num_classes)

def clip_vit_b_32(pretrained=True, **kwargs):
    return create_clip_model(arch='ViT-B/32', pretrained=pretrained, **kwargs)

def clip_vit_b_16(pretrained=True, **kwargs):
    return create_clip_model(arch='ViT-B/16', pretrained=pretrained, **kwargs)

def clip_vit_l_14(pretrained=True, **kwargs):
    return create_clip_model(arch='ViT-L/14', pretrained=pretrained, **kwargs)

def clip_vit_l_14_336px(pretrained=True, **kwargs):
    return create_clip_model(arch='ViT-L/14@336px', pretrained=pretrained, **kwargs)

def clip_rn50(num_classes, classnames, prompt_template=None, **kwargs):
    return create_clip_model(arch='RN50', pretrained=pretrained, **kwargs)

def clip_rn101(pretrained=True, **kwargs):
    return create_clip_model(arch='RN101', pretrained=pretrained, **kwargs)

def clip_rn50x4(pretrained=True, **kwargs):
    return create_clip_model(arch='RN50x4', pretrained=pretrained, **kwargs)

def clip_rn50x16(pretrained=True, **kwargs):
    return create_clip_model(arch='RN50x16', pretrained=pretrained, **kwargs)

def clip_rn50x64(pretrained=True, **kwargs):
    return create_clip_model(arch='RN50x64', pretrained=pretrained, **kwargs)

from .pt_maple import CustomCLIP as MapleCLIP

class Adapter(nn.Module):
    def __init__(self, in_dim, bottleneck_ratio=0.25, residual_weight=0.1):
        super().__init__()
        hidden_dim = int(in_dim * bottleneck_ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim)
        )
        self.residual_weight = residual_weight

        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        return x + self.residual_weight * self.fc(x)

from .pt_maple import CustomCLIP as MapleCLIP

class CLIPMapleWrapper(nn.Module):
    def __init__(self, num_classes, classnames, backbone='ViT-B/16',
                 prompt_template: str = None, per_class_prompts: dict = None,
                 **kwargs):
        super().__init__()

        ctx_init = prompt_template if prompt_template else 'a photo of a'
        print(f"[CLIPMapleWrapper] Initializing MaPLe with template: '{ctx_init}'")

        maple_cfg = {'N_CTX': 2, 'CTX_INIT': ctx_init, 'PROMPT_DEPTH': 9}
        design_details = {"trainer": 'MaPLe', "vision_depth": 9, "language_depth": 9, "maple_length": 2}

        clip_model_obj, _ = clip.load(backbone, device='cpu')
        clip_model_state_dict = clip_model_obj.state_dict()
        maple_ready_clip_model = clip.build_model(clip_model_state_dict, design_details)

        self.inner_model = MapleCLIP(maple_cfg, classnames, maple_ready_clip_model, per_class_prompts)
        self.num_features = maple_ready_clip_model.text_projection.shape[1]
        self.logit_scale = self.inner_model.logit_scale

        for param in self.parameters():
            param.requires_grad = False

        print("[CLIPMapleWrapper] Unfreezing Prompt Learner...")
        for param in self.inner_model.prompt_learner.parameters():
            param.requires_grad = True
    def forward(self, x, **kwargs):

        text_features, image_features = self.inner_model(x)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return {
            'logits_text': logits,
            'feat': image_features,
            'text_feat': text_features
        }

    def get_learned_text_features(self):
        text_features, _ = self.inner_model(None)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def get_text_features_training(self):
        return self.get_learned_text_features()

def clip_maple_vit_b_16(**kwargs):
    kwargs['backbone'] = 'ViT-B/16'
    return CLIPMapleWrapper(**kwargs)

def clip_maple_vit_b_32(**kwargs):
    kwargs['backbone'] = 'ViT-B/32'
    return CLIPMapleWrapper(**kwargs)