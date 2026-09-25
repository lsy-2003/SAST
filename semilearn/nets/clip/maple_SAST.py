

import torch
import torch.nn as nn
from collections import OrderedDict
import copy

from . import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        combined = [x, compound_prompts_deeper_text, 0]
        if isinstance(self.transformer.resblocks, nn.Sequential) and len(self.transformer.resblocks) > 0 and 'MaPLe' in str(type(self.transformer.resblocks[0])):
             outputs = self.transformer(combined)
             x = outputs[0]
        else:

            x = self.transformer(x)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.get('maple_n_ctx', 16)
        ctx_init = cfg.get('maple_ctx_init', None)
        prompt_depth = cfg.get('maple_prompt_depth', 8)

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.proj.shape[1] if hasattr(clip_model.visual, 'proj') else 1024

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.proj = nn.Linear(ctx_dim, vis_dim, dtype=dtype)
        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype)) for _ in range(prompt_depth - 1)]
        )
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer_proj = nn.Linear(ctx_dim, vis_dim, dtype=dtype)
        self.compound_prompt_projections = _get_clones(single_layer_proj, prompt_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        visual_deep_prompts = [proj(p) for p, proj in zip(self.compound_prompts_text, self.compound_prompt_projections)]

        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts


class MaPLeSAST(nn.Module):
    def __init__(self, cfg, num_classes, classnames):
        super().__init__()
        self.cfg = cfg

        self.clip_model = self.load_clip(cfg)

        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)

        self.logit_scale = self.clip_model.logit_scale
        self.text_features = None

        self.freeze_parameters()

        self.freeze_parameters()

    def load_clip(self, cfg):
        backbone_name = cfg.net.replace('maple_SAST_', '').replace('_', '/')

        design_details = {
            "trainer": 'MaPLe',
            "vision_depth": cfg.get('maple_prompt_depth', 8),
            "language_depth": cfg.get('maple_prompt_depth', 8),
            "vision_ctx": cfg.get('maple_n_ctx', 16),
            "language_ctx": cfg.get('maple_n_ctx', 16),
            "maple_length": cfg.get('maple_n_ctx', 16)
        }
        model, _ = clip.load(backbone_name, device='cpu', jit=False, design_details=design_details)
        return model

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.prompt_learner.parameters():
            param.requires_grad = True


    def forward(self, x, only_feat=False, **kwargs):
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))
        prompts, shared_ctx, deep_prompts_text, deep_prompts_vision = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_prompts_text)

        image_features = self.image_encoder(x.type(self.clip_model.dtype), shared_ctx, deep_prompts_vision)

        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if only_feat:
            return image_features.to(torch.float32)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return {'logits': logits, 'feat': image_features.to(torch.float32)}

    def group_matcher(self, coarse=False, prefix=''):
        return {'all': r'.*'}

    def no_weight_decay(self):
        return {name for name, W in self.named_parameters()}

    @torch.no_grad()
    def get_learned_text_features(self):
        self.logit_scale.clamp_(0, 4.6052)
        prompts, shared_ctx, deep_prompts_text, deep_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


def maple_SAST_vit_b_32(pretrained=True, num_classes=200, **kwargs):
    cfg = kwargs.get('cfg')
    classnames = kwargs.get('classnames')
    if classnames is None:
        classnames = [f'class {i}' for i in range(num_classes)]
    return MaPLeSAST(cfg, num_classes=num_classes, classnames=classnames)