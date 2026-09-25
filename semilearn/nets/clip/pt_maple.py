
import torch
import torch.nn as nn
import copy
from . import clip
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .pt_encoders import TextEncoder, CustomImageEncoder

_tokenizer = _Tokenizer()

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, per_class_prompts=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        ctx_init = cfg['CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.ln_pre.weight.shape[0]

        self.compound_prompts_depth = cfg['PROMPT_DEPTH']

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

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype)) for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, vis_dim, dtype=dtype)
        self.compound_prompt_projections = nn.ModuleList([copy.deepcopy(single_layer) for _ in range(self.compound_prompts_depth - 1)])

        classnames = [name.replace("_", " ") for name in classnames]
        final_prompts = []
        prompt_prefix_for_construction = prompt_prefix + " "
        for name in classnames:
            if per_class_prompts and name in per_class_prompts:
                final_prompts.append(prompt_prefix_for_construction + per_class_prompts[name])
            else:
                final_prompts.append(prompt_prefix_for_construction + name + ".")

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in final_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx.float().unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix.float()
        suffix = self.token_suffix.float()
        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        visual_deep_prompts = []
        for i, layer in enumerate(self.compound_prompt_projections):
            layer.float()

            v_prompt = layer(self.compound_prompts_text[i].float())
            visual_deep_prompts.append(v_prompt)

        self.proj.float()
        shared_ctx = self.proj(self.ctx.float())

        return prompts, shared_ctx, self.compound_prompts_text, visual_deep_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, per_class_prompts=None):
        super().__init__()
        self.image_encoder = CustomImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model, per_class_prompts)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image):
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)

        if image is not None:
            image_features = self.image_encoder(image.float(), shared_ctx, deep_compound_prompts_vision)
        else:
            image_features = None

        return text_features, image_features