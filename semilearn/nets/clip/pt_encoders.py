
import torch
import torch.nn as nn
from . import clip

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        self.transformer.float()
        self.ln_final.float()

        x = prompts.float() + self.positional_embedding.float()
        x = x.permute(1, 0, 2)

        if compound_prompts_deeper_text is not None:
            if isinstance(compound_prompts_deeper_text, list):
                compound_prompts_deeper_text = [p.float() for p in compound_prompts_deeper_text]
            elif isinstance(compound_prompts_deeper_text, torch.Tensor):
                compound_prompts_deeper_text = compound_prompts_deeper_text.float()

        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)

        x = outputs[0]
        x = x.permute(1, 0, 2)

        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.float()
        return x

class CustomImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, image, shared_ctx, deep_embs):
        self.image_encoder.float()

        image = image.float()

        if shared_ctx is not None:
            shared_ctx = shared_ctx.float()

        if deep_embs is not None:
            if isinstance(deep_embs, list):
                deep_embs = [p.float() for p in deep_embs]
            elif isinstance(deep_embs, torch.Tensor):
                deep_embs = deep_embs.float()

        return self.image_encoder(image, shared_ctx, deep_embs)