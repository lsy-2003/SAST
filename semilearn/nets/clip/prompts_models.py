
import logging
import torch
from torch import nn

log = logging.getLogger(__name__)

class Adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dtype=torch.float32):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim).to(dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim).to(dtype),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CrossAdapter(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, device="cpu", dtype=torch.float32, residue_weight=0.01, residual=True):
        super(CrossAdapter, self).__init__()
        self.residue_weight = residue_weight
        self.residual = residual
        self.visual_adapter = Adapter(in_dim, hidden_dim, dtype).to(device)
        self.textual_adapter = Adapter(in_dim, hidden_dim, dtype).to(device)

    def forward(self, text_out=None, visual_out=None):
        if text_out is not None:
            if self.residual:
                text_out = text_out + 0.01 * self.textual_adapter(text_out)
            else:
                text_out = self.textual_adapter(text_out)
        if visual_out is not None:
            if self.residual:
                visual_out = visual_out + 0.01 * self.visual_adapter(visual_out)
            else:
                visual_out = self.visual_adapter(visual_out)
        return text_out, visual_out


class DebiasModule(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, device="cpu", dtype=torch.float32, cfg=None):
        super(DebiasModule, self).__init__()
        self.cfg = cfg
        self.cross_adapter = CrossAdapter(in_dim, hidden_dim, device, dtype)
        self.pseudo_adapter = Adapter(in_dim, hidden_dim, dtype).to(device)

    def forward(self, text_out=None, visual_out=None, inference=False):
        if inference:
            return text_out, visual_out, None

        t, v = self.cross_adapter(text_out, visual_out)

        v_pseudo = None
        if visual_out is not None:
            v_pseudo = visual_out + 0.01 * self.pseudo_adapter(visual_out)

        return t, v, v_pseudo


class TextPrefixModel(nn.Module):
    def __init__(
        self, initial_prefix, text_encoder, classes, temperature=0.07, device="cpu"
    ):
        """Define the model for textual prompt tuning.

        :param initial_prefix: initializes tensor of floats
        :param text_encoder: text encoder to use
        :param classes: list of classes' names
        :param temperature: fix parameter, same as clip
        :param device: device in use
        """

        super(TextPrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix
        self.classes = classes

        self.prefix = nn.Parameter(initial_prefix)
        self.text_encoder = text_encoder

    def forward(self, classes):

        out = self.text_encoder(self.prefix, classes)


        return out


class ImagePrefixModel(nn.Module):
    def __init__(
        self,
        initial_prefix,
        image_encoder,
        temperature=0.07,
        device="cpu",
    ):
        super(ImagePrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix


        self.prefix = nn.Parameter(initial_prefix)
        self.image_encoder = image_encoder

    def forward(self, x):


        out = self.image_encoder(x, self.prefix)


        return out


class UPTModel(nn.Module):
    def __init__(
        self,
        coop_embeddings,
        vpt_embeddings,
        vpt_embeddings_deep,
        image_encoder,
        text_encoder,
        classes,
        dim_transformer,
        temperature=0.07,
        device="cpu",
        dtype=torch.float32,
    ):
        super(UPTModel, self).__init__()
        self.device = device
        self.classes = classes
        self.temperature = temperature
        self.dtype = dtype


        self.coop_embeddings = nn.Parameter(coop_embeddings)
        self.vpt_embeddings = nn.Parameter(vpt_embeddings)

        self.coop_length = self.coop_embeddings.size()[1]
        self.coop_dim = self.coop_embeddings.size()[2]

        self.vpt_length = self.vpt_embeddings.size()[1]
        self.vpt_dim = self.vpt_embeddings.size()[2]

        if vpt_embeddings_deep is not None:
            self.vpt_embeddings_deep = nn.Parameter(vpt_embeddings_deep)
        else:
            self.vpt_embeddings_deep = None

        self.proj_coop_pre = nn.Linear(
            self.coop_dim,
            dim_transformer,
            dtype=self.dtype).to(self.device)
        self.proj_coop_post = nn.Linear(
            dim_transformer,
            self.coop_dim,
            dtype=self.dtype).to(self.device)
        self.proj_vpt_pre = nn.Linear(
            self.vpt_dim,
            dim_transformer,
            dtype=self.dtype).to(self.device)
        self.proj_vpt_post = nn.Linear(
            dim_transformer,
            self.vpt_dim,
            dtype=self.dtype).to(self.device)


        from . import model as clip_model_module
        self.transformer = clip_model_module.Transformer(
            width=dim_transformer,
            layers=1,
            heads=1).to(self.device)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, x, classes):

        coop_embeddings = self.coop_embeddings
        coop_embds = self.proj_coop_pre(coop_embeddings).to(self.device)


        vpt_embds = self.proj_vpt_pre(self.vpt_embeddings).to(self.device)

        prompt_seq = torch.cat((coop_embds, vpt_embds), dim=0).to(torch.float32)


        output_seq = self.transformer(prompt_seq).to(torch.float16)


        coop_embs = self.proj_coop_post(output_seq[:len(self.coop_embeddings)].to(self.dtype)).reshape(-1, self.coop_length, self.coop_dim)
        vpt_embs = self.proj_vpt_post(output_seq[len(self.coop_embeddings):].to(self.dtype)).reshape(-1, self.vpt_length, self.vpt_dim)

        text_out = self.text_encoder(coop_embs, classes)
        visual_out = self.image_encoder(x, vpt_embs)

        return text_out, visual_out

class UPT_Adapter(nn.Module):
    def __init__(
        self,
        coop_embeddings,
        vpt_embeddings,
        vpt_embeddings_deep,
        image_encoder,
        text_encoder,
        classes,
        dim_transformer,
        temperature=0.07,
        reduction=4,
        device="cpu",
        dtype=torch.float32,
    ):
        super(UPT_Adapter, self).__init__()
        self.upt_model = UPTModel(
            coop_embeddings,
            vpt_embeddings,
            vpt_embeddings_deep,
            image_encoder,
            text_encoder,
            classes,
            dim_transformer,
            temperature,
            device,
            dtype
        )
        self.textual_adapter = Adapter(512, reduction, dtype=dtype)
        self.visual_adapter = Adapter(512, reduction, dtype=dtype)

    def forward(self, x, classes, unlabeled=False):
        text_out, visual_out = self.upt_model(x, classes)
        return text_out, visual_out