import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from collections import defaultdict

try:
    from semilearn.nets.clip import clip
    from semilearn.nets.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
    _tokenizer = _Tokenizer()
except ImportError:
    print("Error: Could not import semilearn.nets.clip. Make sure this script is run within the project structure.")
    exit()

CLIP_MODEL_NAME = 'ViT-B/32'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def load_clip_model(model_name, device):
    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model

@torch.no_grad()
def encode_texts(model, texts):

    tokens = clip.tokenize(texts).to(DEVICE)
    embeds = model.encode_text(tokens)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.float()

def find_confusable_classes(class_embeds, classnames, threshold=0.8):

    num_classes = len(classnames)

    sim_matrix = class_embeds @ class_embeds.T

    sim_matrix.fill_diagonal_(-1.0)

    confusable_map = defaultdict(list)
    total_confusions = 0

    for i in range(num_classes):
        conf_indices = (sim_matrix[i] > threshold).nonzero(as_tuple=True)[0]
        for idx in conf_indices:
            cid = idx.item()
            confusable_map[i].append(cid)
            total_confusions += 1

    return confusable_map

def select_best_attributes_adaptive(class_id, confusable_ids, class_embeds, attr_embeds_map, attr_texts_map, base_name, prefix="a photo of"):

    if class_id not in attr_embeds_map:
        return []

    attrs_embeds = attr_embeds_map[class_id]
    attrs_texts = attr_texts_map[class_id]

    valid_confusable_ids = [cid for cid in confusable_ids if cid < len(class_embeds)]

    if not valid_confusable_ids:
        return []

    conf_embeds = class_embeds[valid_confusable_ids]

    sims_to_others = attrs_embeds @ conf_embeds.T

    max_sim_to_any_confusable, _ = sims_to_others.max(dim=1)

    final_score = -max_sim_to_any_confusable

    sorted_indices = torch.argsort(final_score, descending=True)

    selected_attrs = []
    MAX_CONTENT_TOKENS = 74

    for idx in sorted_indices:
        attr_text = attrs_texts[idx]

        current_attrs_str = " ".join(selected_attrs + [attr_text])
        full_test_prompt = f"{prefix} {base_name} {current_attrs_str}"

        token_ids = _tokenizer.encode(full_test_prompt)

        if len(token_ids) <= MAX_CONTENT_TOKENS:
            selected_attrs.append(attr_text)
        else:
            break

    return selected_attrs


def generate_enhanced_prompts(attr_path, classnames, q=None):

    try:
        with open(attr_path, 'r') as f:
            attr_data_map = json.load(f)
    except FileNotFoundError:
        print(f"[TextEnhancer] ERROR: File not found {attr_path}")
        return None
    except json.JSONDecodeError:
        print(f"[TextEnhancer] ERROR: JSON Error.")
        return None

    model = load_clip_model(CLIP_MODEL_NAME, DEVICE)

    base_prompts = [f"a photo of {name.replace('_', ' ')}" for name in classnames]
    base_class_embeds = encode_texts(model, base_prompts)

    attr_embeds_map = {}
    attr_texts_map = {}

    for i, name in enumerate(classnames):
        simple_name = name.replace('_', ' ')
        if simple_name in attr_data_map:
            raw_texts = attr_data_map[simple_name]
            texts = [t for t in raw_texts if t and t.strip() != ""]

            if texts:
                attr_texts_map[i] = texts
                attr_embeds_map[i] = encode_texts(model, texts)

    confusable_map = find_confusable_classes(base_class_embeds, classnames)

    per_class_prompts = {}

    for i, name in enumerate(classnames):
        simple_name = name.replace('_', ' ')

        if i in confusable_map:
            conf_ids = confusable_map[i]

            best_attrs = select_best_attributes_adaptive(
                class_id=i,
                confusable_ids=conf_ids,
                class_embeds=base_class_embeds,
                attr_embeds_map=attr_embeds_map,
                attr_texts_map=attr_texts_map,
                base_name=simple_name
            )

            if best_attrs:
                final_prompt_text = simple_name + " " + " ".join(best_attrs)
                per_class_prompts[name] = final_prompt_text
            else:
                per_class_prompts[name] = simple_name
        else:
            per_class_prompts[name] = simple_name


    del model
    torch.cuda.empty_cache()

    return per_class_prompts