

import os
import json
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data
from torchvision.transforms.functional import InterpolationMode
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_resisc(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    dataset_root = os.path.join(data_dir, 'RESISC45')

    crop_size = args.img_size

    interpolation = InterpolationMode.BICUBIC

    transform_weak = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])


    transform_val = transforms.Compose([
        transforms.Resize(crop_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    def read_json_split(json_path, img_root):
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

        category_id_to_label = {cat['id']: i for i, cat in enumerate(data['categories'])}

        paths = []
        targets = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']

            filename = image_id_to_filename[image_id]
            if '@' in filename:
                filename = filename.split('@')[1]

            full_path = os.path.join(img_root, filename)
            label = category_id_to_label[category_id]

            paths.append(full_path)
            targets.append(label)

        return paths, targets

    train_paths, train_targets = read_json_split(os.path.join(dataset_root, 'train.json'), dataset_root)
    val_paths, val_targets = read_json_split(os.path.join(dataset_root, 'val.json'), dataset_root)
    test_paths, test_targets = read_json_split(os.path.join(dataset_root, 'test.json'), dataset_root)

    train_paths += val_paths
    train_targets += val_targets
    val_paths = test_paths
    val_targets = test_targets
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args, train_paths, train_targets, num_classes,
        lb_num_labels=num_labels,
        ulb_num_labels=args.ulb_num_labels,
        lb_imbalance_ratio=args.lb_imb_ratio,
        ulb_imbalance_ratio=args.ulb_imb_ratio,
        include_lb_to_ulb=include_lb_to_ulb
    )

    lb_dset = BasicDataset(alg=alg, data=lb_data, targets=lb_targets, num_classes=num_classes,
                           transform=transform_weak, is_ulb=False, strong_transform=transform_strong)

    ulb_dset = BasicDataset(alg=alg, data=ulb_data, targets=ulb_targets, num_classes=num_classes,
                            transform=transform_weak, is_ulb=True, strong_transform=transform_strong)

    eval_dset = BasicDataset(alg=alg, data=val_paths, targets=val_targets, num_classes=num_classes,
                             transform=transform_val, is_ulb=False)

    test_dset = BasicDataset(alg=alg, data=test_paths, targets=test_targets, num_classes=num_classes,
                             transform=transform_val, is_ulb=False)

    with open(os.path.join(dataset_root, 'train.json'), 'r') as f:
        data = json.load(f)
    classnames = [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]

    print(f"RESISC45: #Labeled: {len(lb_dset)} #Unlabeled: {len(ulb_dset)} #Val: {len(eval_dset)} #Test: {len(test_dset)}")

    return lb_dset, ulb_dset, eval_dset, test_dset, classnames

def get_resisc_tl(args, alg, name, num_classes, data_dir='./data'):
    dataset_root = os.path.join(data_dir, 'RESISC45')

    crop_size = args.img_size

    interpolation = InterpolationMode.BICUBIC

    transform_weak = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])


    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])


    transform_val = transforms.Compose([
        transforms.Resize(crop_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    with open(os.path.join(dataset_root, 'train.json'), 'r') as f:
        data = json.load(f)
    classnames = [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]
    label_to_idx = {c: i for i, c in enumerate(classnames)}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    split_path = os.path.join(current_dir, 'split.json')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split.json not found at {split_path}")

    with open(split_path, 'r') as f:
        split_config = json.load(f)

    if 'resisc' not in split_config:
        raise ValueError(f"Dataset key 'resisc' not found in split.json.")

    seen_classes = set(split_config['resisc']['seen'])
    unseen_classes = set(split_config['resisc']['unseen'])

    seen_indices = set(label_to_idx[c] for c in seen_classes if c in label_to_idx)
    unseen_indices = set(label_to_idx[c] for c in unseen_classes if c in label_to_idx)

    def read_json_split(json_path, img_root):
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        category_id_to_label = {cat['id']: i for i, cat in enumerate(data['categories'])}

        paths = []
        targets = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            filename = image_id_to_filename[image_id]
            if '@' in filename: filename = filename.split('@')[1]
            full_path = os.path.join(img_root, filename)
            label = category_id_to_label[category_id]
            paths.append(full_path)
            targets.append(label)
        return paths, targets

    train_paths, train_targets = read_json_split(os.path.join(dataset_root, 'train.json'), dataset_root)
    val_paths, val_targets = read_json_split(os.path.join(dataset_root, 'val.json'), dataset_root)
    all_train_paths = train_paths + val_paths
    all_train_targets = train_targets + val_targets

    test_paths, test_targets = read_json_split(os.path.join(dataset_root, 'test.json'), dataset_root)

    lb_data, lb_targets = [], []
    ulb_data, ulb_targets = [], []

    for path, target in zip(all_train_paths, all_train_targets):
        if target in seen_indices:
            lb_data.append(path)
            lb_targets.append(target)
        elif target in unseen_indices:
            ulb_data.append(path)
            ulb_targets.append(target)

    lb_dset = BasicDataset(alg=alg, data=lb_data, targets=lb_targets, num_classes=num_classes,
                           transform=transform_weak, is_ulb=False, strong_transform=transform_strong)

    ulb_dset = BasicDataset(alg=alg, data=ulb_data, targets=ulb_targets, num_classes=num_classes,
                            transform=transform_weak, is_ulb=True, strong_transform=transform_strong)

    test_dset = BasicDataset(alg=alg, data=test_paths, targets=test_targets, num_classes=num_classes,
                             transform=transform_val, is_ulb=False)

    print(f"[RESISC45 TL] Seen: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    print(f"[RESISC45 TL] #Labeled (Seen): {len(lb_dset)}, #Unlabeled (Unseen): {len(ulb_dset)}, #Test: {len(test_dset)}")

    return lb_dset, ulb_dset, test_dset, classnames, list(seen_classes), list(unseen_classes)