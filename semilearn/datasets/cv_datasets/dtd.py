

import os, json
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_dtd(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    """
    Standard DTD dataset for SSL/UL.
    Aligning transforms with CAP (CoOp/CLIP standard).
    """
    dataset_root = os.path.join(data_dir, 'DTD')
    crop_size = args.img_size

    interpolation = InterpolationMode.BICUBIC

    transform_weak = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=interpolation),
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

    def read_split_file(split_path, img_root_subdir):
        paths = []
        targets = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                path_part = parts[0]
                label_part = parts[1]
                if '@' in path_part: path_part = path_part.split('@')[1]
                full_path = os.path.join(img_root_subdir, path_part)
                paths.append(full_path)
                targets.append(int(label_part))
        return paths, targets

    train_paths, train_targets = read_split_file(os.path.join(dataset_root, 'train.txt'), os.path.join(dataset_root, 'train'))
    val_paths, val_targets = read_split_file(os.path.join(dataset_root, 'val.txt'), os.path.join(dataset_root, 'val'))
    test_paths, test_targets = read_split_file(os.path.join(dataset_root, 'test.txt'), os.path.join(dataset_root, 'test'))

    train_paths += val_paths
    train_targets += val_targets
    val_paths = test_paths
    val_targets = test_targets

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args, train_paths, train_targets, num_classes,
        lb_num_labels=num_labels, ulb_num_labels=args.ulb_num_labels,
        lb_imbalance_ratio=args.lb_imb_ratio, ulb_imbalance_ratio=args.ulb_imb_ratio,
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

    class_names_path = os.path.join(dataset_root, 'class_names.txt')
    with open(class_names_path) as f:
        classnames = [line.strip() for line in f.readlines()]

    print(f"DTD: #Labeled: {len(lb_dset)} #Unlabeled: {len(ulb_dset)} #Val: {len(eval_dset)} #Test: {len(test_dset)}")
    return lb_dset, ulb_dset, eval_dset, test_dset, classnames

def get_dtd_tl(args, alg, name, num_classes, data_dir='./data'):

    dataset_root = os.path.join(data_dir, 'DTD')

    interpolation = InterpolationMode.BICUBIC
    crop_size = args.img_size

    transform_weak = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=interpolation),
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

    class_names_path = os.path.join(dataset_root, 'class_names.txt')
    with open(class_names_path) as f:
        classnames = [line.strip() for line in f.readlines()]
    label_to_idx = {c: i for i, c in enumerate(classnames)}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    split_path = os.path.join(current_dir, 'split.json')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split.json not found at {split_path}")

    with open(split_path, 'r') as f:
        split_config = json.load(f)

    dataset_key = 'dtd'
    seen_classes = set(split_config[dataset_key]['seen'])
    unseen_classes = set(split_config[dataset_key]['unseen'])

    seen_indices = set(label_to_idx[c] for c in seen_classes if c in label_to_idx)
    unseen_indices = set(label_to_idx[c] for c in unseen_classes if c in label_to_idx)

    def read_split_file(split_path, img_root_subdir):
        paths = []
        targets = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                path_part = parts[0]
                label_part = int(parts[1])
                if '@' in path_part: path_part = path_part.split('@')[1]
                full_path = os.path.join(img_root_subdir, path_part)
                paths.append(full_path)
                targets.append(label_part)
        return paths, targets

    train_paths, train_targets = read_split_file(
        os.path.join(dataset_root, 'train.txt'), os.path.join(dataset_root, 'train')
    )
    val_paths, val_targets = read_split_file(
        os.path.join(dataset_root, 'val.txt'), os.path.join(dataset_root, 'val')
    )
    all_train_paths = train_paths + val_paths
    all_train_targets = train_targets + val_targets

    test_paths, test_targets = read_split_file(
        os.path.join(dataset_root, 'test.txt'), os.path.join(dataset_root, 'test')
    )

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

    print(f"[DTD TL] Seen Classes: {len(seen_classes)}, Unseen Classes: {len(unseen_classes)}")
    print(f"[DTD TL] #Labeled (Seen): {len(lb_dset)}, #Unlabeled (Unseen): {len(ulb_dset)}, #Test: {len(test_dset)}")

    return lb_dset, ulb_dset, test_dset, classnames, list(seen_classes), list(unseen_classes)