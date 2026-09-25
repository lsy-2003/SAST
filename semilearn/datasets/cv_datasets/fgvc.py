import os, json
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data
from torchvision.transforms.functional import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_fgvc(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    dataset_root = os.path.join(data_dir, 'FGVCAircraft')
    interpolation = InterpolationMode.LANCZOS
    crop_size = args.img_size
    transform_weak = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation, antialias=True),
        transforms.RandomCrop(crop_size, padding=int(crop_size * 0.125), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])


    transform_strong = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation, antialias=True),
        transforms.RandomCrop(crop_size, padding=int(crop_size * 0.125), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    STORAGE_TO_SEMANTIC_MAP = {


    }

    def read_split_file(split_path, img_root_subdir, classnames):
        paths = []
        targets = []

        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                path_part, remainder = line.split('.jpg', 1)
                path_part += '.jpg'

                original_label_idx = int(remainder.strip().split()[-1])

                if '@' in path_part:
                    path_part = path_part.split('@')[1]

                full_path = os.path.join(img_root_subdir, path_part)

                dir_name = os.path.basename(os.path.dirname(full_path))

                if dir_name in STORAGE_TO_SEMANTIC_MAP:
                    semantic_label = STORAGE_TO_SEMANTIC_MAP[dir_name]
                    new_label_idx = classnames.index(semantic_label)
                    targets.append(new_label_idx)
                    print(f"映射: {dir_name} -> {semantic_label} (索引: {original_label_idx} -> {new_label_idx})")
                else:
                    targets.append(original_label_idx)

                paths.append(full_path)

        return paths, targets

    class_names_path = os.path.join(dataset_root, 'labels.txt')
    with open(class_names_path) as f:
        classnames = [line.strip() for line in f.readlines()]

    train_paths, train_targets = read_split_file(
        os.path.join(dataset_root, 'train.txt'),
        os.path.join(dataset_root, 'train'),
        classnames
    )
    val_paths, val_targets = read_split_file(
        os.path.join(dataset_root, 'val.txt'),
        os.path.join(dataset_root, 'val'),
        classnames
    )
    test_paths, test_targets = read_split_file(
        os.path.join(dataset_root, 'test.txt'),
        os.path.join(dataset_root, 'test'),
        classnames
    )

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

    print(f"FGVCAircraft: #Labeled: {len(lb_dset)} #Unlabeled: {len(ulb_dset)} #Val: {len(eval_dset)} #Test: {len(test_dset)}")

    return lb_dset, ulb_dset, eval_dset, test_dset, classnames

def get_fgvc_tl(args, alg, name, num_classes, data_dir='./data'):

    dataset_root = os.path.join(data_dir, 'FGVCAircraft')
    interpolation = InterpolationMode.LANCZOS
    crop_size = args.img_size
    transform_weak = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation, antialias=True),
        transforms.RandomCrop(crop_size, padding=int(crop_size * 0.125), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])


    transform_strong = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation, antialias=True),
        transforms.RandomCrop(crop_size, padding=int(crop_size * 0.125), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    class_names_path = os.path.join(dataset_root, 'labels.txt')
    with open(class_names_path) as f:
        classnames = [line.strip() for line in f.readlines()]
    label_to_idx = {c: i for i, c in enumerate(classnames)}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    split_path = os.path.join(current_dir, 'split.json')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split.json not found at {split_path}")

    with open(split_path, 'r') as f:
        split_config = json.load(f)

    if 'fgvcaircraft' not in split_config:
        raise ValueError(f"Dataset key 'fgvcaircraft' not found in split.json.")

    seen_classes = set(split_config['fgvcaircraft']['seen'])
    unseen_classes = set(split_config['fgvcaircraft']['unseen'])

    seen_indices = set(label_to_idx[c] for c in seen_classes if c in label_to_idx)
    unseen_indices = set(label_to_idx[c] for c in unseen_classes if c in label_to_idx)

    STORAGE_TO_SEMANTIC_MAP = {}

    def read_split_file(split_path, img_root_subdir, classnames):
        paths = []
        targets = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                path_part, remainder = line.split('.jpg', 1)
                path_part += '.jpg'
                original_label_idx = int(remainder.strip().split()[-1])

                if '@' in path_part: path_part = path_part.split('@')[1]
                full_path = os.path.join(img_root_subdir, path_part)

                dir_name = os.path.basename(os.path.dirname(full_path))
                if dir_name in STORAGE_TO_SEMANTIC_MAP:
                    semantic_label = STORAGE_TO_SEMANTIC_MAP[dir_name]
                    new_label_idx = classnames.index(semantic_label)
                    targets.append(new_label_idx)
                else:
                    targets.append(original_label_idx)
                paths.append(full_path)
        return paths, targets

    train_paths, train_targets = read_split_file(
        os.path.join(dataset_root, 'train.txt'), os.path.join(dataset_root, 'train'), classnames
    )
    val_paths, val_targets = read_split_file(
        os.path.join(dataset_root, 'val.txt'), os.path.join(dataset_root, 'val'), classnames
    )
    all_train_paths = train_paths + val_paths
    all_train_targets = train_targets + val_targets

    test_paths, test_targets = read_split_file(
        os.path.join(dataset_root, 'test.txt'), os.path.join(dataset_root, 'test'), classnames
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

    print(f"[FGVCAircraft TL] Seen: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    print(f"[FGVCAircraft TL] #Labeled (Seen): {len(lb_dset)}, #Unlabeled (Unseen): {len(ulb_dset)}, #Test: {len(test_dset)}")

    return lb_dset, ulb_dset, test_dset, classnames, list(seen_classes), list(unseen_classes)