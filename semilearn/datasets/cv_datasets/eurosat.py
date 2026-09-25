

import os, json
import glob
import numpy as np
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data
from torchvision.transforms.functional import InterpolationMode
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CORRECTION_DICT = {
    "annual crop land": "AnnualCrop",
    "brushland or shrubland": "HerbaceousVegetation",
    "highway or road": "Highway",
    "industrial buildings or commercial buildings": "Industrial",
    "pasture land": "Pasture",
    "permanent crop land": "PermanentCrop",
    "residential buildings or homes or apartments": "Residential",
    "lake or sea": "SeaLake",
    "river": "River",
    "forest": "Forest",
}

def get_eurosat(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    dataset_root_parent = os.path.join(data_dir, 'EuroSAT')
    image_root = os.path.join(dataset_root_parent, '2750')

    print(f"Loading EuroSAT data from: {image_root}")

    crop_size = args.img_size


    interpolation = InterpolationMode.BICUBIC


    transform_weak = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    class_names_path = os.path.join(dataset_root_parent, 'class_names.txt')
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"class_names.txt not found at {class_names_path}")

    with open(class_names_path) as f:
        classnames = [line.strip() for line in f.readlines()]

    classname_to_idx = {name: i for i, name in enumerate(classnames)}

    folder_to_classname = {v: k for k, v in CORRECTION_DICT.items()}

    all_train_paths = []
    all_train_targets = []

    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")

    for folder_name in os.listdir(image_root):
        folder_path = os.path.join(image_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        if folder_name not in folder_to_classname:
            print(f"Skipping unknown folder: {folder_name}")
            continue

        target_class_name = folder_to_classname[folder_name]

        if target_class_name not in classname_to_idx:
            print(f"Warning: Class name '{target_class_name}' derived from folder '{folder_name}' is not in class_names.txt")
            continue

        class_idx = classname_to_idx[target_class_name]

        class_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

        all_train_paths.extend(class_paths)
        all_train_targets.extend([class_idx] * len(class_paths))

    print(f"Total images loaded: {len(all_train_paths)}")
    if len(all_train_paths) == 0:
        raise RuntimeError("No images loaded! Check folder names vs correction_dict keys.")

    test_paths = []
    test_targets = []
    test_txt_path = os.path.join(dataset_root_parent, 'test.txt')

    if os.path.exists(test_txt_path):
        with open(test_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                path_part = parts[0]
                if '@' in path_part: path_part = path_part.split('@')[1]

                full_path = os.path.join(image_root, path_part)
                test_paths.append(full_path)

                if len(parts) > 1:
                    test_targets.append(int(parts[1]))
                else:
                    test_targets.append(-1)
    else:
        print("Warning: test.txt not found.")

    all_train_paths = np.array(all_train_paths)
    all_train_targets = np.array(all_train_targets, dtype=int)
    test_paths = np.array(test_paths)
    test_targets = np.array(test_targets, dtype=int)


    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args,
        all_train_paths,
        all_train_targets,
        num_classes,
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

    if len(test_paths) > 0:
        eval_dset = BasicDataset(alg=alg, data=test_paths, targets=test_targets, num_classes=num_classes,
                                 transform=transform_val, is_ulb=False)
        test_dset = BasicDataset(alg=alg, data=test_paths, targets=test_targets, num_classes=num_classes,
                                 transform=transform_val, is_ulb=False)
    else:

        eval_dset = BasicDataset(alg=alg, data=lb_data, targets=lb_targets, num_classes=num_classes,
                                 transform=transform_val, is_ulb=False)
        test_dset = eval_dset

    print(f"EuroSAT loaded correctly. Maps: Folder->Class Name->Index.")
    print(f"#Labeled: {len(lb_dset)} #Unlabeled: {len(ulb_dset)} #Val: {len(eval_dset)} ")

    return lb_dset, ulb_dset, eval_dset, test_dset, classnames

def get_eurosat_tl(args, alg, name, num_classes, data_dir='./data'):
    dataset_root_parent = os.path.join(data_dir, 'EuroSAT')
    image_root = os.path.join(dataset_root_parent, '2750')

    crop_size = args.img_size


    interpolation = InterpolationMode.BICUBIC


    transform_weak = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    class_names_path = os.path.join(dataset_root_parent, 'class_names.txt')
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"class_names.txt not found at {class_names_path}")

    with open(class_names_path) as f:
        classnames = [line.strip() for line in f.readlines()]
    classname_to_idx = {name: i for i, name in enumerate(classnames)}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    split_path = os.path.join(current_dir, 'split.json')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split.json not found at {split_path}")

    with open(split_path, 'r') as f:
        split_config = json.load(f)

    if 'eurosat' not in split_config:
        raise ValueError(f"Dataset key 'eurosat' not found in split.json.")

    seen_classes = set(split_config['eurosat']['seen'])
    unseen_classes = set(split_config['eurosat']['unseen'])

    seen_indices = set(classname_to_idx[c] for c in seen_classes if c in classname_to_idx)
    unseen_indices = set(classname_to_idx[c] for c in unseen_classes if c in classname_to_idx)

    folder_to_classname = {v: k for k, v in CORRECTION_DICT.items()}

    all_train_paths = []
    all_train_targets = []

    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")

    for folder_name in os.listdir(image_root):
        folder_path = os.path.join(image_root, folder_name)
        if not os.path.isdir(folder_path): continue
        if folder_name not in folder_to_classname: continue

        target_class_name = folder_to_classname[folder_name]
        if target_class_name not in classname_to_idx: continue

        class_idx = classname_to_idx[target_class_name]
        class_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

        all_train_paths.extend(class_paths)
        all_train_targets.extend([class_idx] * len(class_paths))

    test_paths = []
    test_targets = []
    test_txt_path = os.path.join(dataset_root_parent, 'test.txt')

    if os.path.exists(test_txt_path):
        with open(test_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                path_part = parts[0]
                if '@' in path_part: path_part = path_part.split('@')[1]
                full_path = os.path.join(image_root, path_part)
                test_paths.append(full_path)
                if len(parts) > 1:
                    test_targets.append(int(parts[1]))
                else:
                    test_targets.append(-1)

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

    print(f"[EuroSAT TL] Seen: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    print(f"[EuroSAT TL] #Labeled (Seen): {len(lb_dset)}, #Unlabeled (Unseen): {len(ulb_dset)}, #Test: {len(test_dset)}")

    return lb_dset, ulb_dset, test_dset, classnames, list(seen_classes), list(unseen_classes)