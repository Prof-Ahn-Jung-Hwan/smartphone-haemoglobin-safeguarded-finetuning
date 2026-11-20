from collections import defaultdict
from math import isnan
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from ordinal_regression import encode_ordinal


class AjouMC_AnemiaDataset(Dataset):
    def __init__(self, args, data, mode="train", w_filename=False):
        self.args = args
        self.hybrid_mode = args.hybrid_mode
        self.data = data
        self.mode = mode
        self.w_filename = w_filename

        targets = torch.tensor([anno["Hb"] for _, anno in data])
        self.targets = encode_ordinal(args, targets).long()

        # augmentation (training) / resize and normalise only (evaluation)
        self.transform = (
            T.Compose(
                [
                    T.Resize((args.img_size, args.img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.4),
                    T.RandomPerspective(distortion_scale=0.3, p=0.4),
                    T.RandomApply(
                        [
                            T.RandomAffine(
                                degrees=45,
                                translate=(0.2, 0.2),
                                scale=(0.7, 1.3),
                                shear=15,
                                interpolation=T.InterpolationMode.BILINEAR,
                            )
                        ],
                        p=0.8,
                    ),
                    T.RandomApply([T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5))], p=0.3),
                    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                    T.RandomAdjustSharpness(sharpness_factor=0.5, p=0.3),
                    T.ToTensor(),
                    T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            if mode == "train"
            else T.Compose(
                [
                    T.Resize((args.img_size, args.img_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        )

    def __getitem__(self, index):
        img, Hb, hybrid_anno = torch.zeros(1), torch.zeros(1), torch.zeros(1)

        img_path, data = self.data[index]
        Hb = data["Hb"]

        if self.hybrid_mode:
            hybrid_anno = torch.FloatTensor(data["Hybrid"])

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        Hb = min(Hb, self.args.Hb_maxima)
        Hb = torch.tensor([Hb], dtype=torch.float)

        if self.w_filename:
            return (img, Hb, hybrid_anno, img_path.name)
        else:
            return (img, Hb, hybrid_anno)

    def __len__(self):
        return len(self.data)


OUTER_SEED = 2026   # fixed outer split: every backbone, configuration, input and seed uses the same test folds


def split_by_participant(hb_by_participant, cv_fold, seed, val_inner=0, outer_seed=OUTER_SEED):
    """Participant-level stratified random split: {number: integer Hb} -> per fold (train, validation, test) sets of numbers.

    Each participant has one haemoglobin value, so participants (not photographs) are split with
    StratifiedKFold(shuffle=True) and both photographs of a participant follow their owner.
      outer folds : fixed by outer_seed (independent of seed)
      inner validation set : the validation part of the first split of the training participants into val_inner folds;
                             differs between seeds (no validation set if val_inner <= 1)
    The fold roster (data/splits/) is written by experiments/make_fold_roster.py with this function."""
    import numpy as np
    u = np.array(sorted(hb_by_participant)); y = np.array([hb_by_participant[n] for n in u])
    out = []
    for tr, te in StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=outer_seed).split(u, y):
        va = np.array([], dtype=int)
        if val_inner and val_inner > 1:
            a, b = next(StratifiedKFold(n_splits=val_inner, shuffle=True, random_state=seed).split(u[tr], y[tr]))
            tr, va = tr[a], tr[b]
        out.append((set(u[tr].tolist()), set(u[va].tolist()), set(u[te].tolist())))
    return out


def load_ajoumc_seghgb_dataset(args, cv_fold=5, seed=42, w_filename=False):
    def _load_anno(anno_path):
        """Read the Excel roster and pair each image path with its annotation."""
        df = pd.read_excel(anno_path, engine="openpyxl")
        
        # random seed for eye selection
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        # group the photographs by participant (study number)
        number_data = defaultdict(list)
        for _, row in df.iterrows():
            file_path = row['file_path']
            number = row['number']
            separator = row['separator']
            Hb = row['Hb']
            
            if not isnan(Hb):
                # relative path -> absolute path
                img_full_path = root.joinpath(file_path)
                
                anno_dict = {"Hb": float(Hb)}
                
                # hybrid annotations (not used in the study)
                if args.hybrid_mode and hybrid_anno_path is not None:
                    anno_dict["Hybrid"] = []
                
                number_data[number].append((img_full_path, separator, anno_dict))
        
        # select photographs according to args.eye
        for number, images in number_data.items():
            if args.eye == "left":
                # separator 1 only
                selected = [(img, anno) for img, sep, anno in images if sep == 1]
            elif args.eye == "right":
                # separator 2 only
                selected = [(img, anno) for img, sep, anno in images if sep == 2]
            elif args.eye == "exclusive":
                # participants with both eyes; both photographs used (the study setting)
                separators = [sep for _, sep, _ in images]
                if 1 not in separators or 2 not in separators:
                    continue
                selected = [(img, anno) for img, sep, anno in images if sep in [1, 2]]
            elif args.eye == "random":
                # one eye chosen at random per participant
                left_images = [(img, anno) for img, sep, anno in images if sep == 1]
                right_images = [(img, anno) for img, sep, anno in images if sep == 2]
                
                if left_images and right_images:
                    selected = random.choice([left_images, right_images])
                elif left_images:
                    selected = left_images
                elif right_images:
                    selected = right_images
                else:
                    selected = []
            elif args.eye == "image_level":
                # image-level split: every photograph independent
                selected = [(img, anno) for img, sep, anno in images]
            else:
                # all photographs
                selected = [(img, anno) for img, sep, anno in images]
            
            for img, anno in selected:
                data.append((img, anno))
                targets.append(int(anno["Hb"]))
                groups.append(int(number))  # participant, for the participant-level split

    root = Path(args.dataset_root)
    anno_path = root.joinpath(args.anno_path)
    hybrid_anno_path = root.joinpath(args.hybrid_anno_path) if args.hybrid_mode else None
    
    data = []
    targets = []  # integer Hb, for stratification
    groups = []  # participant number of each photograph

    _load_anno(anno_path)
    
    train_data_list = []
    test_data_list = []
    imbalance_list = []
    img_num_list = []

    # Inner validation set: with val_inner=K, one K-th of the training participants of the fold is set aside.
    #   five folds, K=8 -> train 70 · validation 10 · test 20 (7:1:2);  ten folds, K=9 -> 80 · 10 · 10 (8:1:1)
    #   best.ckpt is chosen on the validation set; the test fold is scored once, after training.
    _K = int(getattr(args, "val_inner", 0) or 0)
    split_mode = getattr(args, "split_mode", None) or "person_skf"

    # (train, validation or None, test) photograph indices for each fold
    import numpy as _np
    if args.eye == "image_level":
        # image-level split (photographs, not participants; not used in the study)
        skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=seed)
        splits = [(tr, None, te) for tr, te in skf.split(data, targets)]
    elif split_mode == "person_skf":
        # participant-level stratified split (both eyes of a participant always in the same fold)
        hb_by_participant = {}
        for gi, t in zip(groups, targets):
            if hb_by_participant.setdefault(gi, t) != t:
                raise ValueError(f"participant {gi} has photographs with different Hb values")
        _g = _np.asarray(groups)
        rows_of = lambda s: _np.where(_np.isin(_g, sorted(s)))[0]
        splits = [(rows_of(tr), rows_of(va) if va else None, rows_of(te))
                  for tr, va, te in split_by_participant(hb_by_participant, cv_fold, seed, _K)]
    else:
        raise ValueError(f"unknown split_mode {split_mode} (person_skf)")

    _val_list = []
    for train_index, val_index, test_index in splits:
        fold_train = [data[i] for i in train_index]
        fold_test = [data[i] for i in test_index]
        if _K > 1 and val_index is not None:
            _val_list.append(AjouMC_AnemiaDataset(args, [data[i] for i in val_index], mode="test", w_filename=w_filename))

        train_dataset = AjouMC_AnemiaDataset(args, fold_train, mode="train", w_filename=w_filename)
        test_dataset = AjouMC_AnemiaDataset(args, fold_test, mode="test", w_filename=w_filename)
        imbalance, _N_SAMPLES_PER_CLASS = calc_imbalance(args, fold_train)

        train_data_list.append(train_dataset)
        test_data_list.append(test_dataset)
        imbalance_list.append(imbalance)
        img_num_list.append(_N_SAMPLES_PER_CLASS)

    args._val_folds = _val_list if _val_list else None
    return train_data_list, test_data_list, imbalance_list, img_num_list


def calc_imbalance(args, data):
    total = len(data) + 1
    imbalance = [1] * (args.ordinal_bins - 1)
    N_SAMPLES_PER_CLASS = [0] * (args.ordinal_bins - 1)
    for _, anno in data:
        Hb = anno["Hb"]
        shifted_Hb = Hb + args.disc_shift
        for k in range(args.ordinal_bins - 1):
            imbalance[k] += 1
            N_SAMPLES_PER_CLASS[k] += 1
            if shifted_Hb >= args.lookup[k] and shifted_Hb < args.lookup[k + 1]:
                break

    imbalance_ratio = []
    for j in range(len(imbalance)):
        imbalance_ratio.append(total / imbalance[j])
    return torch.FloatTensor(imbalance_ratio), N_SAMPLES_PER_CLASS
