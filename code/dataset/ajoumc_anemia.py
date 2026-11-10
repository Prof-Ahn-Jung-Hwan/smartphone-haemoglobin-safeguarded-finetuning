from collections import defaultdict
from math import isnan
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
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

        # ★ 09-24 음성 대조 B — **훈련 자료에서만** 정렬용 라벨을 환자 단위로 섞는다.
        #   · 회귀 정답(Hb)은 건드리지 않는다. 특징을 어느 기준점으로 모을지만 바뀐다
        #   · 같은 환자의 양 눈은 같은 라벨을 받는다 (환자 단위로 섞으므로 자동)
        #   · 한 번 정하면 학습이 끝날 때까지 고정 (epoch·batch 마다 다시 섞지 않는다)
        #   · 환자별 구간 인원수는 유지된다 (값을 재배정하는 순열이므로)
        self.align = None
        if getattr(args, "shuffle_align", False) and mode == "train":
            import numpy as _np, pandas as _pd
            _df = _pd.read_excel(args.anno_path, engine="openpyxl")
            _s2n = {Path(str(f)).stem: n for f, n in zip(_df.file_path, _df.number)}
            # 🔴 09-24 : 외부 보정(인도·이탈리아)도 mode="train" 으로 이 데이터셋을 쓴다. 그 사진들은 내부 명부에 없다.
            #   그 경로는 **외부 쪽에서 따로 배정**하므로(260918_external_grid.finetune 의 align_map) 여기서는 건너뛴다.
            _miss = [p_ for p_, _ in data if Path(str(p_)).stem not in _s2n]
            if _miss:
                print(f"⚠ 정렬 라벨 섞기 건너뜀 — 내부 명부에 없는 사진 {len(_miss)}장 (외부 보정 경로는 자체 배정을 쓴다)")
                # 정렬 배정만 건너뛴다. 아래 영상 변환 초기화는 외부 보정에도 필요하다.
            else:
                _nums = [_s2n[Path(str(p_)).stem] for p_, _ in data]
                _hb = {}
                for _n, (_p, _a) in zip(_nums, data):
                    _hb.setdefault(_n, float(_a["Hb"]))
                _keys = sorted(_hb); _vals = [_hb[k] for k in _keys]
                _rng = _np.random.RandomState(int(getattr(args, "align_seed", 20260924)) + int(getattr(args, "fold", 0)))
                _perm = _rng.permutation(len(_vals))
                _shuf = {k: _vals[_perm[i]] for i, k in enumerate(_keys)}       # 환자 → 섞인 '정렬용 Hb'
                self.align = [_shuf[n] for n in _nums]
                _same = sum(1 for n in _keys if abs(_shuf[n] - _hb[n]) < 1e-9)
                print(f"★ 음성 대조 B — 정렬 라벨 섞음 : 환자 {len(_keys)}명 · 우연히 같은 값 {_same}명 "
                      f"· 난수 {int(getattr(args, 'align_seed', 20260924)) + int(getattr(args, 'fold', 0))}")

        # Augmentation 파이프라인 변경
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
        elif self.align is not None:                      # ★ 음성 대조 B — 정렬용 Hb 를 함께 준다
            return (img, Hb, hybrid_anno, torch.tensor([self.align[index]], dtype=torch.float))
        else:
            return (img, Hb, hybrid_anno)

    def __len__(self):
        return len(self.data)


OUTER_SEED = 2026   # ★ 2026-10-28 바깥 겹 고정값 — 한 번 정하고 바꾸지 않는다 (모든 백본 · 구성 · 입력 · seed 가 같은 시험 겹)


def 사람분할(사람_Hb, cv_fold, seed, val_inner=0, outer_seed=OUTER_SEED):
    """★ 2026-10-28 사람 단위 층화 무작위 분할 — {number: 정수 Hb} → 겹마다 (학습, 검증, 시험) 의 number 집합.

    scikit-learn 1.6.1 의 StratifiedGroupKFold(shuffle=True) 는 층화가 깨진다 (issue #32478 · 실측 10-28).
    사람마다 Hb 가 하나이므로 사람 한 줄로 놓고 StratifiedKFold(shuffle=True) 로 나눈 뒤 사진으로 펼친다.
      바깥 : outer_seed 로 고정 (seed 와 무관)
      안쪽 : 학습 몫 사람을 val_inner 겹으로 나눈 첫 분할의 검증 몫 · seed 마다 다르다 (val_inner ≤ 1 이면 검증 없음)
    분할 명부 (data/splits/) 는 experiments/00_분할명부_만들기.py 가 이 함수로 만든다."""
    import numpy as np
    u = np.array(sorted(사람_Hb)); y = np.array([사람_Hb[n] for n in u])
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
        """Excel 파일에서 데이터를 읽어와 이미지 경로와 annotation을 매칭"""
        df = pd.read_excel(anno_path, engine="openpyxl")
        
        # Eye selection을 위한 랜덤 시드 설정
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        # number마다 데이터 재분류 (patient id 대신 number 사용)
        number_data = defaultdict(list)
        for _, row in df.iterrows():
            file_path = row['file_path']
            number = row['number']
            separator = row['separator']
            Hb = row['Hb']
            
            if not isnan(Hb):
                # 상대 경로를 절대 경로로 변환
                img_full_path = root.joinpath(file_path)
                
                anno_dict = {"Hb": float(Hb)}
                
                # hybrid mode 처리 (필요시 확장 가능)
                if args.hybrid_mode and hybrid_anno_path is not None:
                    # hybrid annotation 처리 로직 추가 가능
                    anno_dict["Hybrid"] = []
                
                number_data[number].append((img_full_path, separator, anno_dict))
        
        # Eye selection 옵션에 따라 데이터 필터링
        for number, images in number_data.items():
            if args.eye == "left":
                # 좌안(separator=1)만 선택
                selected = [(img, anno) for img, sep, anno in images if sep == 1]
            elif args.eye == "right":
                # 우안(separator=2)만 선택
                selected = [(img, anno) for img, sep, anno in images if sep == 2]
            elif args.eye == "exclusive":
                # 1과 2가 모두 있는 number만 선택하고, 1과 2 모두 사용
                separators = [sep for _, sep, _ in images]
                if 1 not in separators or 2 not in separators:
                    # 한쪽 눈만 있는 경우 제외
                    continue
                selected = [(img, anno) for img, sep, anno in images if sep in [1, 2]]
            elif args.eye == "random":
                # 각 number에서 1 또는 2 중 하나를 랜덤하게 선택
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
                # Image-level split: 모든 이미지를 독립적으로 사용
                selected = [(img, anno) for img, sep, anno in images]
            else:
                # 모든 이미지 사용
                selected = [(img, anno) for img, sep, anno in images]
            
            # 선택된 이미지들을 data 리스트에 추가
            for img, anno in selected:
                data.append((img, anno))
                targets.append(int(anno["Hb"]))
                groups.append(int(number))  # Patient-level split을 위한 group 정보

    root = Path(args.dataset_root)
    anno_path = root.joinpath(args.anno_path)
    hybrid_anno_path = root.joinpath(args.hybrid_anno_path) if args.hybrid_mode else None
    
    data = []
    targets = []  # Stratification을 위한 target 리스트
    groups = []  # Patient-level split을 위한 group 리스트

    _load_anno(anno_path)
    
    train_data_list = []
    test_data_list = []
    imbalance_list = []
    img_num_list = []

    # ★ 2026-09-12 검증 분할 (기본은 꺼짐 · 10-28 부터 분할 방식은 split_mode — 옛 실행 재현은 sgkf_old)
    #   val_inner=K 이면 이 겹의 학습 몫에서 **사람(number) 단위로 1/K** 을 검증으로 뗀다.
    #   5겹·K=8 -> 학습 70 · 검증 10 · 시험 20 (7:1:2)   10겹·K=9 -> 80 · 10 · 10 (8:1:1)
    #   best.ckpt 는 검증으로 고르고, 시험 겹은 **학습이 끝난 뒤 한 번만** 채점한다.
    _K = int(getattr(args, "val_inner", 0) or 0)
    split_mode = getattr(args, "split_mode", None) or "person_skf"

    # Eye 옵션에 따라 split 방법 선택 → (학습, 검증 또는 None, 시험) 사진 색인
    import numpy as _np
    if args.eye == "image_level":
        # Image-level split: StratifiedKFold 사용 (groups 없음)
        skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=seed)
        splits = [(tr, None, te) for tr, te in skf.split(data, targets)]
    elif split_mode == "person_skf":
        # ★ 2026-10-28 기본 — 사람 단위 층화 무작위 분할 (사람분할 참고 · 두 눈은 저절로 같은 겹)
        사람_Hb = {}
        for gi, t in zip(groups, targets):
            if 사람_Hb.setdefault(gi, t) != t:
                raise ValueError(f"number {gi} 의 사진마다 Hb 가 다르다 — 사람 단위 층화를 쓸 수 없다")
        _g = _np.asarray(groups)
        색인 = lambda s: _np.where(_np.isin(_g, sorted(s)))[0]
        splits = [(색인(tr), 색인(va) if va else None, 색인(te))
                  for tr, va, te in 사람분할(사람_Hb, cv_fold, seed, _K)]
    elif split_mode == "sgkf_old":
        # 옛 방식 (~10-27 · 옛 결과 재현용) — 바깥 StratifiedGroupKFold(shuffle=False) · 안쪽 shuffle=True
        #   🔴 scikit-learn 1.6.1 의 StratifiedGroupKFold(shuffle=True) 는 층화가 깨진다 (안쪽 검증이 사실상 무작위)
        gkf = StratifiedGroupKFold(n_splits=cv_fold, shuffle=False)
        splits = []
        for tr, te in gkf.split(data, targets, groups=groups):
            va = None
            if _K > 1:
                _inner = StratifiedGroupKFold(n_splits=_K, shuffle=True, random_state=seed)
                _a, _b = next(_inner.split(tr, [targets[i] for i in tr], groups=[groups[i] for i in tr]))
                tr, va = tr[_a], tr[_b]
            splits.append((tr, va, te))
    else:
        raise ValueError(f"split_mode {split_mode} 를 모른다 (person_skf · sgkf_old)")

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
