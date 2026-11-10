import numpy as np
import torch

from .loss import OrdinalReweightingLoss


def create_ordinal_lookup(args):
    minima = args.Hb_minima + args.disc_shift
    maxima = args.Hb_maxima + args.disc_shift
    bins = args.ordinal_bins

    # ★ 09-25 : 구간 간격 스위치. 기본값 'log' = 지금까지와 완전히 같다.
    #   'linear' 는 "왜 로그인가" 를 가르는 대조군 (간격만 바꾸고 나머지는 동일)
    if str(getattr(args, "bin_spacing", "log")).lower() == "linear":
        lookup_vals = np.linspace(minima, maxima, bins)
    else:
        lookup_vals = np.exp(np.linspace(np.log(minima), np.log(maxima), bins))
    lookup_table = {k: float(v) for k, v in enumerate(lookup_vals)}
    return lookup_table


_SCRAMBLE_MAP = {}


def _scramble_map(args):
    """★ 09-24 음성 대조 — Hb 를 잘게 썬 조각마다 **무작위 구간 이름표**를 고정 배정한다.

    왜 이렇게 하나 : 기준점(anchor)들은 서로 직교·대등해서 **이름표만 바꾸면 아무것도 바뀌지 않는다**
    (대칭이라 학습 난이도가 같다). 그래서 이름표가 아니라 **"어떤 Hb 들이 한 구간에 묶이는가"** 를 흩는다.
    원래는 한 구간 = 붙어 있는 Hb 범위인데, 이 대조에서는 한 구간 = **여기저기 흩어진 Hb 조각들**이 된다.
    즉 '구간이 Hb 순서를 따른다' 는 성질만 없애고, 구간 수·라벨 분포·학습 난이도는 최대한 유지한다.
    """
    key = (args.ordinal_bins, int(getattr(args, "bin_scramble_seed", 0)), int(getattr(args, "bin_scramble_fine", 8)))
    if key not in _SCRAMBLE_MAP:
        K, seed, fine = key
        g = np.random.RandomState(seed)
        lab = np.tile(np.arange(K), fine)          # 각 구간이 fine 조각씩 갖도록
        g.shuffle(lab)
        _SCRAMBLE_MAP[key] = lab                    # 잘게 썬 조각 → 구간 이름표
    return _SCRAMBLE_MAP[key]


def encode_ordinal(args, decoded_labels: torch.Tensor):
    decoded_labels = decoded_labels + args.disc_shift

    encoded_labels = torch.zeros_like(decoded_labels)

    for k in range(args.ordinal_bins - 1):
        mask = (decoded_labels >= args.lookup[k]) & (decoded_labels < args.lookup[k + 1])
        encoded_labels[mask] = k

    mask = decoded_labels >= args.Hb_maxima + args.disc_shift
    encoded_labels[mask] = args.ordinal_bins - 1

    mode = str(getattr(args, "bin_scramble", "") or "")
    if mode == "fine":                              # ★ 음성 대조 : Hb ↔ 구간 대응을 흩는다
        K = args.ordinal_bins
        fine = int(getattr(args, "bin_scramble_fine", 8))
        lo, hi = float(args.lookup[0]), float(args.lookup[K - 1])
        x = decoded_labels.clamp(min=lo, max=hi)
        t = (torch.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))      # 로그 축에서 0~1
        idx = (t * (K * fine - 1)).round().long().clamp(0, K * fine - 1)  # 잘게 썬 조각 번호
        lab = torch.as_tensor(_scramble_map(args), device=decoded_labels.device)
        encoded_labels = lab[idx].to(encoded_labels.dtype)

    return encoded_labels


def decode_ordinal(args, pred: torch.Tensor):
    pred = pred.cpu().detach()
    B, K, _ = pred.shape
    assert K == args.ordinal_bins - 1, "Mismatch in number of bins"

    p_geq = torch.softmax(pred, -1)[..., -1]

    p_class = []
    for k in range(K):
        if k < K - 1:
            p_k = p_geq[:, k] - p_geq[:, k + 1]
        else:
            p_k = p_geq[:, k]
        p_class.append(p_k)

    p_class = torch.stack(p_class, dim=1)

    midpoints = torch.tensor(
        [(args.lookup[k] + args.lookup[k + 1]) / 2.0 for k in range(K)],
        device=pred.device,
        dtype=pred.dtype,
    ).unsqueeze(0)

    expected_hb = torch.sum(p_class * midpoints, dim=1, keepdim=True)
    return expected_hb
