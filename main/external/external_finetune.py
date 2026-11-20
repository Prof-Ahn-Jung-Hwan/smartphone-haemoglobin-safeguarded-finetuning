# -*- coding: utf-8 -*-
"""Local fine-tuning at the external locations (India, Italy) — the engine used by the study.

Functions used by `main/analysis/external_evaluation.py` and the demo
  read_ext · build_args · load_model · predict · finetune

Command
  split   For each location, draw 50 fine-tuning candidates at random, ten times; participants who are not
          candidates form the evaluation set (India 45, Italy 72). The first n candidates are used for fine-tuning
          (n = 30 in the main analysis; 10–50 in the sensitivity analysis), so the evaluation set is the same for
          every n.

Fine-tuning protocol (identical for every configuration)
  Backbone weights frozen; the layers after the backbone (S1, S2, head) updated with AdamW for 30 epochs,
  batch = n // 3, learning rate 1e-4 (--few-lr); the epoch with the smallest MAE in the fine-tuning participants
  is kept. Without S3 the model is in training mode, so batch-normalisation statistics are updated from the
  fine-tuning batches; S3 (`finetune(..., freeze_extra="bn")`) keeps those layers in evaluation mode.

Outputs  main/results/external/splits/<location>/rep<r>/{eval.csv, calib_random.csv}
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; MAIN = HERE.parent; ROOT = MAIN.parent
CODE = ROOT / "code"; TRAIN = MAIN / "results/stage5/train"     # trained development models (overridden by callers)
DATA = HERE / "data"; OUT = MAIN / "results/external"
SITES = ["India", "Italy"]; NS = [10, 20, 30, 40, 50]; POOL = 50


# ─────────────────────────── common ───────────────────────────
def setup():
    # Several workers share the CPU; the thread count comes from OMP_NUM_THREADS (default 4).
    # Results do not depend on the number of threads.
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    os.chdir(CODE)
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))


def read_ext(inp):
    """External photographs of one input region (data/arm_ext_<input>.csv: key · Hb · Country · full_path)."""
    d = pd.read_csv(DATA / f"arm_ext_{inp}.csv"); d.columns = [c.lstrip("\ufeff") for c in d.columns]
    miss = [p for p in d.full_path if not Path(p).exists()]
    if miss:
        sys.exit(f"{inp}: {len(miss)} external images missing, e.g. {miss[:3]}")
    return d


def build_args(name, fold, device, epochs=None, lr=None):
    import yaml
    from types import SimpleNamespace
    from ordinal_regression import create_ordinal_lookup
    ns = SimpleNamespace(**yaml.safe_load(open(TRAIN / f"{name}-fold{fold}" / "config.yaml")))
    ns.device = device
    if epochs is not None: ns.epochs = epochs
    if lr is not None: ns.lr = float(lr)
    ns.lookup = create_ordinal_lookup(ns)
    return ns


def load_model(args, ck):
    import torch
    from model import HbNet
    m = HbNet(args).to(args.device)
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval()


def to_samples(df):
    return [(Path(p), {"Hb": float(h), "Hybrid": []}) for p, h in zip(df.full_path, df.Hb)]


_CACHE = {}


def make_dataset(args, samples, mode):
    """Same output as the training dataset, faster: the first transform is a fixed Resize (224) and the random
    transforms come after it, so each photograph is resized once and cached; Resize uses no random numbers,
    so the random stream is unchanged."""
    import torch, torchvision.transforms as T
    from PIL import Image
    from dataset.ajoumc_anemia import AjouMC_AnemiaDataset

    class _Cached(AjouMC_AnemiaDataset):
        def __init__(s, *a, **k):
            super().__init__(*a, **k)
            ts = s.transform.transforms
            assert isinstance(ts[0], T.Resize), "first transform is not Resize — caching would change the images"
            s._resize, s._rest = ts[0], T.Compose(ts[1:])

        def __getitem__(s, i):
            path, data = s.data[i]
            key = (str(path), tuple(s._resize.size))
            im = _CACHE.get(key)
            if im is None:
                im = s._resize(Image.open(path).convert("RGB")); _CACHE[key] = im
            img = s._rest(im)
            hb = torch.tensor([min(data["Hb"], s.args.Hb_maxima)], dtype=torch.float)
            hyb = torch.FloatTensor(data["Hybrid"]) if s.hybrid_mode else torch.zeros(1)
            return (img, hb, hyb, path.name) if s.w_filename else (img, hb, hyb)

    return _Cached(args, samples, mode=mode, w_filename=True)


def predict(args, model, df, batch=64):
    import torch
    from torch.utils.data import DataLoader
    from dataset.ajoumc_anemia import AjouMC_AnemiaDataset
    from ordinal_regression import decode_ordinal
    dl = DataLoader(make_dataset(args, to_samples(df), "test"),
                    batch_size=batch, shuffle=False, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for img, _, anno, _ in dl:
            o = model(img.to(args.device), anno.to(args.device))
            p = decode_ordinal(args, o["logits"]) if args.use_ordinal_regression else o["logits"]
            out += p.squeeze(-1).cpu().tolist()
    return pd.DataFrame(dict(key=df.key.to_numpy(), y=df.Hb.to_numpy(), pred=out, Country=df.Country.to_numpy()))




# ─────────────────────────── split ───────────────────────────
def hb_order(pool, rng):
    """Alternative order stratified by haemoglobin (written as calib_hbstrat.csv; not used in the paper)."""
    q = pd.qcut(pool.Hb.rank(method="first"), 5, labels=False)
    bins = [list(rng.permutation(pool.index[q == b].to_numpy())) for b in range(5)]
    order = []
    while any(bins):
        for b in bins:
            if b: order.append(b.pop(0))
    return order


def cmd_split(a):
    base = read_ext("auto_scca")[["key", "Hb", "Country"]]
    rows = []
    for si, site in enumerate(SITES):
        S = base[base.Country == site].reset_index(drop=True)
        for r in range(a.reps):
            rng = np.random.default_rng(20260918 + 1000 * r + si)
            perm = rng.permutation(len(S))
            pool, ev = S.loc[perm[:POOL]], S.loc[perm[POOL:]]
            d = OUT / "splits" / site / f"rep{r}"; d.mkdir(parents=True, exist_ok=True)
            ev.to_csv(d / "eval.csv", index=False)
            orders = {"random": list(pool.index), "hbstrat": hb_order(pool, rng)}
            for st, o in orders.items():
                c = S.loc[o].reset_index(drop=True); c.insert(0, "draw_order", range(1, len(c) + 1))
                c.to_csv(d / f"calib_{st}.csv", index=False)
                for n in NS:
                    rows.append(dict(site=site, rep=r, order=st, n=n, n_eval=len(ev), hb_finetune=c.Hb[:n].mean(),
                                     hb_eval=ev.Hb.mean(), hb_difference=c.Hb[:n].mean() - ev.Hb.mean()))
    R = pd.DataFrame(rows); R.to_csv(OUT / "splits/split_summary.csv", index=False)
    print(R.groupby(["site", "order", "n"]).agg(n_eval=("n_eval", "first"), hb_difference=("hb_difference", "mean"),
          hb_difference_sd=("hb_difference", "std")).round(2).to_string())
    print(f"\nsaved: {OUT/'splits'}")




# ─────────────────────────── fine-tuning ───────────────────────────
def finetune(args, model, init_sd, calib, a, seed):
    import torch
    from torch.utils.data import DataLoader
    from dataset.ajoumc_anemia import AjouMC_AnemiaDataset, calc_imbalance
    from misc import MAE_error, adjust_learning_rate, setup_seed
    from model import create_optimizer
    from model.loss import AnemiaLoss
    from ordinal_regression import decode_ordinal, encode_ordinal
    setup_seed(seed)
    model.load_state_dict(init_sd); model.train()
    for p in model.backbone.parameters(): p.requires_grad = False
    # S3: with freeze_extra="bn" the backbone's batch-normalisation layers are kept in evaluation mode, so their
    # running statistics are not updated from the local photographs. Without it, model.train() updates them
    # even though the backbone weights are frozen.
    _fe = str(getattr(a, "freeze_extra", "") or getattr(args, "freeze_extra", "") or "")
    _bn_fix = "bn" in _fe
    def _bn_eval():
        if not _bn_fix: return
        for _m in model.backbone.modules():
            if isinstance(_m, torch.nn.modules.batchnorm._BatchNorm): _m.eval()
    _bn_eval()
    data = to_samples(calib); batch = max(2, len(data) // 3)
    dl = DataLoader(make_dataset(args, data, "train"),
                    batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    _, cls_num = calc_imbalance(args, data)
    opt, sch = create_optimizer(args, model, len(dl)); loss_fn = AnemiaLoss(args, cls_num)
    best, step = float("inf"), 0
    n_updates = 0                    # optimizer steps actually taken
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(args.epochs):
        adjust_learning_rate(opt, ep, args); model.train(); _bn_eval(); seen, tot = 0, 0.0
        for img, target, anno, fname in dl:
            model.zero_grad()
            img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
            o = model(img, anno, encode_ordinal(args, target).long())
            loss = loss_fn(o, target, ep >= args.start_ib_epoch)
            if not torch.isfinite(loss): opt.zero_grad(); continue      # skip NaN or inf losses
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            n_updates += 1
            if sch is not None: sch.step_update(step)
            step += 1
            p = decode_ordinal(args, o["logits"]) if args.use_ordinal_regression else o["logits"]
            tot += MAE_error(p, target).sum().item(); seen += len(target)
        if seen and tot / seen < best:
            best = tot / seen; best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # never return a model that was not actually fine-tuned
    if n_updates == 0 or not np.isfinite(best):
        raise RuntimeError(f"fine-tuning failed — {n_updates} updates, best MAE {best} (seed {seed})")
    model.load_state_dict(best_sd)
    for p in model.parameters(): p.requires_grad = True   # unfreeze everything for the next fold model
    return model


if __name__ == "__main__":
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--reps", type=int, default=10)
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("split", parents=[c]).set_defaults(func=cmd_split)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); a.func(a)
