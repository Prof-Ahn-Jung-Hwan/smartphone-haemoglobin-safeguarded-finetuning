"""
Ajou University Medical Center & ETRI Anemia Analysis AI Project

This script is a modified version of train.py, specifically for
Bayesian optimization of 'bins' hyperparameter.
- Based on train_original.py structure
- Uses Mixed Precision Training
- Measures and saves training time
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from misc import MAE_error, adjust_learning_rate, log_data_split, plot_logs, setup_seed
from model import HbNet, create_optimizer
from model.loss import AnemiaLoss
from ordinal_regression import create_ordinal_lookup, decode_ordinal, encode_ordinal


def main(args):

    args.lookup = create_ordinal_lookup(args)
    print(f"Ordinal lookup table: {args.lookup}")

    train_folds, test_folds, imbalance_folds, cls_num_list_folds = load_dataset(
        args, cv_fold=int(getattr(args, "cv_fold", None) or 5), seed=args.seed
    )
    VAL = getattr(args, "_val_folds", None) is not None          # ★ 09-12 검증 분할을 쓰나

    log_data_split(args, train_folds[args.fold], test_folds[args.fold])

    args.seed += args.fold
    setup_seed(args.seed)

    train_sampler = None

    train_dataloader = DataLoader(
        train_folds[args.fold],
        batch_size=args.train_batch_size,
        num_workers=args.train_workers,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=(train_sampler is None),
        persistent_workers=True,
        batch_sampler=train_sampler,
    )
    test_dataloader = DataLoader(
        test_folds[args.fold],
        batch_size=args.test_batch_size,
        num_workers=args.test_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    cls_num_list = cls_num_list_folds[args.fold]
    val_dataloader = None
    if VAL:
        val_ds = args._val_folds[args.fold]
        val_dataloader = DataLoader(val_ds, batch_size=args.test_batch_size, num_workers=args.test_workers,
                                    shuffle=False, pin_memory=True, drop_last=False)
        with open(os.path.join(args.log_dir, "val.txt"), "w") as f:
            for pth, anno in val_ds.data:
                f.write(f"{pth}\t{anno['Hb']}\n")
        print(f"★ 검증 분할 : 학습 {len(train_folds[args.fold])} · 검증 {len(val_ds)} · 시험 {len(test_folds[args.fold])}")

    print(f"Class number list: {cls_num_list}")

    model = HbNet(args).to(args.device)
    optimizer, scheduler = create_optimizer(args, model, len(train_dataloader))
    scaler = torch.amp.GradScaler()  # mixed precision 학습을 위해 scaler 사용
    loss_fn = AnemiaLoss(args, cls_num_list)

    xs, ys1, ys2 = [], [], []
    txs, tys1, tys2 = [], [], []
    best_test_mae = float("inf")
    global_step = 0
    
    # 전체 학습 시간 측정 시작
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_result, global_step = train_one_epoch(
            args, train_dataloader, model, optimizer, scaler, loss_fn, scheduler, epoch, global_step
        )  # scaler 입력부분 추가

        xs.append(epoch)
        ys1.append(train_result["train/loss"])
        ys2.append(train_result["train/mae"])
        
        
        if (epoch + 1) % 5 == 0:
            # ★ 검증을 쓰면 best 는 **검증 겹**으로 고른다. 시험 겹은 여기서 열지 않는다
            test_result = test_one_epoch(args, val_dataloader if VAL else test_dataloader, model, loss_fn, epoch)
            txs.append(epoch)
            tys1.append(test_result["test/loss"])
            tys2.append(test_result["test/mae"])

            if test_result["test/mae"] < best_test_mae:
                best_test_mae = test_result["test/mae"]
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(args.log_dir, f"best.ckpt"),
                )

            print(
                f'TEST REPORT >> loss: {test_result["test/loss"]:.4f}  mae: {test_result["test/mae"]:.4f}   best valid mae: {best_test_mae:.4f}'
            )

        plot_logs(
            xs,
            txs,
            ys1,
            ys2,
            tys1,
            tys2,
            best_mae=best_test_mae,
            fig_name=os.path.join(args.log_dir, "log.png"),
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
            },
            os.path.join(args.log_dir, "last.ckpt"),
        )
    
    # ★ 09-12 검증을 썼으면 — 학습이 끝난 뒤 시험 겹을 **한 번만** 채점한다 (마지막 회차 · 검증으로 고른 회차)
    extra = ""
    # ★ 09-14 — lr 이 너무 커 검증 MAE 가 끝까지 nan 이면 best.ckpt 가 한 번도 저장되지 않아 아래에서 죽는다.
    #   그 경우에만 last.ckpt 를 best 로 복사하고 표시를 남긴다 → 겹이 "실패" 가 아니라 "성적 inf" 로 끝나
    #   lr 선택에서 자연히 탈락한다. 정상 학습(한 번이라도 best 가 저장된 경우)에는 아무 영향이 없다.
    if not os.path.exists(os.path.join(args.log_dir, "best.ckpt")) and os.path.exists(os.path.join(args.log_dir, "last.ckpt")):
        import shutil
        shutil.copy(os.path.join(args.log_dir, "last.ckpt"), os.path.join(args.log_dir, "best.ckpt"))
        with open(os.path.join(args.log_dir, "DIVERGED.txt"), "w") as f:
            f.write("검증 MAE 가 한 번도 유한값이 아니었다 (발산). last.ckpt 를 best.ckpt 로 복사했다.\n")
        print("⚠️ 발산 — best.ckpt 없음 → last.ckpt 를 best 로 사용")
    if VAL:
        last_mae, last_pred = predict_all(args, test_dataloader, model)
        ck = torch.load(os.path.join(args.log_dir, "best.ckpt"), map_location="cpu", weights_only=False)   # device 가 정수라 map_location 에 못 준다
        model.load_state_dict(ck["state_dict"])
        best_mae, best_pred = predict_all(args, test_dataloader, model)
        with open(os.path.join(args.log_dir, "test_pred.csv"), "w") as f:     # 사진별 예측 — 양눈 평균 MAE 용
            f.write("file_path,y,pred_best,pred_last\n")
            for (pth, anno), pb, pl in zip(test_folds[args.fold].data, best_pred, last_pred):
                f.write(f"{pth},{float(anno['Hb'])},{pb:.4f},{pl:.4f}\n")
        extra = (f"Test MAE at best-val epoch: {best_mae:.4f}\nTest MAE at last epoch: {last_mae:.4f}\n"
                 f"Best epoch: {int(ck['epoch']) + 1}\n")
        print(extra)

    # 전체 학습 시간 측정 종료
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # 시간 포맷팅 (시:분:초)
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s ({total_training_time:.2f} seconds)")
    print(f"Best validation MAE: {best_test_mae:.4f}")
    print("="*60)
    
    # 학습 시간을 파일로 저장
    with open(os.path.join(args.log_dir, "training_time.txt"), "w") as f:
        f.write(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s\n")
        f.write(f"Total training time (seconds): {total_training_time:.2f}\n")
        f.write(f"Best validation MAE: {best_test_mae:.4f}\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Average time per epoch: {total_training_time / args.epochs:.2f} seconds\n")
        f.write(extra)


def train_one_epoch(args, dataloader, model, optimizer, scaler, loss_fn, scheduler, epoch, global_step):
    model.train()
    pbar = tqdm(dataloader)
    log_losses = []
    log_maes = []
    seen = 0
    for batch in pbar:
        align = None
        if len(batch) == 4:                       # ★ 09-24 음성 대조 B — (img, Hb, anno, 정렬용 Hb)
            img, target, anno, align = batch
            align = align.to(args.device)
        else:
            img, target, anno = batch
        model.zero_grad()

        img, target, anno = (
            img.to(args.device),
            target.to(args.device),
            anno.to(args.device),
        )
        seen += len(target)
        label = encode_ordinal(args, target if align is None else align).long()
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # mixed precision 학습을 위해 autocast 사용
            model_out = model(img, anno, label)
            loss = loss_fn(model_out, target, epoch >= args.start_ib_epoch, align=align)

        scaler.scale(loss).backward()  # mixed precision 관련 코드
        scaler.unscale_(optimizer)  # mixed precision 관련 코드
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # 안정화를 위한 기울기 제한
        scaler.step(optimizer)  # mixed precision 관련 코드
        scaler.update()  # mixed precision 관련 코드
        if scheduler is not None:
            scheduler.step_update(global_step)

        global_step += 1

        mae_error = MAE_error(
            (decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]),
            target,
        )

        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())

        pbar.set_description(
            f"[{epoch}/{args.epochs}] lr:  {optimizer.param_groups[0]['lr']:.7f}  loss: {log_losses[-1]:.4f}  MAE: {mae_error.mean().item():.4f}({sum(log_maes) / seen:.4f})"
        )

    log_losses = sum(log_losses) / len(log_losses)
    log_maes = sum(log_maes) / seen
    return {"train/loss": log_losses, "train/mae": log_maes}, global_step


@torch.no_grad()
@torch.no_grad()
def predict_all(args, dataloader, model):
    """사진별 예측과 MAE. 순서는 dataloader(shuffle=False)의 dataset.data 순서 그대로다."""
    model.eval(); preds, err, seen = [], 0.0, 0
    for img, target, anno in dataloader:
        img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
        out = model(img, anno)
        p = decode_ordinal(args, out["logits"]) if args.use_ordinal_regression else out["logits"]
        err += MAE_error(p, target).sum().item(); seen += len(target)
        preds += p.float().reshape(len(target), -1)[:, 0].cpu().tolist()
    return err / seen, preds


def test_one_epoch(args, dataloader, model, loss_fn, epoch):
    model.eval()
    pbar = tqdm(dataloader)
    log_losses = []
    log_maes = []
    seen = 0
    for img, target, anno in pbar:
        img, target, anno = (
            img.to(args.device),
            target.to(args.device),
            anno.to(args.device),
        )
        seen += len(target)
        model_out = model(img, anno)

        loss = loss_fn(model_out, target)
        mae_error = MAE_error(
            (decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]),
            target,
        )

        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())

        pbar.set_description(
            f"[TEST@({epoch})Epoch] loss: {log_losses[-1]:.4f}  MAE: {mae_error.mean().item():.4f}({sum(log_maes) / seen:.4f})"
        )

    log_losses = sum(log_losses) / len(log_losses)
    log_maes = sum(log_maes) / seen
    return {"test/loss": log_losses, "test/mae": log_maes}


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument(
        "-c",
        "--config",
        default="configs/ajoumc_rxt50_image.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument("--fold", default=0, type=int, help="Fold number for cross-validation")
    parser.add_argument("--bins", type=int, help="Number of bins for ordinal regression")
    parser.add_argument("--no_enc", action="store_true", help="Disable encoder usage")
    parser.add_argument("--no_ord", action="store_true", help="Disable ordinal regression")
    parser.add_argument("--force-lr", dest="force_lr", type=float, default=None,
                        help="OR 유무에 따른 lr 하드코딩을 무시하고 이 값을 쓴다 (ablation 용)")
    parser.add_argument("--force-lambda", dest="force_lambda", type=str, default=None,
                        help="lambda_1,lambda_2,lambda_3 를 쉼표로 (ablation 용). 예: 0.000001,0.0001,1.5")
    parser.add_argument("--no-attn", dest="no_attn", action="store_true", default=None,
                        help="VE/OR 을 켠 채로 어텐션 풀링만 끈다 (경량화 ablation). "
                             "기본은 config 값(없으면 False) — 주지 않으면 동작이 그대로다.")
    parser.add_argument("--force-attn", dest="force_attn", action="store_true", default=None,
                        help="VE/OR 을 끈 채로도 어텐션 풀링을 켠다 (matched baseline 용). "
                             "기본은 config 값(없으면 False) — 주지 않으면 동작이 그대로다.")
    parser.add_argument("--arc-m", dest="arc_m", type=float, default=None,
                        help="OR 머리(ArcFace)의 각도여백 m. 기본은 config 값(없으면 0.5). "
                             "0 이면 여백 없이 코사인+스케일만 쓴다 (ablation 용).")
    parser.add_argument("--exp-name", type=str, help="A name for the experiment, used for log directory.")
    parser.add_argument("--log-dir", type=str, default="log_bins", help="Base log directory")
    parser.add_argument("--cv-fold", dest="cv_fold", type=int, default=None, help="겹 수 (기본 5)")
    parser.add_argument("--val-inner", dest="val_inner", type=int, default=0,
                        help="검증 분할: 학습 몫의 1/K 을 사람 단위로 뗀다. 0=끔(기본). 5겹이면 8, 10겹이면 9")

    args_config, remaining_args = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining_args)

    if args.bins is not None:
        args.ordinal_bins = args.bins

    if args.no_enc:
        args.use_encoder = False
    if args.no_ord:
        args.use_ordinal_regression = False

    if not args.use_ordinal_regression:
        args.lr = 0.0005
        args.lambda_1 = 0.000001
        args.lambda_2 = 0.01
        args.lambda_3 = 1

    # ★ 2026-08-22 추가 — ablation 용 탈출구. **기본 동작은 그대로다.**
    #   위 하드코딩 때문에 OR 을 끈 arm 만 lr 5e-4 가 되어, 머리 비교에 lr 교락이 생긴다.
    #   원본 src/train_ablation.py:365 는 이 블록을 통째로 주석 처리해 네 arm 을 2e-5 로 맞췄다.
    #   여기서는 인자를 줄 때만 덮어써서, 이미 끝난 학습의 재현성을 깨지 않는다.
    if getattr(args, "force_lr", None) is not None:
        args.lr = float(args.force_lr)
    if getattr(args, "force_lambda", None):
        l1, l2, l3 = (float(x) for x in str(args.force_lambda).split(","))
        args.lambda_1, args.lambda_2, args.lambda_3 = l1, l2, l3

    # ★ 2026-08-24 — 어텐션 풀링을 머리와 분리한다. 기본값 False 라 기존 실행은 그대로다.
    #   여기서 args 에 확정해 두면 config.yaml 에 함께 덤프되고, test_bins.py 가
    #   그 yaml 로 args 를 다시 만들기 때문에 평가 때도 같은 구조가 세워진다.
    #   config 에 force_attn 키가 있으면 parser.set_defaults 가 그 값을 기본으로 쓴다.
    #   CLI 로 --force-attn 을 주면 True 로 덮인다. 둘 다 없으면 None -> False.
    args.force_attn = bool(getattr(args, "force_attn", None) or False)
    args.no_attn    = bool(getattr(args, "no_attn", None) or False)

    # ★ 2026-08-25 — ArcFace 각도여백. 기본 0.5 라 기존 실행은 그대로다.
    #   args 에 확정해 두면 config.yaml 에 덤프되고, test_bins.py 가 그 yaml 로
    #   args 를 다시 만들기 때문에 평가 때도 같은 머리가 세워진다.
    _m = getattr(args, "arc_m", None)
    args.arc_m = 0.5 if _m is None else float(_m)

    # ★ 2026-10-28 — 분할 방식 (dataset/ajoumc_anemia.py 사람분할). config.yaml 에 남겨 어느 방식으로 나눴는지 보이게 한다.
    from dataset.ajoumc_anemia import OUTER_SEED
    args.split_mode = getattr(args, "split_mode", None) or "person_skf"
    args.outer_seed = OUTER_SEED

    if args.exp_name:
        log_dir_name = f"{args.exp_name}-fold{args.fold}"
    else:
        log_dir_name = f"image-bins{args.ordinal_bins}-fold{args.fold}"

    base_log_dir = args.log_dir if hasattr(args, 'log_dir') and args.log_dir else "log_bins"
    args.log_dir = os.path.join(base_log_dir, "train", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    main(args)
