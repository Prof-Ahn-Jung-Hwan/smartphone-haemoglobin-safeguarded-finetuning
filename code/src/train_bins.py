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
    VAL = getattr(args, "_val_folds", None) is not None          # inner validation set in use?

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
        print(f"split: train {len(train_folds[args.fold])} · validation {len(val_ds)} · test {len(test_folds[args.fold])}")

    print(f"Class number list: {cls_num_list}")

    model = HbNet(args).to(args.device)
    optimizer, scheduler = create_optimizer(args, model, len(train_dataloader))
    scaler = torch.amp.GradScaler()  # mixed precision
    loss_fn = AnemiaLoss(args, cls_num_list)

    xs, ys1, ys2 = [], [], []
    txs, tys1, tys2 = [], [], []
    best_test_mae = float("inf")
    global_step = 0
    
    # total training time
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_result, global_step = train_one_epoch(
            args, train_dataloader, model, optimizer, scaler, loss_fn, scheduler, epoch, global_step
        )

        xs.append(epoch)
        ys1.append(train_result["train/loss"])
        ys2.append(train_result["train/mae"])
        
        
        if (epoch + 1) % 5 == 0:
            # with an inner validation set, best.ckpt is chosen on it; the test fold is not used here
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
    
    # with an inner validation set, the test fold is scored once after training (best-validation and last epoch)
    extra = ""
    # if the validation MAE was never finite (divergence at a large learning rate), no best.ckpt exists:
    # copy last.ckpt and mark the fold as diverged, so that the learning rate is not selected.
    if not os.path.exists(os.path.join(args.log_dir, "best.ckpt")) and os.path.exists(os.path.join(args.log_dir, "last.ckpt")):
        import shutil
        shutil.copy(os.path.join(args.log_dir, "last.ckpt"), os.path.join(args.log_dir, "best.ckpt"))
        with open(os.path.join(args.log_dir, "DIVERGED.txt"), "w") as f:
            f.write("Validation MAE was never finite (diverged); last.ckpt was copied to best.ckpt.\n")
        print("diverged — no best.ckpt; last.ckpt used as best")
    if VAL:
        last_mae, last_pred = predict_all(args, test_dataloader, model)
        ck = torch.load(os.path.join(args.log_dir, "best.ckpt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        best_mae, best_pred = predict_all(args, test_dataloader, model)
        with open(os.path.join(args.log_dir, "test_pred.csv"), "w") as f:     # per-photograph predictions (both-eye average later)
            f.write("file_path,y,pred_best,pred_last\n")
            for (pth, anno), pb, pl in zip(test_folds[args.fold].data, best_pred, last_pred):
                f.write(f"{pth},{float(anno['Hb'])},{pb:.4f},{pl:.4f}\n")
        extra = (f"Test MAE at best-val epoch: {best_mae:.4f}\nTest MAE at last epoch: {last_mae:.4f}\n"
                 f"Best epoch: {int(ck['epoch']) + 1}\n")
        print(extra)

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s ({total_training_time:.2f} seconds)")
    print(f"Best validation MAE: {best_test_mae:.4f}")
    print("="*60)
    
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
        img, target, anno = batch
        model.zero_grad()

        img, target, anno = (
            img.to(args.device),
            target.to(args.device),
            anno.to(args.device),
        )
        seen += len(target)
        label = encode_ordinal(args, target).long()
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # mixed precision
            model_out = model(img, anno, label)
            loss = loss_fn(model_out, target, epoch >= args.start_ib_epoch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping
        scaler.step(optimizer)
        scaler.update()
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
    """Per-photograph predictions and MAE, in the order of dataset.data (shuffle=False)."""
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
                        help="use this learning rate instead of the default set below")
    parser.add_argument("--force-lambda", dest="force_lambda", type=str, default=None,
                        help="lambda_1,lambda_2,lambda_3, comma separated (e.g. 0.000001,0.0001,1.5)")
    parser.add_argument("--no-attn", dest="no_attn", action="store_true", default=None,
                        help="disable attention pooling (default: config value, else False)")
    parser.add_argument("--force-attn", dest="force_attn", action="store_true", default=None,
                        help="enable attention pooling (default: config value, else False)")
    parser.add_argument("--arc-m", dest="arc_m", type=float, default=None,
                        help="ArcFace angular margin m of the ordinal head (default: config value, else 0.5)")
    parser.add_argument("--exp-name", type=str, help="A name for the experiment, used for log directory.")
    parser.add_argument("--log-dir", type=str, default="log_bins", help="Base log directory")
    parser.add_argument("--cv-fold", dest="cv_fold", type=int, default=None, help="number of cross-validation folds (default 5)")
    parser.add_argument("--val-inner", dest="val_inner", type=int, default=0,
                        help="inner validation set: 1/K of the training participants (0 = none; 9 for ten folds, 8 for five)")

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

    # explicit learning rate / loss weights override the defaults above when given
    if getattr(args, "force_lr", None) is not None:
        args.lr = float(args.force_lr)
    if getattr(args, "force_lambda", None):
        l1, l2, l3 = (float(x) for x in str(args.force_lambda).split(","))
        args.lambda_1, args.lambda_2, args.lambda_3 = l1, l2, l3

    # attention pooling flags are fixed here so that they are written to config.yaml and the same model is
    # rebuilt from that file at evaluation
    args.force_attn = bool(getattr(args, "force_attn", None) or False)
    args.no_attn    = bool(getattr(args, "no_attn", None) or False)

    # ArcFace margin, likewise written to config.yaml
    _m = getattr(args, "arc_m", None)
    args.arc_m = 0.5 if _m is None else float(_m)

    # splitting method (dataset/ajoumc_anemia.split_by_participant), recorded in config.yaml
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
