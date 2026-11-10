import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

MAE_error = lambda x, y: nn.L1Loss(reduction="none")(x.cpu(), y.cpu())


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU


def log_data_split(args, train_fold, test_fold):
    train_split_path = os.path.join(args.log_dir, f"train.txt")
    test_split_path = os.path.join(args.log_dir, f"test.txt")
    with open(train_split_path, "w") as f:
        for d in train_fold.data:
            file_name = d[0]
            Hb = d[1]["Hb"]
            f.write(f"{file_name}\t{Hb}\n")
        f.close()
    with open(test_split_path, "w") as f:
        for d in test_fold.data:
            file_name = d[0]
            Hb = d[1]["Hb"]
            f.write(f"{file_name}\t{Hb}\n")
        f.close()


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     epoch = epoch + 1
#     if epoch > args.start_ib_epoch:
#         lr = args.lr * 0.01
#         if epoch > args.epochs * 0.9:
#             lr = args.lr * 0.000001
#         elif epoch > args.epochs * 0.8:
#             lr = args.lr * 0.0001
#     else:
#         if epoch <= args.epochs * 0.025:
#             lr = args.lr * epoch / (args.epochs * 0.025)
#         elif epoch > args.epochs * 0.9:
#             lr = args.lr * 0.0001
#         elif epoch > args.epochs * 0.8:
#             lr = args.lr * 0.01
#         else:
#             lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
        
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    # ★ 2026-09-13 — 감쇠 시점을 전체 epoch 의 80 %·90 % 로 비례화한다.
    #   예전에는 160·180 에 박혀 있어 epoch 100 이면 감쇠가 한 번도 일어나지 않았다.
    #   epoch 200 에서는 0.8·0.9 × 200 = 160·180 이라 **기존 실행과 완전히 같다.**
    E = int(getattr(args, "epochs", 200))
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > int(0.9 * E):
        lr = args.lr * 0.0001
    elif epoch > int(0.8 * E):
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(),loc='lower right')


def plot_logs(
    train_ticks,
    test_ticks,
    train_loss,
    train_mae,
    test_loss,
    test_mae,
    fig_name="log.png",
    best_mae=None,
):
    plt.plot(train_ticks, train_loss, "r-", label="train/loss")
    plt.plot(train_ticks, train_mae, "b--", label="train/mae")
    plt.plot(test_ticks, test_loss, "g+", label="test/loss")
    plt.plot(test_ticks, test_mae, "mo", label="test/mae")

    legend_without_duplicate_labels(plt)

    if best_mae is not None:
        text_to_display = f"Best MAE: {best_mae:.4f}" 
        plt.text(
            0.98,  
            0.98,  
            text_to_display,
            transform=plt.gca().transAxes, 
            horizontalalignment="right",  
            verticalalignment="top",  
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),  
        )

    plt.savefig(fig_name)
    plt.close()
