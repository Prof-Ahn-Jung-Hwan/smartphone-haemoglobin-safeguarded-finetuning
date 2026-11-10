from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OrdinalReweightingLoss:
    def __init__(self, args, cls_num_list):
        self.args = args

        self.ordinal_regression_loss = OrdinalRegressionLoss(args, cls_num_list)

        ## 여러 imbalance loss 시도함. 기존 ib_focal loss가 가장 좋았음
        if args.imb_loss == "ib_focal_loss":
            self.ordinal_imb_loss = OrdinalIBFocalLoss(
                args, cls_num_list=cls_num_list, alpha=args.alpha, gamma=args.gamma
            )
        elif args.imb_loss == "ib_loss":
            self.ordinal_imb_loss = OrdinalIBLoss(args, cls_num_list=cls_num_list, alpha=args.alpha)
        elif args.imb_loss == "lda_loss":
            self.ordinal_imb_loss = OrdinalLDAMLoss(args, cls_num_list=cls_num_list)
        else:
            raise ValueError(f"Invalid imbalance loss: {args.imb_loss}")

    def __call__(self, ord_p, encoded_gt, z, ib_mode=False):
        if ib_mode:
            return self.ordinal_imb_loss(ord_p, encoded_gt, z)
        else:
            return self.ordinal_regression_loss(ord_p, encoded_gt)


import numpy as np


class OrdinalRegressionLoss:
    def __init__(self, args, cls_num_list):
        self.args = args

        per_cls_weights = 1.0 / (np.array(cls_num_list) + 1e-8)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.device)

        self.weight = per_cls_weights

        # self.bce_loss = nn.BCELoss(weight=None)
        self.bce_loss = nn.CrossEntropyLoss()

    def __call__(self, ord_p, encoded_gt):
        # ord_log_nk, ord_log_pk = torch.chunk(ord_p.softmax(-1), 2, -1)
        # ord_log_pk = ord_log_pk.squeeze(-1)

        valid_mask = torch.arange(ord_p.size(1), device=encoded_gt.device).unsqueeze(0) <= (encoded_gt)

        # ce = F.binary_cross_entropy(ord_log_pk, valid_mask.float(), weight=self.weight)
        # return ce
        ord_p = rearrange(ord_p, "b k p -> (b k) p")
        valid_mask = rearrange(valid_mask, "b k -> (b k)")

        return self.bce_loss(ord_p, valid_mask.long())


class OrdinalIBFocalLoss(nn.Module):
    def __init__(self, args, cls_num_list, alpha=10000.0, gamma=0.0):
        super().__init__()

        assert alpha > 0

        self.alpha = alpha
        self.epsilon = 0.001
        per_cls_weights = 1.0 / (np.array(cls_num_list) + 1e-8)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.device)
        self.weight = per_cls_weights
        self.gamma = gamma

    def ib_focal_loss(self, input_values, ib, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values * ib
        return loss.mean()

    def forward(self, ord_p, encoded_gt, features):
        features = torch.sum(torch.abs(features), 1).reshape(-1, 1)

        ord_log_nk, ord_log_pk = torch.chunk(ord_p.softmax(-1), 2, -1)
        ord_log_pk = ord_log_pk.squeeze(-1)

        valid_mask = torch.arange(ord_log_pk.size(1), device=encoded_gt.device).unsqueeze(0) <= (encoded_gt)

        grads = torch.sum(torch.abs(ord_log_pk - valid_mask.float()), 1)  # N * 1
        ib = grads * (features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)

        bce = stable_binary_cross_entropy(ord_p, valid_mask.float(), reduction="none", weight=self.weight)

        return self.ib_focal_loss(
            # F.binary_cross_entropy(ord_log_pk, valid_mask.float(), reduction="none", weight=self.weight),
            bce,
            ib.reshape(-1, 1),
            self.gamma,
        )


def stable_binary_cross_entropy(
    ord_p_logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:

    log_probs = F.log_softmax(ord_p_logits, dim=-1)

    log_prob_pos = log_probs[..., 1]
    log_prob_neg = log_probs[..., 0]

    bce_loss_elements = -(target * log_prob_pos + (1.0 - target) * log_prob_neg)

    if weight is not None:
        bce_loss_elements = bce_loss_elements * weight

    if reduction == "none":
        final_loss = bce_loss_elements
    elif reduction == "mean":
        final_loss = bce_loss_elements.mean()
    elif reduction == "sum":
        final_loss = bce_loss_elements.sum()
    else:
        raise ValueError(f"Reduction must be 'none', 'mean', or 'sum', but got {reduction}")

    return final_loss


class OrdinalIBLoss(nn.Module):
    def __init__(self, args, cls_num_list, alpha=1000.0):
        super().__init__()

        assert alpha > 0

        self.alpha = alpha
        self.epsilon = 0.001
        per_cls_weights = 1.0 / (np.array(cls_num_list) + 1e-8)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.device)
        self.weight = per_cls_weights

    def ib_loss(self, input_values, ib):
        loss = input_values * ib
        return loss.mean()

    def forward(self, ord_p, encoded_gt, features):
        features = torch.sum(torch.abs(features), 1).reshape(-1, 1)

        ord_log_nk, ord_log_pk = torch.chunk(ord_p.softmax(-1), 2, -1)
        ord_log_pk = ord_log_pk.squeeze(-1)

        valid_mask = (torch.arange(ord_log_pk.size(1), device=encoded_gt.device).unsqueeze(0) <= (encoded_gt)).float()

        ce = F.binary_cross_entropy(ord_log_pk, valid_mask.float(), reduction="none", weight=self.weight)

        valid_mask = torch.stack([1 - valid_mask, valid_mask], -1)

        grads = torch.sum(torch.abs(F.softmax(ord_p, -1)) - valid_mask, -1)

        ib = grads * features
        ib = self.alpha / (ib + self.epsilon)

        # grads = torch.sum(torch.abs(ord_log_pk - valid_mask.float()), 1)  # N * 1
        # ib = grads * (features.reshape(-1))
        # ib = self.alpha / (ib + self.epsilon)

        return self.ib_loss(ce, ib)


class OrdinalLDAMLoss(nn.Module):
    def __init__(self, args, cls_num_list, max_m=0.5, s=30):
        super().__init__()

        m_list = 1.0 / (
            np.sqrt(np.sqrt(np.stack([cls_num_list[0] - np.array(cls_num_list), cls_num_list], -1))) + 1e-8
        )
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(args.device)
        self.m_list = m_list
        assert s > 0
        self.s = s

        per_cls_weights = 1.0 / (
            np.array(np.stack([cls_num_list[0] - np.array(cls_num_list), cls_num_list], -1)) + 1e-8
        )
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.device)
        self.weight = per_cls_weights

    def ib_loss(self, input_values, ib):
        loss = input_values * ib
        return loss.mean()

    def forward(self, ord_p, encoded_gt, features):

        num_classes = ord_p.size(1)
        ord_log_nk, ord_log_pk = torch.chunk(ord_p.softmax(-1), 2, -1)
        ord_log_pk = ord_log_pk.squeeze(-1)

        valid_mask = (torch.arange(ord_log_pk.size(1), device=encoded_gt.device).unsqueeze(0) <= (encoded_gt)).int()
        total_loss = 0
        for i in range(num_classes):
            x = ord_p[:, i]
            target = valid_mask[:, i]
            m_list = self.m_list[i]

            index = torch.zeros_like(x, dtype=torch.uint8)  # BxC
            index.scatter_(1, target.int().view(-1, 1), 1)  # B -> BxC

            index_float = index.type(torch.cuda.FloatTensor)
            batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))  # 1xC @  CxB -> 1xB
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m  # BxC - 1xB
            output = torch.where(index, x_m, x)

            loss = F.cross_entropy(self.s * output, target.long())
            total_loss += loss

        return total_loss / num_classes
