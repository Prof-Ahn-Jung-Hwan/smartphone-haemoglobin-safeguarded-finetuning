import torch
import torch.nn.functional as F
from torch import linalg as LA

from ordinal_regression import OrdinalReweightingLoss, encode_ordinal


class AnemiaLoss:
    def __init__(self, args, cls_num_list):
        self.args = args

        if args.use_encoder:
            self.mmd_loss = MMDLoss(args)

        if args.use_ordinal_regression:
            self.ordinal_reweighting_loss = OrdinalReweightingLoss(args, cls_num_list)

    def __call__(self, model_out, gt, ib_mode=False, align=None):
        # ★ 09-24 : align 이 오면 **정렬(VE)용 구간 라벨만** 그것으로 만든다 (음성 대조 B). 회귀 정답은 gt 그대로
        encoded_gt = encode_ordinal(self.args, gt if align is None else align)
        loss = 0

        z = model_out["z"]
        logits = model_out["logits"]
        z_prior = model_out["z_prior"]

        if self.args.use_encoder:
            mmd_loss, l2_z_mean, z_mean = self.mmd_loss(z, z_prior, encoded_gt)
            loss += self.args.lambda_1 * l2_z_mean + self.args.lambda_2 * mmd_loss

        if self.args.use_ordinal_regression:
            ord_loss = self.ordinal_reweighting_loss(logits, encoded_gt, z, ib_mode=ib_mode)
            loss += self.args.lambda_3 * ord_loss
        else:
            reg_loss = F.mse_loss(logits, gt.float())
            loss += self.args.lambda_3 * reg_loss

        return loss


class MMDLoss:
    def __init__(self, args):
        self.args = args
        self.num_cls = args.ordinal_bins - 1

    def __call__(self, z, z_prior, encoded_gt):
        encoded_gt = encoded_gt.squeeze(-1)
        y_valid = [i_cls in encoded_gt for i_cls in range(self.num_cls)]
        z_mean = torch.stack([z[encoded_gt == i_cls].mean(dim=0) for i_cls in range(self.num_cls)], dim=0)
        l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
        mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
        return mmd_loss, l2_z_mean, z_mean[y_valid]
