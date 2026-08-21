import torch
import torch.nn as nn
import torch.nn.functional as F


class Orthogonal_Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Orthogonal_Loss, self).__init__()
        self.epsilon = epsilon
    
    def compute_orthogonal_loss(self, embeddings):
        B, L, C = embeddings.shape
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        cosine_sim_matrix = torch.einsum('blc,bkc->blk', embeddings_norm, embeddings_norm)
        cosine_sim_squared = cosine_sim_matrix ** 2
        eye_mask = torch.eye(L, device=embeddings.device).unsqueeze(0)
        cosine_sim_squared = cosine_sim_squared * (1 - eye_mask)
        cosine_loss = cosine_sim_squared.mean()
        return cosine_loss
    def forward(self, embeddings, args):
        Loss_normal_text = self.compute_orthogonal_loss(embeddings[:, 0:args.prompt_num,:])
        Loss_abnormal_text = self.compute_orthogonal_loss(embeddings[:,args.prompt_num:,:])
        orthogonal_loss = Loss_normal_text + Loss_abnormal_text

        # Inter-role margin: normal 和 abnormal 的 mean embedding 应该分离
        margin_target = float(getattr(args, "inter_role_margin", 0.3))
        margin_weight = float(getattr(args, "inter_role_margin_weight", 0.5))
        normal_mean = F.normalize(embeddings[:, :args.prompt_num, :].mean(dim=1), dim=-1)
        abnormal_mean = F.normalize(embeddings[:, args.prompt_num:, :].mean(dim=1), dim=-1)
        sim = (normal_mean * abnormal_mean).sum(dim=-1)  # [B]
        margin_loss = F.relu(sim - margin_target).mean()

        return orthogonal_loss + margin_weight * margin_loss

class FocalLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(FocalLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, gt, gamma=2.0, alpha=1, mask_ratio=1.0):
        gt_one_hot = F.one_hot(gt.long(), num_classes=2).permute(0, 3, 1, 2).float()  
        pt = (pred * gt_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** gamma  # (1 - p_t)^γ
        focal_loss = -alpha * focal_weight * torch.log(pt + self.epsilon)
        fg_mask = gt > 0
        bg_mask = ~fg_mask
        fg_loss = focal_loss * fg_mask.float()
        bg_loss = focal_loss * bg_mask.float()
        fg_pixels = fg_mask.sum().float()
        bg_pixels = bg_mask.sum().float()
        fg_loss_final = fg_loss.sum() / (fg_pixels + self.epsilon)
        bg_loss_final = bg_loss.sum() / (bg_pixels + self.epsilon)
        loss = fg_loss_final  + bg_loss_final
        return loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def compute_loss(self, pred, target):
        target_sum = torch.sum(target)
        intersection = torch.sum(pred * target) 
        union = torch.sum(pred) + target_sum 
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        loss = 1 - dice
        return loss 

    def forward(self, pred, target):
        target = target.float()
        # 只计算前景通道 Dice（异常区域）。
        # 异常检测中背景像素远多于前景，background Dice 对梯度贡献极小且噪声大。
        loss = self.compute_loss(pred[:, 1, :, :], target.clone())
        return loss
