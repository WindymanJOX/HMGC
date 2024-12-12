import torch
import torch.nn.functional as F

def dice_loss(pred, target, epsilon=1e-6):
    """
    计算 Dice Loss。

    参数：
    - pred: 预测值，形状为 [batch_size, num_classes, depth, height, width]。
    - target: 目标值（标签），形状为 [batch_size, num_classes, depth, height, width]。
    - epsilon: 防止分母为零的平滑值，默认为 1e-6。

    返回：
    - dice_loss: 平均 Dice Loss，标量。
    """
    # 将预测值经过 sigmoid 激活（如果尚未经过激活）
    pred = torch.sigmoid(pred)
    
    # 展平形状为 [batch_size, num_classes, -1]
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
    target = F.one_hot(target).permute(0, 4, 1, 2, 3).float()
    target_flat = target.view(target.shape[0], target.shape[1], -1)
    
    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
    
    # 计算 Dice 系数
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    
    # 取 Dice Loss
    dice_loss = 1 - dice_score.mean()
    
    return dice_loss