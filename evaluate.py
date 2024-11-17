import torch
import torch.nn.functional as F
import numpy
from tqdm import tqdm

from utils.dice_score import dice_loss
@torch.inference_mode()
def evaluate(net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    epoch_loss = 0
    
    for batch in dataloader:
        image, mask_true = batch[0], batch[1]

        # move images and labels to correct device and type
        image = image.to(device=device)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        # predict the mask
        mask_pred = net(image)
        if net.n_classes == 1:
            loss = criterion(mask_pred.squeeze(1), mask_true.float())
            loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
        else:
            loss = criterion(mask_pred, mask_true)
            loss += dice_loss(
                F.softmax(mask_pred, dim=1).float(),
                F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
        
        epoch_loss += loss.item()
            
        if net.n_classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float().squeeze(1)
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
    avg_loss = epoch_loss/len(dataloader)
    return dice_score / max(num_val_batches, 1), avg_loss