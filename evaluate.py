import torch
import torch.nn.functional as F
import numpy
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
@torch.inference_mode()
def evaluate(net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    epoch_loss = 0
    
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        # predict the mask
        mask_pred, _= net(image)
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

def calEvaluationBinary(predict, target):
    if torch.is_tensor(predict):   
        predict = predict.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    total_iou = 0.0
    total_recall = 0.0
    total_ppv = 0.0
    total_acc = 0.0
    for i in range(len(predict)):
        pre_split = predict[i]
        tar_split = target[i]
        pre_split = numpy.atleast_1d(pre_split.astype(numpy.bool))
        tar_split = numpy.atleast_1d(tar_split.astype(numpy.bool))
        tp = numpy.count_nonzero(pre_split & tar_split)
        fn = numpy.count_nonzero(~pre_split & tar_split)
        fp = numpy.count_nonzero(~tar_split & pre_split)
        tn = numpy.count_nonzero(~tar_split & ~pre_split)
        try:
            recall = tp / float(tp + fn)
            iou = tp / float(tp + fn + fp)
            ppv = tp / float(tp + fp)
            acc = float(tp + tn) / float(tp + tn + fp + fn)
            # recall = metric.recall(pre_split, tar_split)
            # iou = metric.jc(pre_split, tar_split)
            # ppv = metric.precision(pre_split, tar_split)
            # acc = float(tp + tn) / float(tp + tn + fp + fn)
        except ZeroDivisionError:
            iou = 0.0
            recall = 0.0
            ppv = 0.0
            acc = 0.0
        total_iou += iou
        total_recall += recall
        total_ppv += ppv
        total_acc += acc

    return total_recall / len(predict), total_iou / len(predict), \
        total_ppv / len(predict), total_acc / len(predict)

def evaluation(net, dataloader, device, amp=True):
    net.eval()
    num_val_batches = len(dataloader)
    _recall_total = 0
    _iou_total = 0
    _ppv_total = 0
    _acc_total = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm.tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            if net.n_classes==1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float().squeeze(1)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                mask_pred = mask_pred.argmax(dim=1)

            _recall, _iou, _ppv, _acc = calEvaluationBinary(mask_pred, mask_true)
            _iou_total += _iou
            _recall_total += _recall
            _ppv_total += _ppv
            _acc_total += _acc

    net.train()
    return _recall_total / max(num_val_batches, 1), _iou_total / max(num_val_batches, 1), \
            _ppv_total / max(num_val_batches, 1), _acc_total / max(num_val_batches, 1)