import datetime

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from models.g_loss import GroupLoss
from models.fcl import doFCL

from utils.utils import *
from evaluate import evaluate
from utils.logger import create_logger
from utils.dice_score import dice_loss

def trainer(f_extrator, fcl, optimizer, scheduler, criterion, conf,
            train_loader: DataLoader, val_loader: DataLoader):

    device = torch.device(conf.device)
    group = GroupLoss(conf.f_dim+conf.n_cls, conf.n_cls, conf.group_epoch, use_gpu=True)
    fcl = fcl.to(device)
    f_extrator.to(device)

    log_name = conf.dataset + '_' + conf.net
    log_name += '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = create_logger(f'log/{log_name}')

    logger.info(conf)

    for epoch in range(conf.epochs):
        f_extrator.train(), fcl.train()
        logger.info(f'epoch: {epoch}')
        ce_dice_epoch_loss = 0
        group_epoch_loss = 0
        for batch in train_loader:
            img= batch['image'].to(device=device)
            mask = batch['mask'].to(device=device, dtype=torch.long)
            logits = f_extrator(img)
            xf = f_extrator.xfs
            xf = xf[0:1, ...]
            _mask = mask[0:1, ...]
            xf = xf.flatten(-2).transpose(1, 2)
            _mask = _mask.flatten(-2)

            ori_xf_n_sample = xf.shape[1]
            mask_fcl = _mask
            one_hot_label = one_hot_encode(f_extrator.n_classes, mask_fcl).to(device)
            xf = torch.cat([xf, one_hot_label], dim=-1)
            
            if epoch >= conf.group_epoch:
                xf, mask_fcl = doFCL(fcl, mask_fcl, xf, device)

            droped_xf_n_sample = xf.shape[1]

            mask_group = mask_fcl.squeeze(0)
            group_loss = group(xf.squeeze(0), mask_group, epoch)

            loss = criterion(logits, mask)
            loss += dice_loss(
                F.softmax(logits, dim=1).float(),
                F.one_hot(mask, f_extrator.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            ce_dice_loss = loss.item()
            ce_dice_epoch_loss += ce_dice_loss
            group_epoch_loss += group_loss.item()

            loss += group_loss
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(f_extrator.parameters(), 1.0)
            optimizer.step()

        logger.info(f'ori xf n_sample: {ori_xf_n_sample}')
        logger.info(f'droped xf n_sample: {droped_xf_n_sample}')
        # update class_center_matrix every epoch
        group.update()

        ce_dice_epoch_loss /= len(train_loader)
        group_epoch_loss /= len(train_loader)

        logger.info(f'ce+dice loss: {ce_dice_epoch_loss}')
        logger.info(f'group loss: {group_epoch_loss}')

        # Evaluation round
        val_score, val_loss = evaluate(f_extrator, val_loader, device, criterion)
        scheduler.step()

        logger.info(f'val dice score: {val_score}')
        logger.info(f'val ce+dice loss: {val_loss}')
        