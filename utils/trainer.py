import datetime

import torch
from torch.utils.data.dataloader import DataLoader

from models.g_loss import GroupLoss
from models.fcl.fcl import doFCL

from utils.utils import *
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
            # img= batch['image'].to(device=device)
            # mask = batch['mask'].to(device=device, dtype=torch.long)
            img= batch[0].to(device=device)
            mask = batch[1].to(device=device, dtype=torch.long)
            logits = f_extrator(img)[0]
            xf = f_extrator.xrs[0][0:1]
            _mask = mask[0:1]
            xf = xf.flatten(-3).transpose(1, 2)
            _mask = _mask.flatten(-3)

            one_hot_label = one_hot_encode(conf.n_cls, _mask).to(device)
            xf = torch.cat([xf, one_hot_label], dim=-1)
            
            # if epoch >= conf.group_epoch:
            #     xf, _mask = doFCL(fcl, _mask, xf, device)
            
            xf, _mask = doFCL(fcl, _mask, xf, device)

            _mask = _mask.squeeze(0)
            group_loss = group(xf.squeeze(0), _mask, epoch)

            loss = criterion(logits, mask)
            loss += dice_loss(logits, mask)
            ce_dice_loss = loss.item()
            ce_dice_epoch_loss += ce_dice_loss
            group_epoch_loss += group_loss.item()

            loss += group_loss
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(f_extrator.parameters(), 1.0)
            optimizer.step()

        # update CGC_matrix every epoch
        group.update()

        ce_dice_epoch_loss /= len(train_loader)
        group_epoch_loss /= len(train_loader)

        logger.info(f'ce+dice loss: {ce_dice_epoch_loss}')
        logger.info(f'group loss: {group_epoch_loss}')

        scheduler.step()
        