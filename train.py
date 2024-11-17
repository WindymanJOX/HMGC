import os
import yaml
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from models.fcl.fcl import FCL
from models.unet2d import UNet
from models import nnunet3d
from utils.trainer import trainer

def train_HMGC(conf):
    if conf.net == 'unet2d':
        f_extrator = UNet(conf.in_dim, conf.n_cls)
    elif conf.net == 'unet3d':
        f_extrator = nnunet3d.get_net(conf.in_dim, conf.n_cls)
    
    fcl = FCL(conf.vd_p, conf.f_dim+conf.n_cls, conf.n_cls)
    
    # train_dir = conf.train_dir
    # val_dir = conf.val_dir

    X = torch.rand(size=(20, 4, 128, 128, 128))
    Y = torch.randint(0, 4, (20, 128, 128, 128))

    train_set = TensorDataset(X, Y)
    val_set = TensorDataset(X, Y)

    loader_args = dict(batch_size=conf.batch_size, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    module_params = list(f_extrator.parameters()) + list(fcl.parameters())
    if conf.optim == 'adam':
        optimizer = optim.Adam(module_params, lr=conf.lr, weight_decay=conf.weight_decay)
    elif conf.optim == 'sgd':
        optimizer = optim.SGD(module_params, lr=conf.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.epochs, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss() if conf.n_cls > 1 \
                else torch.nn.BCEWithLogitsLoss()
    
    trainer(f_extrator, fcl, optimizer, scheduler, 
            criterion, conf, train_loader, val_loader)

def main():
    para = argparse.ArgumentParser()
    conf = None
    # with open('config/unet2d.yml', 'r', encoding='utf-8') as f:
    #     conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open('config/unet3d.yml', 'r', encoding='utf-8') as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    para.set_defaults(**conf)
    conf = para.parse_args()
    
    train_HMGC(conf)

if __name__ == '__main__':
    main()