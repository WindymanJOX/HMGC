import os
import glob
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import SimpleITK as sitk

from models import nnunet3d

def none_zero_crop_3D(input: np.ndarray, crop_box=None):
    """
    如果crop_box不为None，按照crop_box裁剪返回，否则找到crop_box返回crop_box
    Args:
        input: [[d,] D, H, W]
        crop_box: None or list[min_D, max_D, min_H, max_H, min_W, max_W]
    """
    if crop_box is None:
        if len(input.shape) == 4:
            input = input[0, ...]
        crop_idxs = np.where(input>0)
        min_D, max_D = crop_idxs[0].min(), crop_idxs[0].max()
        min_H, max_H = crop_idxs[1].min(), crop_idxs[1].max()
        min_W, max_W = crop_idxs[2].min(), crop_idxs[2].max()
        crop_box = [min_D, max_D, min_H, max_H, min_W, max_W]
        return crop_box
    
    else:
        input = input[..., crop_box[0]:crop_box[1],
                      crop_box[2]:crop_box[3],
                      crop_box[4]:crop_box[5]]
        
        return input

def predict_BraTS(model: nn.Module, dir: str, save_dir: str)->None:
    """
    Args:
        model: 
        dir: 包含4个模态nii.gz文件的文件夹
        save_dir: 
    """
    mod_paths = {}

    for mod_path in glob.glob(os.path.join(dir, '*.nii.gz')):
        if 't1.' in mod_path:
            mod_paths['t1'] = mod_path
        elif 't1ce.' in mod_path:
            mod_paths['t1ce'] = mod_path
        elif 't2.' in mod_path:
            mod_paths['t2'] = mod_path
        elif 'flair.' in mod_path:
            mod_paths['flair'] = mod_path
        elif 'seg.' in mod_path:
            seg_path = mod_path

    t1_npy = sitk.GetArrayFromImage(sitk.ReadImage(mod_paths['t1']))
    t1ce_npy = sitk.GetArrayFromImage(sitk.ReadImage(mod_paths['t1ce']))
    t2_npy = sitk.GetArrayFromImage(sitk.ReadImage(mod_paths['t2']))
    flair_npy = sitk.GetArrayFromImage(sitk.ReadImage(mod_paths['flair']))
    seg_npy = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    mod_list = [t1_npy, t1ce_npy, t2_npy, flair_npy]
    # [4, 155, 240, 240]
    mod_npy = np.stack(mod_list, axis=0)

    # crop_box = none_zero_crop_3D(mod_npy)
    crop_box = [13, 141, 56, 184, 56, 184]
    # [4, 128, 128, 128]
    mod_npy = none_zero_crop_3D(mod_npy, crop_box)
    seg_npy = none_zero_crop_3D(seg_npy, crop_box)
    # mod_npy = mod_npy[:, crop_box[0]:crop_box[1], ...]

    mod_ts = torch.tensor(mod_npy, dtype=torch.float32).cuda().unsqueeze(0)
    mod_ts = (mod_ts - mod_ts.min()) / (mod_ts.max() - mod_ts.min())
    seg_ts = torch.tensor(seg_npy, dtype=torch.float32)

    # pad = torch.nn.ZeroPad3d(padding = (8, 8, 8, 8, 0, 0)).cuda()
    # mod_ts = pad(mod_ts)

    model = model.cuda()

    model.eval()
    with autocast():
        # [3, D, H, W]
        logits = model(mod_ts)[0].squeeze(0)
        # [DxHxW, 32]
        xrs = model.xrs[0].squeeze(0).flatten(1).T

        prob = torch.sigmoid(logits)
        pred = torch.where(prob>0.5, 1, 0)

    # [D, H, W]
    pred = torch.sum(pred, dim=0).detach().cpu().numpy().astype(np.uint8)
    _pred = np.zeros_like(pred)

    # 恢复原本的标签
    _pred[pred == 1] = 2
    _pred[pred == 2] = 1
    _pred[pred == 3] = 4

    _pred_nii = sitk.GetImageFromArray(_pred)

    data_name = mod_paths['t1'].split(os.sep)[-1].split('_')[:-1]
    data_name = '_'.join(data_name)+'.nii.gz'
    save_path = os.path.join(save_dir, data_name)
    sitk.WriteImage(_pred_nii, save_path)
    # 保存特征向量用于降维可视化
    xrs = xrs.detach().cpu().numpy()
    seg_npy = seg_ts.flatten(0).unsqueeze(1).numpy()
    vectors = np.concatenate([xrs, seg_npy], axis=1)
    np.save('vectors.npy', vectors)

# _dir = r'G:\datasets\BraTS2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001'
# save_dir = r'F:\code\HMGC'
# model = nnunet3d.get_net().cuda()
# predict_BraTS(model, _dir, save_dir)

seg_path = r"G:\datasets\BraTS2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii.gz"
seg_npy = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
crop_box = [13, 141, 56, 184, 56, 184]
seg_npy = none_zero_crop_3D(seg_npy, crop_box)
np.save(r'BraTS20_Training_001_seg.npy', seg_npy)
