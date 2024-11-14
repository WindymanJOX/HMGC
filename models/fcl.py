import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *

class Grouping_2(nn.Module):
    def __init__(self, in_dim, out_dim=133,dropout=0.0):
        super(Grouping_2, self).__init__()
        self.proj = nn.Linear(in_dim*2, out_dim)
        self.drop = nn.Dropout(p=dropout, inplace=True)

    def forward(self, M_new, M_pre, X):
        X = self.drop(X)
        X1 = torch.bmm(M_new, X)
        X2 = torch.bmm(M_pre,X)
        X = torch.cat([X1,X2],dim=-1)
        X = self.proj(X)

        return X

class M_NEW(nn.Module):
    def __init__(self,in_dim,hidden=96,ratio=[2,2,1,1]):
        super(M_NEW, self).__init__()
        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_dim,
                                              out_channels=hidden*ratio[0],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[0]),
                                    nn.LeakyReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=hidden*ratio[0],
                                              out_channels=hidden*ratio[1],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[1]),
                                    nn.LeakyReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[1],
                                              out_channels=hidden * ratio[2],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[2]),
                                    nn.LeakyReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[2],
                                              out_channels=hidden * ratio[3],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[3]),
                                    nn.LeakyReLU(inplace=True))
        self.conv_last = nn.Conv2d(in_channels=hidden * ratio[3],
                                              out_channels=1,
                                              kernel_size=1)

    def forward(self, X):
        # compute abs(x_i, x_j)
        x_i = X.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        M_new = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1)
        M_new = F.softmax(M_new, dim=-1)
        return M_new

class VD(nn.Module):
    def __init__(self, k, in_dim, num_classes):
        super(VD, self).__init__()
        # mask[1, n_sample]
        self.k = k
        self.num_classes = num_classes
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, M, X, mask):
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores/100)
        idx = []
        values = []
        for j in range(self.num_classes):
            idx_j = torch.where(mask[-1] == j)[0]
            way_scores = scores[idx_j]
            intra_scores = way_scores - way_scores.mean()
            _, way_idx = torch.topk(intra_scores,
                                    int(self.k * intra_scores.shape[0]), 
                                    largest=False)
            way_values = way_scores[way_idx]
            way_idx = idx_j[way_idx]
            idx.append(way_idx)
            values.append(way_values)
        values = torch.cat(values, dim=0)
        idx = torch.cat(idx, dim=0)

        X = X[0:, idx, :]
        values = torch.unsqueeze(values, -1)
        X = torch.mul(X, values)
        M = M[0:, idx, :]
        M = M[0:, :, idx]
        return M, X, mask[0:, idx]

class FCL(nn.Module):
    def __init__(self, vd_p, in_dim, num_classes):
        super(FCL, self).__init__()
        # 迭代次数
        l_n = len(vd_p)
        self.l_n = l_n
        start_m = M_NEW(in_dim=in_dim)
        start_grouping = Grouping_2(in_dim=in_dim,out_dim=in_dim)
        self.add_module('start_m', start_m)
        self.add_module('start_g', start_grouping)
        for l in range(l_n):

            down_m = M_NEW(in_dim=in_dim)
            down_g = Grouping_2(in_dim=in_dim,out_dim=in_dim)

            vd = VD(vd_p[l], in_dim=in_dim, num_classes=num_classes)
            
            self.add_module('down_m_{}'.format(l),down_m)
            self.add_module('down_g_{}'.format(l),down_g)
            self.add_module('vd_{}'.format(l),vd)

        bottom_m = M_NEW(in_dim=in_dim)
        bottom_g = Grouping_2(in_dim=in_dim,out_dim=in_dim)
        self.add_module('bottom_m', bottom_m)
        self.add_module('bottom_g', bottom_g)

    def forward(self, M_init, X, mask):
        M_pre = M_init
        M_new = self._modules['start_m'](X)
        X = self._modules['start_g'](M_new, M_pre, X)
        new_mask = mask

        for i in range(self.l_n):
            M_pre = M_new
            M_new = self._modules['down_m_{}'.format(i)](X)
            X = self._modules['down_g_{}'.format(i)](M_new, M_pre, X)
            M_new, X, new_mask = self._modules['vd_{}'.format(i)](M_new, X, new_mask)

        M_pre = M_new
        M_new = self._modules['bottom_m'](X)
        X = self._modules['bottom_g'](M_new, M_pre, X)
        return X, new_mask
    
def doFCL(fcl: FCL, mask, xf, device):
    xf, _mask = g_ancor(xf, mask)
    M_init = label2correlaiton(_mask).to(device)
    
    xf, _mask = fcl(M_init, xf, _mask)
    return xf, _mask