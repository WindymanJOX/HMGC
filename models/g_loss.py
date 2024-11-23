import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupLoss(nn.Module):
    def __init__(self, feat_dim, n_cls, start_epoch = 1, use_gpu=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = n_cls
        self.start_epoch = start_epoch
        self.ICG = True
        self.weight_cent = 1e-3
        self.IRCG = True
        self.distmethod = 'eu'
        # supervision = 2, no supervision=1
        self.tau = 1
        self.eta = 2
        self.use_gpu = use_gpu

        self.matrix = torch.randn((self.num_classes, feat_dim))
        self.grad = torch.zeros((self.num_classes, feat_dim))
        self.count = torch.zeros((self.num_classes, 1))

        if self.use_gpu:
            self.matrix = self.matrix.cuda()
            self.grad = self.grad.cuda()
            self.count = self.count.cuda()

        matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
        self.graph_weight_matrix = torch.mm(matrix_norm, matrix_norm.transpose(1, 0))

    def forward(self, xf, target, epoch):
        if self.training:
            with torch.no_grad():
                if xf.dim() == 5:
                    xf, target = self.mul_label_process_3D_batch_mean(xf, target)
                self.grad.index_add_(0, target, xf.detach().to(torch.float32))
                self.count.index_add_(0, target, torch.ones_like(target.view(-1, 1), dtype=torch.float32))

        if epoch >= self.start_epoch:

            if self.ICG is True:
                centers = self.matrix[target]
                ICGL = torch.pow(xf - centers, 2).sum(-1).mean()
                ICGL *= self.weight_cent
            else:
                if self.use_gpu:
                    ICGL = torch.tensor(0).cuda()
                else:
                    ICGL = torch.tensor(0)

            if self.IRCG is True:

                xf_norm = xf / torch.norm(xf, keepdim=True, dim=-1)
                matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
                samples_similarity_matrix = torch.mm(xf_norm, matrix_norm.transpose(1, 0)) 

                similarity_matrix = self.graph_weight_matrix[target]

                A = torch.exp(similarity_matrix / self.tau)
                B = torch.exp(samples_similarity_matrix / self.tau)

                euclidean_dist = torch.pow(A - B, 2).sum(-1).mean()
                IRCGL = euclidean_dist * self.eta
            else:
                if self.use_gpu:
                    IRCGL =  torch.tensor(0).cuda()
                else:
                    IRCGL =  torch.tensor(0)
            return IRCGL + ICGL

        else:
            if self.use_gpu:
                return torch.tensor(0).cuda()
            else:
                return torch.tensor(0)
            
    def mul_label_process_3D(self, xf, target):
        """
        Args:
            xf[bs, d, D, H, W]
            target[bs, n_cls, D, H, W]
        """
        # recover region target to normal target
        _target = torch.where(target, 1, 0)
        # [bs, D, H, W]
        _target = torch.sum(_target, dim=1) - 1
        # [n_voxel, d]
        _xf = xf.flatten(-3).transpose(1, 2).contiguous().view(-1, self.feat_dim)
        # [n_voxel]
        _target = _target.flatten(-3).contiguous().view(-1) 
        # all non-zero voxel idx
        region_idx = torch.where(_target>0)[0]
        _xf = _xf[region_idx, :]
        _target = _target[region_idx]

        return _xf, _target
    
    def mul_label_process_3D_batch_mean(self, xf: torch.Tensor, target: torch.Tensor):
        """
        Args:
            xf[bs, d, D, H, W]
            target[bs, n_cls, D, H, W]
        """
        # recover region target to normal target
        _target = torch.where(target, 1, 0)
        _target = torch.sum(_target, dim=1) - 1 # [bs, D, H, W]
        
        _xf = xf.flatten(-3).transpose(1, 2) # [bs, n_voxel, d]
        _target = _target.flatten(-3) # [bs, n_voxel]

        batch_mean_xf = []
        batch_mean_target = []
        for b in range(_target.shape[0]):
            _xf_b = _xf[b]
            _target_b = _target[b]
            for c in range(self.num_classes):
                _xf_b_c = _xf_b[_target_b==c]
                _xf_b_c = _xf_b_c.mean(dim=0)
                batch_mean_xf.append(_xf_b_c)
                batch_mean_target.append(torch.tensor(c, dtype=torch.long))
        
        batch_mean_xf = torch.stack(batch_mean_xf, dim=0).cuda()
        batch_mean_xf.requires_grad_()
        batch_mean_target = torch.stack(batch_mean_target, dim=0).cuda()

        return batch_mean_xf, batch_mean_target
    
    def mul_label_process_3D_all(self, xf: torch.Tensor, target: torch.Tensor):
        """
        Args:
            xf[bs, d, D, H, W]
            target[bs, n_cls, D, H, W]
        """
        # recover region target to normal target
        _target = torch.where(target, 1, 0)
        _target = torch.sum(_target, dim=1) # [bs, D, H, W]
        
        _xf = xf.flatten(-3).transpose(1, 2) # [bs, n_voxel, d]
        _target = _target.flatten(-3) # [bs, n_voxel]

        if _target.shape[0] == 1:
            _xf = _xf.squeeze(0)
            _target = _target.squeeze(0)
        
        else:
            _xf = _xf.reshape(-1, _xf.shape[-1])
            _target = _target.reshape(-1)

        return _xf, _target

    def update(self):
        # reset matrix
        index = torch.where(self.count > 0)[0]
        self.matrix[index] = (self.grad[index] / self.count[index]).detach()

        matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
        self.graph_weight_matrix = torch.mm(matrix_norm, matrix_norm.transpose(1, 0))

        # reset and update
        nn.init.constant_(self.grad, 0.)
        nn.init.constant_(self.count, 0.)