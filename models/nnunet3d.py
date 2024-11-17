import torch
from models import PlainConvUNet

def get_net(dim_in, n_cls, pth = None):
    norm_kwargs = dict(num_groups=32, eps=1e-05, affine=True)
    nonlin_kwargs = dict(inplace=True)

    model = PlainConvUNet(input_channels=dim_in, n_stages=6, features_per_stage=(64, 128, 256, 512, 512, 512),
                        conv_op=torch.nn.modules.conv.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2), n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                        num_classes=n_cls, n_conv_per_stage_decoder=(2, 2, 2, 2, 2), conv_bias=True, norm_op=torch.nn.GroupNorm, 
                        norm_op_kwargs=norm_kwargs, dropout_op=None, dropout_op_kwargs=None, nonlin=torch.nn.LeakyReLU, 
                        nonlin_kwargs=nonlin_kwargs, deep_supervision=True)
    if pth is not None:
        state_dict = torch.load(r"")
        new_state_dict = {}
        for k, value in state_dict['network_weights'].items():
            key = k
            if key not in model.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)

    return model

