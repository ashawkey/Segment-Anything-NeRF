import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class SkipConnMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, skip_layers=[], bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.skip_layers = skip_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                fin = self.dim_in
            elif l in self.skip_layers:
                fin = self.dim_hidden + self.dim_in
            else:
                fin = self.dim_hidden
            
            if l == num_layers - 1:
                fout = self.dim_out
            else:
                fout = self.dim_hidden
            
            net.append(nn.Linear(fin, fout, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        x_in = x
        for l in range(self.num_layers):
            if l in self.skip_layers:
                x = torch.cat([x, x_in], dim=-1)
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.leaky_relu(x, inplace=True)
        return x

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 ):

        super().__init__(opt)

        self.geom_feat_dim = 15

        # grid
        self.grid, self.grid_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048 * self.bound)
        self.grid_mlp = MLP(self.grid_in_dim, 1 + self.geom_feat_dim, 64, 3, bias=False)

        # view-dependency
        self.view_encoder, self.view_in_dim = get_encoder('sh', input_dim=3, degree=4)
        self.view_mlp = MLP(self.geom_feat_dim + self.view_in_dim, 3, 32, 3, bias=False)

        # feature MLP
        if self.opt.with_sam:
            self.s_grid, self.s_dim = get_encoder("hashgrid", input_dim=3, num_levels=16, level_dim=8, base_resolution=16, log2_hashmap_size=19, desired_resolution=512)
            self.samvit_mlp = nn.Sequential(
                SkipConnMLP(self.s_dim + self.geom_feat_dim + self.view_in_dim + 4, 256, 256, 5, skip_layers=[2], bias=True),
                nn.LayerNorm(256),
            )

        # proposal network
        self.prop_encoders = nn.ModuleList()
        self.prop_mlp = nn.ModuleList()

        # hard coded 2-layer prop network
        prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=128)
        prop0_mlp = MLP(prop0_in_dim, 1, 16, 2, bias=False)
        self.prop_encoders.append(prop0_encoder)
        self.prop_mlp.append(prop0_mlp)

        prop1_encoder, prop1_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=256)
        prop1_mlp = MLP(prop1_in_dim, 1, 16, 2, bias=False)
        self.prop_encoders.append(prop1_encoder)
        self.prop_mlp.append(prop1_mlp)


    def common_forward(self, x):

        f = self.grid(x, bound=self.bound)
        f = self.grid_mlp(f)

        sigma = trunc_exp(f[..., 0])
        feat = f[..., 1:]
    
        return sigma, feat

    def forward(self, x, d, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        
        sigma, feat = self.common_forward(x)

        d = self.view_encoder(d)
        
        f_color = torch.cat([feat, d], dim=-1)

        return {
            'sigma': sigma,
            'color': f_color,
        }


    def density(self, x, proposal=-1):

        # proposal network
        if proposal >= 0 and proposal < len(self.prop_encoders):
            sigma = trunc_exp(self.prop_mlp[proposal](self.prop_encoders[proposal](x, bound=self.bound)).squeeze(-1))
        # final NeRF
        else:
            sigma, _ = self.common_forward(x)

        return {
            'sigma': sigma,
        }
    
    def apply_total_variation(self, w):
        if self.opt.with_sam:
            self.s_grid.grad_total_variation(w)
        else:
            self.grid.grad_total_variation(w)

    def apply_weight_decay(self, w):
        if self.opt.with_sam:
            self.s_grid.grad_weight_decay(w)
        else:
            self.grid.grad_weight_decay(w)

    # optimizer utils
    def get_params(self, lr):

        params = []

        params.extend([
            {'params': self.grid.parameters(), 'lr': lr},
            {'params': self.grid_mlp.parameters(), 'lr': lr}, 
            {'params': self.view_mlp.parameters(), 'lr': lr}, 
            {'params': self.prop_encoders.parameters(), 'lr': lr},
            {'params': self.prop_mlp.parameters(), 'lr': lr},
        ])

        if self.opt.with_sam:
            params.extend([
                {'params': self.s_grid.parameters(), 'lr': lr},
                {'params': self.samvit_mlp.parameters(), 'lr': lr},
            ])

        return params