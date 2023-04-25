import os
import cv2
import math
import json
import tqdm
import mcubes
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from torch_efficient_distloss import eff_distloss

from .utils import custom_meshgrid
from meshutils import *

@torch.cuda.amp.autocast(enabled=False)
def distort_loss(bins, weights):
    # bins: [N, T+1], in [0, 1]
    # weights: [N, T]

    intervals = bins[..., 1:] - bins[..., :-1]
    mid_points = bins[..., :-1] + intervals / 2

    loss = eff_distloss(weights, mid_points, intervals)

    return loss


@torch.cuda.amp.autocast(enabled=False)
def proposal_loss(all_bins, all_weights):
    # all_bins: list of [N, T+1]
    # all_weights: list of [N, T]

    def loss_interlevel(t0, w0, t1, w1):
        # t0, t1: [N, T+1]
        # w0, w1: [N, T]
        cw1 = torch.cat([torch.zeros_like(w1[..., :1]), torch.cumsum(w1, dim=-1)], dim=-1)
        inds_lo = (torch.searchsorted(t1[..., :-1].contiguous(), t0[..., :-1].contiguous(), right=True) - 1).clamp(0, w1.shape[-1] - 1)
        inds_hi = torch.searchsorted(t1[..., 1:].contiguous(), t0[..., 1:].contiguous(), right=True).clamp(0, w1.shape[-1] - 1)

        cw1_lo = torch.take_along_dim(cw1[..., :-1], inds_lo, dim=-1)
        cw1_hi = torch.take_along_dim(cw1[..., 1:], inds_hi, dim=-1)
        w = cw1_hi - cw1_lo

        return (w0 - w).clamp(min=0) ** 2 / (w0 + 1e-8)

    bins_ref = all_bins[-1].detach()
    weights_ref = all_weights[-1].detach()
    loss = 0
    for bins, weights in zip(all_bins[:-1], all_weights[:-1]):
        loss += loss_interlevel(bins_ref, weights_ref, bins, weights).mean()

    return loss


# MeRF-like contraction
@torch.cuda.amp.autocast(enabled=False)
def contract(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True) # [N, 1], [N, 1]
    scale = 1 / mag.repeat(1, C)
    scale.scatter_(1, idx, (2 - 1 / mag) / mag)
    z = torch.where(mag < 1, x, x * scale)
    return z.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def uncontract(z):
    # z: [..., C]
    shape, C = z.shape[:-1], z.shape[-1]
    z = z.view(-1, C)
    mag, idx = z.abs().max(1, keepdim=True) # [N, 1], [N, 1]
    scale = 1 / (2 - mag.repeat(1, C)).clamp(min=1e-8)
    scale.scatter_(1, idx, 1 / (2 * mag - mag * mag).clamp(min=1e-8))
    x = torch.where(mag < 1, z, z * scale)
    return x.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def sample_pdf(bins, weights, T, perturb=False):
    # bins: [N, T0+1]
    # weights: [N, T0]
    # return: [N, T]
    
    N, T0 = weights.shape
    weights = weights + 0.01  # prevent NaNs
    weights_sum = torch.sum(weights, -1, keepdim=True) # [N, 1]
    pdf = weights / weights_sum
    cdf = torch.cumsum(pdf, -1).clamp(max=1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) # [N, T+1]
    
    u = torch.linspace(0.5 / T, 1 - 0.5 / T, steps=T).to(weights.device)
    u = u.expand(N, T)

    if perturb:
        u = u + (torch.rand_like(u) - 0.5) / T
        
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True) # [N, t]

    below = torch.clamp(inds - 1, 0, T0)
    above = torch.clamp(inds, 0, T0)

    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above)

    bins_t = torch.clamp(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0)), 0, 1) # [N, t]
    bins = bins_g0 + bins_t * (bins_g1 - bins_g0) # [N, t]

    return bins


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.05):
    # rays: [N, 3], [N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [N, 1], far [N, 1]

    tmin = (aabb[:3] - rays_o) / (rays_d + 1e-15) # [N, 3]
    tmax = (aabb[3:] - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).amax(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).amin(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=min_near)

    return near, far


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # bound for ray marching (world space)
        self.real_bound = opt.bound

        # bound for grid querying
        if self.opt.contract:
            self.bound = 2
        else:
            self.bound = opt.bound
        
        self.cascade = 1 + math.ceil(math.log2(self.bound))

        self.grid_size = opt.grid_size
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb_train = torch.FloatTensor([-self.real_bound, -self.real_bound, -self.real_bound, self.real_bound, self.real_bound, self.real_bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.cuda_ray = opt.cuda_ray
        
        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            self.register_buffer('density_grid', density_grid)
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
        else:
            # spacing functions
            self.spacing_fn = lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
            self.spacing_fn_inv = lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))
    
    def forward(self, x, d, **kwargs):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x, **kwargs):
        raise NotImplementedError()

    def update_aabb(self, aabb):
        # aabb: tensor of [6]
        if not torch.is_tensor(aabb):
            aabb = torch.from_numpy(aabb).float()
        self.aabb_train = aabb.clamp(-self.real_bound, self.real_bound).to(self.aabb_train.device)
        self.aabb_infer = self.aabb_train.clone()
        print(f'[INFO] update_aabb: {self.aabb_train.cpu().numpy().tolist()}')

    @torch.no_grad()
    def export_mesh(self, save_path, resolution=None, decimate_target=1e5, dataset=None, S=128):

        # only for the inner mesh inside [-1, 1]
        if resolution is None:
            resolution = self.grid_size

        device = self.aabb_train.device

        if self.cuda_ray:
            density_thresh = min(self.mean_density, self.density_thresh)
        else:
            density_thresh = self.density_thresh

        # sigmas = np.zeros([resolution] * 3, dtype=np.float32)
        sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        # query
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        val = self.density(pts.to(device))['sigma'].reshape(len(xs), len(ys), len(zs)) # [S, 1] --> [x, y, z]
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val

        # use the density_grid as a baseline mask (also excluding untrained regions)
        if self.cuda_ray:
            mask = torch.zeros([self.grid_size] * 3, dtype=torch.float32, device=device)
            all_indices = torch.arange(self.grid_size**3, device=device, dtype=torch.int)
            all_coords = raymarching.morton3D_invert(all_indices).long()
            mask[tuple(all_coords.T)] = self.density_grid[0]
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[resolution] * 3, mode='nearest').squeeze(0).squeeze(0)
            mask = (mask > density_thresh)
            sigmas = sigmas * mask

        sigmas = torch.nan_to_num(sigmas, 0)
        sigmas = sigmas.cpu().numpy()

        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)

        vertices = vertices / (resolution - 1.0) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)

        ### visibility test.
        if dataset is not None:
            visibility_mask = self.mark_unseen_triangles(vertices, triangles, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
            vertices, triangles = remove_masked_trigs(vertices, triangles, visibility_mask, dilation=self.opt.visibility_mask_dilation)

        ### reduce floaters by post-processing...
        vertices, triangles = clean_mesh(vertices, triangles, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=True, remesh=False)
        
        ### decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(os.path.join(save_path, f'mesh_0.ply'))

        # for the outer mesh [1, inf]
        if self.bound > 1:

            reso = self.grid_size
            target_reso = self.opt.env_reso
            decimate_target //= 2 # empirical...

            if self.cuda_ray:
                all_indices = torch.arange(reso**3, device=device, dtype=torch.int)
                all_coords = raymarching.morton3D_invert(all_indices).cpu().numpy()

            # for each cas >= 1
            for cas in range(1, self.cascade):
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / target_reso

                # remap from density_grid
                occ = torch.zeros([reso] * 3, dtype=torch.float32, device=device)
                if self.cuda_ray:
                    occ[tuple(all_coords.T)] = self.density_grid[cas]
                else:
                    # query
                    X = torch.linspace(-bound, bound, reso).split(S)
                    Y = torch.linspace(-bound, bound, reso).split(S)
                    Z = torch.linspace(-bound, bound, reso).split(S)

                    for xi, xs in enumerate(X):
                        for yi, ys in enumerate(Y):
                            for zi, zs in enumerate(Z):
                                xx, yy, zz = custom_meshgrid(xs, ys, zs)
                                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                                    val = self.density(pts.to(device))['sigma'].reshape(len(xs), len(ys), len(zs)) # [S, 1] --> [x, y, z]
                                occ[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val

                # interpolate the occ grid to desired resolution to control mesh size...
                occ = F.interpolate(occ.unsqueeze(0).unsqueeze(0), [target_reso] * 3, mode='trilinear').squeeze(0).squeeze(0)
                occ = torch.nan_to_num(occ, 0)
                occ = (occ > density_thresh).cpu().numpy()

                vertices_out, triangles_out = mcubes.marching_cubes(occ, 0.5)

                vertices_out = vertices_out / (target_reso - 1.0) * 2 - 1 # range in [-1, 1]

                # remove the center (already covered by previous cascades)
                _r = 0.45
                vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})')

                if vertices_out.shape[0] == 0: 
                    print(f'[WARN] empty mesh at cas {cas}, consider reducing --bound')
                    continue

                vertices_out = vertices_out * (bound - half_grid_size)

                # remove the out-of-AABB region
                xmn, ymn, zmn, xmx, ymx, zmx = self.aabb_train.cpu().numpy().tolist()
                xmn += half_grid_size
                ymn += half_grid_size
                zmn += half_grid_size
                xmx -= half_grid_size
                ymx -= half_grid_size
                zmx -= half_grid_size
                vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})')

                # clean mesh
                vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)

                if vertices_out.shape[0] == 0: 
                    print(f'[WARN] empty mesh at cas {cas}, consider reducing --bound')
                    continue

                # decimate
                if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
                    vertices_out, triangles_out = decimate_mesh(vertices_out, triangles_out, decimate_target, optimalplacement=False)

                vertices_out = vertices_out.astype(np.float32)
                triangles_out = triangles_out.astype(np.int32)

                print(f'[INFO] exporting outer mesh at cas {cas}, v = {vertices_out.shape}, f = {triangles_out.shape}')
        
                if dataset is not None:
                    visibility_mask = self.mark_unseen_triangles(vertices_out, triangles_out, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
                    vertices_out, triangles_out = remove_masked_trigs(vertices_out, triangles_out, visibility_mask, dilation=self.opt.visibility_mask_dilation)
                
                if self.opt.contract:
                    vertices_out = uncontract(torch.from_numpy(vertices_out)).cpu().numpy()
                
                mesh_out = trimesh.Trimesh(vertices_out, triangles_out, process=False) # important, process=True leads to seg fault...
                mesh_out.export(os.path.join(save_path, f'mesh_{cas}.ply'))

    def render(self, rays_o, rays_d, **kwargs):
        
        if self.cuda_ray:
            return self.run_cuda(rays_o, rays_d, **kwargs)
        elif self.training:
            return self.run(rays_o, rays_d, **kwargs)
        else: # staged inference
            N = rays_o.shape[0]
            device = rays_o.device

            depth = torch.empty((N), device=device)
            image = torch.empty((N, 3), device=device)
            weights_sum = torch.empty((N), device=device)

            head = 0
            while head < N:
                tail = min(head + self.opt.max_ray_batch, N)
                results_ = self.run(rays_o[head:tail], rays_d[head:tail], **kwargs)
                depth[head:tail] = results_['depth']
                weights_sum[head:tail] = results_['weights_sum']
                image[head:tail] = results_['image']
                head += self.opt.max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum

            return results

    def run(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full', update_proposal=True, **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, [0]])
            fars = torch.minimum(fars, cam_near_far[:, [1]])
        
        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}
    
        # hierarchical sampling
        if self.training:
            all_bins = []
            all_weights = []

        s_nears = self.spacing_fn(nears) # [N, 1]
        s_fars = self.spacing_fn(fars) # [N, 1]
        
        bins = None
        weights = None

        for prop_iter in range(len(self.opt.num_steps)):
            
            if prop_iter == 0:
                # uniform sampling
                bins = torch.linspace(0, 1, self.opt.num_steps[prop_iter] + 1, device=device).unsqueeze(0) # [1, T+1]
                bins = bins.expand(N, -1) # [N, T+1]
                if perturb:
                    bins = bins + (torch.rand_like(bins) - 0.5) / (self.opt.num_steps[prop_iter])
                    bins = bins.clamp(0, 1)
            else:
                # pdf sampling
                bins = sample_pdf(bins, weights, self.opt.num_steps[prop_iter] + 1, perturb).detach() # [N, T+1]

            real_bins = self.spacing_fn_inv(s_nears * (1 - bins) + s_fars * bins) # [N, T+1] in [near, far]

            rays_t = (real_bins[..., 1:] + real_bins[..., :-1]) / 2 # [N, T]

            xyzs = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * rays_t.unsqueeze(2) # [N, T, 3]
            
            if prop_iter != len(self.opt.num_steps) - 1:
                # query proposal density
                with torch.set_grad_enabled(update_proposal):
                    sigmas = self.density(xyzs if not self.opt.contract else contract(xyzs), proposal=prop_iter)['sigma'] # [N, T]
            else:
                # last iter: query nerf
                dirs = rays_d.view(-1, 1, 3).expand_as(xyzs) # [N, T, 3]
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
                outputs = self(xyzs if not self.opt.contract else contract(xyzs), dirs, shading=shading)
                sigmas = outputs['sigma']
                rgbs = outputs['color']

            # sigmas to weights
            deltas = (real_bins[..., 1:] - real_bins[..., :-1]) # [N, T]
            deltas_sigmas = deltas * sigmas # [N, T]

            # opaque background
            if self.opt.background == 'last_sample':
                deltas_sigmas = torch.cat([deltas_sigmas[..., :-1], torch.full_like(deltas_sigmas[..., -1:], torch.inf)], dim=-1)

            alphas = 1 - torch.exp(-deltas_sigmas) # [N, T]
            transmittance = torch.cumsum(deltas_sigmas[..., :-1], dim=-1) # [N, T-1]
            transmittance = torch.cat([torch.zeros_like(transmittance[..., :1]), transmittance], dim=-1) # [N, T]
            transmittance = torch.exp(-transmittance) # [N, T]
            
            weights = alphas * transmittance # [N, T]
            weights.nan_to_num_(0)

            if self.training:
                all_bins.append(bins)
                all_weights.append(weights)

        # composite
        weights_sum = torch.sum(weights, dim=-1) # [N]
        depth = torch.sum(weights * rays_t, dim=-1) # [N]
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3]

        # extra results
        if self.training:
            results['num_points'] = xyzs.shape[0] * xyzs.shape[1]
            results['weights'] = weights

            if self.opt.lambda_proposal > 0 and update_proposal:
                results['proposal_loss'] = proposal_loss(all_bins, all_weights)
            
            if self.opt.lambda_distort > 0:
                results['distort_loss'] = distort_loss(bins, weights)

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        results['weights_sum'] = weights_sum
        results['depth'] = depth
        results['image'] = image

        return results

    def run_cuda(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full', **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]
        
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])
        
        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, self.opt.dt_gamma, self.opt.max_steps)

            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                outputs = self(xyzs, dirs, shading=shading)
                sigmas = outputs['sigma']
                rgbs = outputs['color']

            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, self.opt.T_thresh)

            results['num_points'] = xyzs.shape[0]
            results['weights'] = weights
            results['weights_sum'] = weights_sum
        
        else:
            
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < self.opt.max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, self.opt.dt_gamma, self.opt.max_steps)
                
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    outputs = self(xyzs, dirs, shading=shading)
                    sigmas = outputs['sigma']
                    rgbs = outputs['color']

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, self.opt.T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        
        results['depth'] = depth
        results['image'] = image

        return results

    @torch.no_grad()
    def mark_unseen_triangles(self, vertices, triangles, mvps, H, W):
        # vertices: coords in world system
        # mvps: [B, 4, 4]
        device = self.aabb_train.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).contiguous().float().to(device)
        
        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).contiguous().int().to(device)

        mask = torch.zeros_like(triangles[:, 0]) # [M,], for face.

        import nvdiffrast.torch as dr
        glctx = dr.RasterizeGLContext(output_db=False)

        for mvp in tqdm.tqdm(mvps):

            vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp.to(device), 0, 1)).float().unsqueeze(0) # [1, N, 4]

            # ENHANCE: lower resolution since we don't need that high?
            rast, _ = dr.rasterize(glctx, vertices_clip, triangles, (H, W)) # [1, H, W, 4]

            # collect the triangle_id (it is offseted by 1)
            trig_id = rast[..., -1].long().view(-1) - 1
            
            # no need to accumulate, just a 0/1 mask.
            mask[trig_id] += 1 # wrong for duplicated indices, but faster.
            # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

        mask = (mask == 0) # unseen faces by all cameras

        print(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')
        
        return mask # [N]


    @torch.no_grad()
    def mark_untrained_grid(self, dataset, S=64):
        
        # data: reference to the dataset object

        poses = dataset.poses # [B, 4, 4]
        intrinsics = dataset.intrinsics # [4] or [B/1, 4]
        cam_near_far = dataset.cam_near_far if hasattr(dataset, 'cam_near_far') else None # [B, 2]
  
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        if isinstance(intrinsics, np.ndarray):
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy, cx, cy = torch.chunk(intrinsics, 4, dim=-1)
        
        mask_cam = torch.zeros_like(self.density_grid)
        mask_aabb = torch.zeros_like(self.density_grid)

        # pc = []
        poses = poses.to(mask_cam.device)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # first, mark out-of-AABB region
                        mask_min = (cas_world_xyzs >= (self.aabb_train[:3] - half_grid_size)).sum(-1) == 3
                        mask_max = (cas_world_xyzs <= (self.aabb_train[3:] + half_grid_size)).sum(-1) == 3
                        mask_aabb[cas, indices] += (mask_min & mask_max).reshape(-1)

                        # second, mark out-of-camera region
                        # split pose to batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            cam_xyzs[:, :, 2] *= -1 # crucial, camera forward is negative now...

                            if torch.is_tensor(fx):
                                cx_div_fx = cx[head:tail] / fx[head:tail]
                                cy_div_fy = cy[head:tail] / fy[head:tail]
                            else:
                                cx_div_fx = cx / fx
                                cy_div_fy = cy / fy
                            
                            min_near = self.opt.min_near if cam_near_far is None else cam_near_far[head:tail, 0].unsqueeze(1)
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > min_near # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < (cx_div_fx * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < (cy_div_fy * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask = (mask_z & mask_x & mask_y).sum(0).bool().reshape(-1) # [N]

                            # for visualization
                            # pc.append(cas_world_xyzs[0][mask])

                            # update mask_cam 
                            mask_cam[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[((mask_cam == 0) | (mask_aabb == 0))] = -1

        print(f'[mark untrained grid] {((mask_cam == 0) | (mask_aabb == 0)).sum()} from {self.grid_size ** 3 * self.cascade}')

    
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid

        with torch.no_grad():

            tmp_grid = - torch.ones_like(self.density_grid)
            
            # full update.
            if self.iter_density < 16:
                X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
                Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
                Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long() # [N]
                            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # query density
                                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                                    sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                                # assign 
                                tmp_grid[cas, indices] = sigmas

            # partial update (half the computation)
            else:
                N = self.grid_size ** 3 // 4 # H * H * H / 4
                for cas in range(self.cascade):
                    # random sample some positions
                    coords = torch.randint(0, self.grid_size, (N, 3), device=self.aabb_train.device) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.aabb_train.device)
                    occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # query density
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                    # assign 
                    tmp_grid[cas, indices] = sigmas

            # ema update
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
            
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid.detach(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')
        return None
        