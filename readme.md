# nerf-template

A simple template for practicing NeRF projects.

This is basically a clean and enhanced version of [torch-ngp](https://github.com/ashawkey/torch-ngp) focusing on static NeRF reconstruction of realistic scenes.

Notable changes that improve performance:
* dataset: random sampling from all training images at each step.
* dataset: adaptive number of rays during training based on number of points evaluated.
* model: proposal network for sampling points (non --cuda_ray mode).
* model: spatial contraction for unbounded scenes.


# Install

```bash
git clone https://github.com/ashawkey/nerf_template.git
cd nerf_template
```

### Install with pip
```bash
pip install -r requirements.txt
```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Tested environments
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.

# Usage

We majorly support COLMAP dataset like [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).
Please download and put them under `./data`.

For custom datasets:
```bash
# prepare your video or images under /data/custom, and run colmap (assumed installed):
python scripts/colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python scripts/colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
```

### Basics
First time running will take some time to compile the CUDA extensions.
```bash
## -O: instant-ngp
# prune sampling points by maintaining a density grid
python main.py data/bonsai/ --workspace trial_bonsai_ngp --enable_cam_center --downscale 4 -O --background random --bound 8

## -O2: nerfstudio nerfacto
# use proposal network to predict sampling points
python main.py data/bonsai/ --workspace trial_bonsai_nerfacto --enable_cam_center --downscale 4 -O2

# MeRF network backbone
python main.py data/bonsai/ --workspace trial_bonsai_nerfacto --enable_cam_center --downscale 4 -O2 --backbone merf
```

### Advanced Usage
```bash
### -O: equals
--fp16 --preload
--cuda_ray --mark_untrained
--adaptive_num_rays --random_image_batch

### -O2: equals
--fp16 --preload
--contract --bound 128
--adaptive_num_rays --random_image_batch 

### load checkpoint
--ckpt latest # by default we load the latest checkpoint in the workspace
--ckpt scratch # train from scratch.
--ckpt trial/checkpoints/xxx.pth # specify it by path

### training
--num_rays 4096 # number of rays to evaluate per training step
--adaptive_num_rays # ignore --num_rays and use --num_points to dynamically adjust number of rays.
--num_points 262144 # targeted number of points to evaluate per training step (to adjust num_rays)

### testing
--test # test, save video and mesh
--test_no_video # do not save video
--test_no_mesh # do not save mesh

### dataset related
--data_format [colmap|nerf] # dataset format
--enable_cam_center # use camera center instead of sparse point cloud center as the scene center (colmap dataset only) (only for 360-degree captured datasets, do not use this for forward-facing datasets!)
--enable_cam_near_far # estimate camera near & far from sparse points (colmap dataset only)

--bound 16 # scene bound set to [-16, 16]^3.
--scale 0.3 # camera scale, if not specified, automatically estimate one based on camera positions.

### visualization 
--vis_pose # viusalize camera poses and sparse points (sparse points are colmap dataset only)
--gui # open gui (only for testing, training in gui is not well supported!)

### balance between surface quality / rendering quality

# increase these weights to get better surface quality but worse rendering quality
--lambda_tv 1e-7 # total variation loss
--lambda_entropy 1e-3 # entropy on rendering weights (transparency, alpha), encourage them to be either 0 or 1
```

Please check the `scripts` directory for more examples on common datasets, and check `main.py` for all options.

### Performance reference 

|        | Bonsai | Counter | Kitchen | Room | Bicycle | Garden | Stump |
| ---    | --- | --- | --- | --- | --- | --- | --- |
| MipNeRF 360 (~days)          | 33.46 | 29.55 | 32.23 | 31.63 | 24.57 | 26.98 | 26.40 | 
| ours-ngp (~8 minutes)        | 28.99 | 25.18 | 26.42 | 28.58 | 21.31 | 23.70 | 22.73 |
| ours-nerfacto (~12 minutes)  | 31.10 | 26.65 | 30.61 | 31.44 | 23.74 | 25.31 | 25.48 |

Ours are tested on a V100. 
Please check the commands under `scripts/` to reproduce.

### Acknowledgement
This repository is based on:
* [torch-ngp](https://github.com/ashawkey/torch-ngp)
* [DearPyGui](https://github.com/hoffstadt/DearPyGui)
* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
* [nerfacc](https://github.com/KAIR-BAIR/nerfacc)