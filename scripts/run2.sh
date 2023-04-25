CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_llff_data/fern/ --workspace trial_fern --downscale 4
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_llff_data/fern/ --workspace trial2_fern --downscale 4 --with_sam --init_ckpt trial_fern/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_llff_data/horns/ --workspace trial_horns --downscale 4
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_llff_data/horns/ --workspace trial2_horns --downscale 4 --with_sam --init_ckpt trial_horns/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_llff_data/orchids/ --workspace trial_orchids --downscale 4
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_llff_data/orchids/ --workspace trial2_orchids --downscale 4 --with_sam --init_ckpt trial_orchids/checkpoints/ngp.pth --iters 5000

