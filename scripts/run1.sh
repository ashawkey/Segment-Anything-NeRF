CUDA_VISIBLE_DEVICES=5 python main.py data/room/ --workspace trial_room --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/room/ --workspace trial2_room --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_room/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py data/counter/ --workspace trial_counter --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/counter/ --workspace trial2_counter --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_counter/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py data/kitchen/ --workspace trial_kitchen --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/kitchen/ --workspace trial2_kitchen --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_kitchen/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py data/bonsai/ --workspace trial_bonsai --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/bonsai/ --workspace trial2_bonsai --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_bonsai/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py data/bicycle/ --workspace trial_bicycle --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/bicycle/ --workspace trial2_bicycle --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_bicycle/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py data/stump/ --workspace trial_stump --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/stump/ --workspace trial2_stump --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_stump/checkpoints/ngp.pth --iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py data/garden/ --workspace trial_garden --enable_cam_center --downscale 4
CUDA_VISIBLE_DEVICES=5 python main.py data/garden/ --workspace trial2_garden --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_garden/checkpoints/ngp.pth --iters 5000
