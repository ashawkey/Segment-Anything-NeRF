# Segment-Anything NeRF

🎉🎉🎉 Welcome to the Segment-Anything NeRF GitHub repository! 🎉🎉🎉

**Segment-Anything NeRF** is a novel approach for performing segmentation in a Neural Radiance Fields (NeRF) framework. Our approach renders the semantic feature of a certain view directly, eliminating the need for the forward process of the backbone of the segmentation model. By leveraging the light-weight [SAM](https://github.com/facebookresearch/segment-anything) decoder, we can achieve **interactive 3D-consistent segmentation** at 5 FPS (rendering 512x512 image) on a V100.



https://user-images.githubusercontent.com/23420768/235310664-b432b87e-cfa9-4cbe-9224-1ca90c2e380d.mp4



https://user-images.githubusercontent.com/23420768/235310629-f0d5c1fe-7b07-4c32-9500-bb27880fa50d.mp4






# News
[2023/4/29] Add a demo of Open-Vocabulary Segmentation in NeRF based on [X-Decoder](https://github.com/microsoft/X-Decoder).



# Key features

* Learn 3D consistent SAM backbone features along with RGB and density, so we can bypass the ViT-Huge encoder and use ray marching to produce SAM features efficiently.
* Online distillation with camera augmentation and caching for robust and fast training (~1 hour per scene for two stages on a V100).

NOTE: This is a **work in progress**, more demonstration (e.g., open-vocabulary segmentation) and a technical report is on the way!


# Install

```bash
git clone https://github.com/ashawkey/Segment-Anything-NeRF.git
cd Segment-Anything-NeRF

# download SAM ckpt
mkdir pretrained && cd pretrained
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
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
cd gridencoder
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the gridencoder/ folder, since this only install the built extension.)
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

First time running will take some time to compile the CUDA extensions.
```bash
### train rgb
python main.py data/garden/ --workspace trial_garden --enable_cam_center --downscale 4

### train sam features
# --with_sam: enable sam prediction
# --init_ckpt: specify the latest checkpoint from rgb training
python main.py data/garden/ --workspace trial2_garden --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_garden/checkpoints/ngp.pth --iters 5000

### test sam (interactive GUI, recommended!)
# left drag & middle drag & wheel scroll: move camera
# right click: add/remove point marker
# NOTE: only square images are supported for now!
python main.py data/garden/ --workspace trial2_garden --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_garden/checkpoints/ngp.pth --test --gui

# test sam (without GUI, random points query)
python main.py data/garden/ --workspace trial2_garden --enable_cam_center --downscale 4 --with_sam --init_ckpt trial_garden/checkpoints/ngp.pth --test
```

Please check the `scripts` directory for more examples on common datasets, and check `main.py` for all options.

# Acknowledgement

* [Segment-Anything](https://github.com/facebookresearch/segment-anything):
    ```
    @article{kirillov2023segany,
        title={Segment Anything},
        author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
        journal={arXiv:2304.02643},
        year={2023}
    }
    ```
* [X-Decoder](https://github.com/microsoft/X-Decoder):
    ```
    @article{zou2022generalized,
      title={Generalized Decoding for Pixel, Image, and Language},
      author={Zou, Xueyan and Dou, Zi-Yi and Yang, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Behl, Harkirat and Wang, Jianfeng and Yuan, Lu and others},
      journal={arXiv preprint arXiv:2212.11270},
      year={2022}
    }
    ```

# Citation

If you find this work useful, a citation will be appreciated via:

```
@misc{segment-anything-nerf,
    Author = {Jiaxiang Tang and Xiaokang Chen and Diwen Wan and Jingbo Wang and Gang Zeng},
    Year = {2023},
    Note = {https://github.com/ashawkey/Segment-Anything-NeRF},
    Title = {Segment-Anything NeRF}
}
```
