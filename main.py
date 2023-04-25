import torch
import argparse

from nerf.gui import NeRFGUI
from nerf.utils import *

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="recommended settings")
    parser.add_argument('-O2', action='store_true', help="recommended settings")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    ### testing options
    parser.add_argument('--save_cnt', type=int, default=50, help="save checkpoints for $ times during training")
    parser.add_argument('--eval_cnt', type=int, default=10, help="perform validation for $ times during training")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_no_video', action='store_true', help="test mode: do not save video")
    parser.add_argument('--test_no_mesh', action='store_true', help="test mode: do not save mesh")
    parser.add_argument('--camera_traj', type=str, default='interp', help="interp for interpolation, circle for circular camera")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='colmap', choices=['nerf', 'colmap', 'dtu'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'all'])
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--random_image_batch', action='store_true', help="randomly sample rays from all images per step in training")
    parser.add_argument('--downscale', type=int, default=1, help="downscale training images")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1, help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--enable_cam_near_far', action='store_true', help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true', help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--T_thresh', type=float, default=1e-4, help="minimum transmittance to continue ray marching")

    ### training options
    parser.add_argument('--iters', type=int, default=20000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, nargs='*', default=[256, 96, 48], help="num steps sampled per ray for each proposal level (only valid when NOT using --cuda_ray)")
    parser.add_argument('--contract', action='store_true', help="apply spatial contraction as in mip-nerf 360, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--background', type=str, default='last_sample', choices=['white', 'random', 'last_sample'], help="training background mode")

    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096 * 4, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution")
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--diffuse_step', type=int, default=1000, help="training iters that only trains diffuse color for better initialization")
    
    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true', help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18, help="target num points for each training step, only work with adaptive num_rays")

    # regularizations
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_proposal', type=float, default=1, help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_distort', type=float, default=0.002, help="loss scale (only for non-cuda-ray mode)")

    ### mesh options
    parser.add_argument('--mcubes_reso', type=int, default=512, help="resolution for marching cubes")
    parser.add_argument('--env_reso', type=int, default=256, help="max layers (resolution) for env mesh")
    parser.add_argument('--decimate_target', type=int, default=3e5, help="decimate target for number of triangles, <=0 to disable")
    parser.add_argument('--mesh_visibility_culling', action='store_true', help="cull mesh faces based on visibility in training dataset")
    parser.add_argument('--visibility_mask_dilation', type=int, default=5, help="visibility dilation")
    parser.add_argument('--clean_min_f', type=int, default=8, help="mesh clean: min face count for isolated mesh")
    parser.add_argument('--clean_min_d', type=int, default=5, help="mesh clean: min diameter for isolated mesh")

    ### GUI options
    parser.add_argument('--vis_pose', action='store_true', help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1000, help="GUI width")
    parser.add_argument('--H', type=int, default=1000, help="GUI height")
    parser.add_argument('--radius', type=float, default=1, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.preload = True
        opt.cuda_ray = True
        opt.mark_untrained = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True
    
    if opt.O2:
        opt.fp16 = True
        opt.bound = 128 # large enough
        opt.preload = True
        opt.contract = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True
    
    if opt.contract:
        # mark untrained is not correct in contraction mode...
        opt.mark_untrained = False

    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else: # nerf
        from nerf.provider import NeRFDataset
    
    seed_everything(opt.seed)

    from nerf.network import NeRFNetwork

    model = NeRFNetwork(opt)
    
    # criterion = torch.nn.MSELoss(reduction='none')
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            if not opt.test_no_video:
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)] # set up metrics
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True) # test and save video
            
            # if not opt.test_no_mesh:
            #     # need train loader to get camera poses for visibility test
            #     if opt.mesh_visibility_culling:
            #         train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
            #     trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
        
    else:
        
        optimizer = torch.optim.Adam(model.get_params(opt.lr), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // max(1, opt.save_cnt)) # save ~50 times during the training
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        # colmap can estimate a more compact AABB
        if not opt.contract and opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval, save_interval=save_interval)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()

            trainer.metrics = [PSNRMeter(),]
            trainer.train(train_loader, valid_loader, max_epoch)
            
            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            trainer.evaluate(valid_loader)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            # trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)