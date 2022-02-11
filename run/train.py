import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
from render_vid import render_vid
torch.manual_seed(0)
np.random.seed(0)


def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)


def test(model, dataset, visualizer, opt, test_steps=0):
    print('-----------------------------------Testing-----------------------------------')
    model.eval()
    total_num = dataset.total
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    count=0
    for i in range(0, total_num, opt.test_num_step): # 1 if test_steps == 10000 else opt.test_num_step
        data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        edge_mask = torch.zeros([height, width], dtype=torch.bool)
        edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
        edge_mask=edge_mask.reshape(-1) > 0
        np_edge_mask=edge_mask.numpy().astype(bool)
        totalpixel = pixel_idx.shape[1]
        tmpgts = {}
        tmpgts["gt_image"] = data['gt_image'].clone()
        tmpgts["gt_mask"] = data['gt_mask'].clone() if "gt_mask" in data else None

        # data.pop('gt_image', None)
        data.pop('gt_mask', None)

        visuals = None
        stime = time.time()
        ray_masks = []
        ray_depth_masks = []
        xyz_world_sect_plane_lst = []
        for k in range(0, totalpixel, chunk_size):
            start = k
            end = min([k + chunk_size, totalpixel])
            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            model.set_input(data)

            if opt.bgmodel.endswith("plane"):
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                if len(bg_ray_lst) > 0:
                    bg_ray_all = bg_ray_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                    bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                data["bg_ray"] = bg_ray

                # xyz_world_sect_plane_lst.append(xyz_world_sect_plane)
            model.test(gen_points=True)
            curr_visuals = model.get_current_visuals(data=data)

            # print("loss", mse2psnr(torch.nn.MSELoss().to("cuda")(curr_visuals['coarse_raycolor'], tmpgts["gt_image"].view(1, -1, 3)[:, start:end, :].cuda())))
            # print("sum", torch.sum(torch.square(tmpgts["gt_image"].view(1, -1, 3)[:, start:end, :] - tmpgts["gt_image"].view(height, width, 3)[data["pixel_idx"][0,...,1].long(), data["pixel_idx"][0,...,0].long(),:])))
            chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height, width, 3)).astype(chunk.dtype)
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk

            else:
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
            if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                ray_masks.append(model.output["ray_mask"] > 0)
        ray_masks = torch.cat(ray_masks, dim=1)
        # visualizer.save_neural_points(data["id"].cpu().numpy()[0], (raydir.cuda() + data["campos"][:, None, :]).squeeze(0), None, data, save_ref=True)
        # exit()
        # print("curr_visuals",curr_visuals)
        pixel_idx=pixel_idx.to(torch.long)
        if 'gt_image' in model.visual_names:
            visuals['gt_image'] = torch.zeros((height*width, 3), dtype=torch.float32)
            visuals['gt_image'][edge_mask,:] =  tmpgts['gt_image'].clone()
        if 'gt_mask' in curr_visuals:
            visuals['gt_mask'] = np.zeros((height, width, 3)).astype(chunk.dtype)
            visuals['gt_mask'][np_edge_mask,:] = tmpgts['gt_mask']
        if 'ray_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            print(visuals['ray_masked_coarse_raycolor'].shape, ray_masks.cpu().numpy().shape)
            visuals['ray_masked_coarse_raycolor'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'ray_depth_masked_gt_image' in model.visual_names:
            visuals['ray_depth_masked_gt_image'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['ray_depth_masked_gt_image'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'gt_image_ray_masked' in model.visual_names:
            visuals['gt_image_ray_masked'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['gt_image_ray_masked'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        for key, value in visuals.items():
            visualizer.print_details("{}:{}".format(key, visuals[key].shape))

            visuals[key] = visuals[key].reshape(height, width, 3)
        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i)

        acc_dict = {}
        if "coarse_raycolor" in opt.test_color_loss_items:
            loss = torch.nn.MSELoss().to("cuda")(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), visuals["gt_image"].view(1, -1, 3).cuda())
            acc_dict.update({"coarse_raycolor": loss})
            print("coarse_raycolor", loss, mse2psnr(loss))

        if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
            masked_gt = tmpgts["gt_image"].view(1, -1, 3).cuda()[ray_masks,:].reshape(1, -1, 3)
            ray_masked_coarse_raycolor = torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3)[:,edge_mask,:][ray_masks,:].reshape(1, -1, 3)

            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_gt")
            # filepath = os.path.join("/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # tmpgtssave = tmpgts["gt_image"].view(1, -1, 3).clone()
            # tmpgtssave[~ray_masks,:] = 1.0
            # img = np.array(tmpgtssave.view(height,width,3))
            # save_image(img, filepath)
            #
            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_coarse_raycolor")
            # filepath = os.path.join(
            #     "/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # csave = torch.zeros_like(tmpgts["gt_image"].view(1, -1, 3))
            # csave[~ray_masks, :] = 1.0
            # csave[ray_masks, :] = torch.as_tensor(visuals["coarse_raycolor"]).view(1, -1, 3)[ray_masks,:]
            # img = np.array(csave.view(height, width, 3))
            # save_image(img, filepath)

            loss = torch.nn.MSELoss().to("cuda")(ray_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_masked_coarse_raycolor", loss, mse2psnr(loss)))

        if "ray_depth_mask" in model.output and "ray_depth_masked_coarse_raycolor" in opt.test_color_loss_items:
            ray_depth_masks = model.output["ray_depth_mask"].reshape(model.output["ray_depth_mask"].shape[0], -1)
            masked_gt = torch.masked_select(tmpgts["gt_image"].view(1, -1, 3).cuda(), (ray_depth_masks[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
            ray_depth_masked_coarse_raycolor = torch.masked_select(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), ray_depth_masks[..., None].expand(-1, -1, 3).reshape(1, -1, 3))

            loss = torch.nn.MSELoss().to("cuda")(ray_depth_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_depth_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_depth_masked_coarse_raycolor", loss, mse2psnr(loss)))
        print(acc_dict.items())
        visualizer.accumulate_losses(acc_dict)
        count += 1

    visualizer.print_losses(count)
    # psnr = visualizer.get_psnr(opt.test_color_loss_items[0])
    visualizer.reset()

    print('--------------------------------Finish Test Rendering--------------------------------')

    report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, ["psnr", "ssim", "rmse"],
                   [i for i in range(0, count)],
                   imgStr="step-%04d-{}_raycolor.png".format("coarse"))


    print('--------------------------------Finish Evaluation--------------------------------')
    return

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def main():
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    print("opt.color_loss_items ", opt.color_loss_items)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)

    data_loader = create_data_loader(opt)
    dataset_size = len(data_loader)
    print('# training images = {}'.format(dataset_size))
    if opt.resume_dir:
        resume_dir = opt.resume_dir
        resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
        opt.resume_iter = resume_iter
        if resume_iter is None:
            epoch_count = 1
            total_steps = 0
            print("No previous checkpoints, start from scratch!!!!")
        else:
            opt.resume_iter = resume_iter
            states = torch.load(
                os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)))
            epoch_count = states['epoch_count']
            total_steps = states['total_steps']
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Continue training from {} epoch'.format(opt.resume_iter))
            print("Iter: ", total_steps)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
        epoch_count = 1
        total_steps = 0
    print("opt.resume_dir ", opt.resume_dir, opt.resume_iter)

    # load model
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(32, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.split = "test"
    # test_dataset = create_dataset(test_opt)

    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()
    if total_steps > 0:
        for scheduler in model.schedulers:
            for i in range(total_steps):
                scheduler.step()
    for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(data_loader):
            if opt.maximum_step is not None and total_steps >= opt.maximum_step:
                break
            total_steps += 1

            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters(total_steps=total_steps)

            losses = model.get_current_losses()
            visualizer.accumulate_losses(losses)

            if opt.lr_policy.startswith("iter"):
                model.update_learning_rate(opt=opt, total_steps=total_steps)

            if total_steps and total_steps % opt.print_freq == 0:
                if opt.show_tensorboard:
                    visualizer.plot_current_losses_with_tb(total_steps, losses)
                visualizer.print_losses(total_steps)
                visualizer.reset()

            if hasattr(opt, "save_point_freq") and total_steps and total_steps % opt.save_point_freq == 0:
                visualizer.save_neural_points(total_steps, model.neural_points.xyz, model.neural_points.points_embeding, data, save_ref=opt.load_points==0)

            # if opt.train_and_test == 1 and total_steps % opt.test_freq == 0:
            #     test(model, test_dataset, visualizer, test_opt, total_steps)

            # if opt.vid == 1 and total_steps % opt.test_freq == 0:
            #     model.opt.no_loss = 1
            #     render_vid(model, test_dataset, visualizer, test_opt, total_steps)
            #     model.opt.no_loss = 0
            try:
                if total_steps % opt.save_iter_freq == 0 and total_steps > 0:
                    other_states = {
                        'epoch_count': epoch,
                        'total_steps': total_steps,
                    }
                    print('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                    model.save_networks(total_steps, other_states)
                    # if opt.vid == 1:
                    #     model.opt.is_train = 0
                    #     model.opt.no_loss = 1
                    #     test_opt.nerf_splits = ["test"]
                    #     test_opt.name = opt.name + "/test_{}".format(total_steps)
                    #     test_opt.test_num = 999
                    #     render_vid(model, test_dataset, Visualizer(test_opt), test_opt, total_steps)
                    #     model.opt.no_loss = 0
                    #     model.opt.is_train = 1

            except Exception as e:
                print(e)
            if total_steps % opt.test_freq == 0 and total_steps < (opt.maximum_step - 1) and total_steps > 0:
                test_opt.nerf_splits = ["test"]
                test_opt.split = "test"
                test_opt.name = opt.name + "/test_{}".format(total_steps)
                test_opt.test_num_step = opt.test_num_step
                test_dataset = create_dataset(test_opt)
                model.opt.is_train = 0
                model.opt.no_loss = 1
                test(model, test_dataset, Visualizer(test_opt), test_opt, total_steps)
                model.opt.no_loss = 0
                model.opt.is_train = 1

        # try:
        #     print("saving the model at the end of epoch")
        #     other_states = {'epoch_count': epoch, 'total_steps': total_steps}
        #     model.save_networks('latest', other_states)
        #
        # except Exception as e:
        #     print(e)

        if opt.vid == 1:
            model.opt.is_train = 0
            model.opt.no_loss = 1
            render_vid(model, test_dataset, visualizer, test_opt, total_steps)
            model.opt.no_loss = 0
            model.opt.is_train = 1

        if opt.maximum_step is not None and total_steps == opt.maximum_step:
            print('{}: End of stepts {} / {} \t Time Taken: {} sec'.format(
                opt.name, total_steps, opt.maximum_step,
                time.time() - epoch_start_time))
            break

        print('{}: End of epoch {} / {} \t Time Taken: {} sec'.format(
            opt.name, epoch, opt.niter + opt.niter_decay,
            time.time() - epoch_start_time))
        if not opt.lr_policy.startswith("iter"):
            model.update_learning_rate(opt=opt)

    other_states = {
        'epoch_count': epoch,
        'total_steps': total_steps,
    }
    print('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
    model.save_networks(total_steps, other_states)

    test_opt.nerf_splits = ["test"]
    test_opt.split = "test"
    test_opt.test_num_step=1
    test_opt.name = opt.name + "/test_{}".format(total_steps)
    test_dataset = create_dataset(test_opt)
    model.opt.no_loss = 1
    model.opt.is_train = 0
    test(model, test_dataset, Visualizer(test_opt), test_opt, total_steps)
    # model.opt.no_loss = 0
    # model.opt.is_train = 1

if __name__ == '__main__':
    main()
