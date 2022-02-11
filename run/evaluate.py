import os, sys, time, argparse, cv2
import numpy as np
try:
    from skimage.measure import compare_ssim
    from skimage.measure import compare_psnr
except:
    from skimage.metrics import structural_similarity
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

    def compare_ssim(gt, img, win_size, multichannel=True):
        return structural_similarity(gt, img, win_size=win_size, multichannel=multichannel)


import torch
from skimage.metrics import mean_squared_error
import lpips



parser = argparse.ArgumentParser(description="compute scores")

parser.add_argument('-i', '--imgFolder', help="The folder that contain output images.")
parser.add_argument('-g', '--gtFolder', default=None, help="The folder that contain gt images. By default it uses imgFolder")
parser.add_argument('-o', '--outFolder', default=None, help="The folder that contain output files. By default it uses imgFolder")

parser.add_argument('-is', '--imgStr', default="step-%04d-fine_raycolor.png", help="The string format for input images.")
parser.add_argument('-gs', '--gtStr', default="step-%04d-gt_image.png", help="The string format for GT images.")

parser.add_argument('-l', '--id_list', nargs='+', default=list(range(999)),  help="The list of ids to test. By default it's 0~999.")

parser.add_argument('-m', '--metrics', nargs='+', default=["psnr", "ssim", "lpips", "vgglpips"],  help="The list of metrics to compute. By default it computes psnr, ssim and rmse.")


def report_metrics(gtFolder, imgFolder, outFolder, metrics, id_list, imgStr="step-%04d-fine_raycolor.png", gtStr="step-%04d-gt_image.png", use_gpu=False, print_info=True):
    total ={}
    loss_fn, loss_fn_vgg = None, None
    if print_info:
        print("test id_list", id_list)
        print(gtFolder, imgFolder, outFolder)
        print(imgStr, gtStr)
    if "lpips" in metrics:
        loss_fn = lpips.LPIPS(net='alex', version='0.1') #   we follow NVSF to use alex 0.1,  NeRF use lpips.LPIPS(net='vgg')
        loss_fn = loss_fn.cuda() if use_gpu else loss_fn
    if "vgglpips" in metrics:
        loss_fn_vgg = lpips.LPIPS(net='vgg', version='0.1') #lpips.LPIPS(net='vgg')
        loss_fn_vgg = loss_fn_vgg.cuda() if use_gpu else loss_fn_vgg
    for i in id_list:

        img = cv2.imread(imgFolder+"/"+imgStr%i)
        gt = cv2.imread(gtFolder+"/"+gtStr%i)
        # print("img", imgFolder+"/"+imgStr%i)
        if img is None or gt is None:
            break

        img = np.asarray(img, np.float32)/255.0
        gt = np.asarray(gt, np.float32)/255.0
        for key in metrics:
            if key == "psnr":
                val = compare_psnr(gt, img)
            elif key == "ssim":
                val = compare_ssim(gt, img, 11, multichannel=True)
            elif key == "lpips":
                # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_tensor = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                gt_tensor = torch.from_numpy(gt)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                img_tensor = img_tensor.cuda() if use_gpu else img_tensor
                gt_tensor = gt_tensor.cuda() if use_gpu else gt_tensor
                val = loss_fn(img_tensor, gt_tensor).item()
            elif key == "vgglpips":
                # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_tensor = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                gt_tensor = torch.from_numpy(gt)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
                img_tensor = img_tensor.cuda() if use_gpu else img_tensor
                gt_tensor = gt_tensor.cuda() if use_gpu else gt_tensor
                val = loss_fn_vgg(img_tensor, gt_tensor).item()
            elif key == "rmse":
                val = np.sqrt(mean_squared_error(gt, img))
            else:
                raise NotImplementedError("metrics of {} not implemented".format(key))
            if key not in total:
                total[key] = [val]
            else:
                total[key].append(val)
    del loss_fn
    del loss_fn_vgg

    torch.cuda.empty_cache()
    print(len(id_list), "images computed")
    if len(total) > 0:
        outStr = ""
        for key in total.keys():
            vals = np.asarray(total[key]).reshape(-1)
            np.savetxt(outFolder+"/"+key+'.txt', vals)
            outStr+= key + ": %.6f\n"%np.mean(vals)
        print(outStr)
        with open(outFolder+"/scores.txt", "w") as f:
            f.write(outStr)



############################
if __name__ == '__main__':
    args = parser.parse_args()

    if args.gtFolder is None:
        args.gtFolder = args.imgFolder
    if args.outFolder is None:
        args.outFolder = args.imgFolder

    report_metrics(args.gtFolder, args.imgFolder, args.outFolder, args.metrics, args.id_list, imgStr=args.imgStr, gtStr=args.gtStr, use_gpu=True, print_info=False)



# python run/evaluate.py -i ${nrCheckpoint}/dragon-test/images -g ${nrCheckpoint}/dragon-test/images -is step-%04d-fine_raycolor.png
# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/lego_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pcollego360_load_confcolordir_KNN8_LRelu_grid320_553_agg2_prl2e3/test_250000/images --imgStr "lego_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/lego_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pcollego360_load_confcolordir_KNN8_LRelu_grid320_553_agg2_prl2e3/test_250000/images --imgStr "lego_test_64_50_%d_infer.png"



# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/ship_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pcolship360_load_confcolordir_KNN8_LRelu_grid320_553_agg2_prl2e3/test_250000/images --imgStr "ship_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/ship_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pcolship360_load_confcolordir_KNN8_LRelu_grid320_553_agg2_prl2e3/test_250000/images --imgStr "ship_test_64_50_%d_infer.png"



# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/chair_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pchair360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_zeroone0001_confree_80_pru1_e4_prl2e3.sh/test_250000/images --imgStr "chair_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/chair_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pchair360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_zeroone0001_confree_80_pru1_e4_prl2e3.sh/test_250000/images --imgStr "chair_test_64_50_%d_infer.png"



# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/materials_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/materials360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_zeroone0001_confree_80_pru1_e4_prl2e3/test_250000/images --imgStr "materials_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/materials_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/materials360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_zeroone0001_confree_80_pru1_e4_prl2e3/test_250000/images --imgStr "materials_test_64_50_%d_infer.png"



# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/drums_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/drums360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_pru6_e4_prle3/test_250000/images --imgStr "drums_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/drums_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/drums360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_pru6_e4_prle3/test_250000/images --imgStr "drums_test_64_50_%d_infer.png"


# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/ficus_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pficus360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_prl8e3/test_250000/images --imgStr "ficus_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/ficus_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pficus360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_prl8e3/test_250000/images --imgStr "ficus_test_64_50_%d_infer.png"


# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/mic_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pmic360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_prl2e3/test_250000/images --imgStr "mic_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/mic_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/pmic360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask0_agg2_prl2e3/test_250000/images --imgStr "mic_test_64_50_%d_infer.png"



# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/hotdog_8_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/photdog360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask5_agg2_zeroone0.0001_confree_80_pru1_e4_prl2e3/test_250000/images --imgStr "hotdog_test_8_50_%d_infer.png"

# python run/evaluate.py -i /home/xharlie/user_space/dev/npbg/nerf_synth/npbg_results/hotdog_64_50/ -g /home/xharlie/user_space/codes/testNr/checkpoints/photdog360_confcolordir_KNN8_LRelu_grid320_333_fullgeomask5_agg2_zeroone0.0001_confree_80_pru1_e4_prl2e3/test_250000/images --imgStr "hotdog_test_64_50_%d_infer.png"