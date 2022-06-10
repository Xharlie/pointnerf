#!/bin/bash
nrCheckpoint="../checkpoints"
nrDataRoot="../data_src"
name='inftest_scan8'

resume_iter=best
data_root="${nrDataRoot}/dtu"
scan="scan8"

load_points=0
feat_grad=1
conf_grad=1
dir_grad=1
color_grad=1
vox_res=800
normview=2
full_comb=2
default_conf=1 #1000
prune_thresh=0.1
prune_iter=10000

feedforward=0
ref_vid=0
bgmodel="plane" #"plane"
depth_occ=1
depth_vid="0"
trgt_id=3
manual_depth_view=1 #2

init_view_num=3
pre_d_est="${nrCheckpoint}/MVSNet/model_000015.ckpt"
manual_std_depth=0.0
depth_conf_thresh=0.8

uni_depth=1
geo_cnsst_num=10
appr_feature_str0="imgfeat_0_0123 dir_0 point_conf"
point_conf_mode="01" # 0 for only at features, 1 for multi at weight
point_dir_mode="01" # 0 for only at features, 1 for color branch
point_color_mode="01" # 0 for only at features, 1 for color branch

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
feature_init_method="rand" #"rand" # "zeros"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=20
radius_limit_scale=0
depth_limit_scale=0
vscale=" 2 2 1 "
kernel_size=" 3 3 3 "
SR=40
K=8
P=20
NN=2

act_type="LeakyReLU"

agg_intrp_order=1
agg_distance_kernel="linear" #"avg" #"feat_intrp"
weight_xyz_freq=2
weight_feat_dim=8

point_features_dim=63
shpnt_jitter="uniform" #"uniform" # uniform gaussian

which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer1=2 #2
shading_feature_mlp_layer2=0 #1
shading_feature_mlp_layer3=2 #1
shading_alpha_mlp_layer=1
shading_color_mlp_layer=4
shading_feature_num=256
dist_xyz_freq=5
num_feat_freqs=3
dist_xyz_deno=0


raydist_mode_unit=1
dataset_name='dtu_ft'
pin_data_in_memory=1
model='mvs_points_volumetric'
near_plane=2.0
far_plane=4.725
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #
domain_size='1'
dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4

num_pos_freqs=10
num_viewdir_freqs=4 #6

random_sample='random'

random_sample_size=48 # 32 * 32 = 1024
batch_size=1
plr=0.002 # 0.0005 #0.00015
lr=0.001 # 0.0005 #0.00015
lr_policy="iter_exponential_decay"
lr_decay_iters=30000
lr_decay_exp=0.1
gpu_ids='3'
checkpoints_dir="${nrCheckpoint}/dtu/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_color_dir_agg2"

save_iter_freq=10000
save_point_freq=308000 #1 #308000 #1ss
maximum_step=0 #800000

niter=10000 #1000000
niter_decay=10000 #250000
n_threads=2
train_and_test=0 #1
test_num=10
test_freq=1000 #1200 #1200 #30184 #30184 #50000
print_freq=40
test_num_step=1

prob_freq=2000 #1000 is bad #10001
prob_num_step=1

zero_epsilon=1e-3

color_loss_weights=" 1.0 0.0 0.0 "
visual_items='ray_depth_masked_coarse_raycolor ray_depth_masked_gt_image '
color_loss_items='ray_masked_coarse_raycolor ray_miss_coarse_raycolor ray_depth_masked_coarse_raycolor'
#test_color_loss_items='ray_depth_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor ray_masked_coarse_raycolor'
test_color_loss_items='ray_depth_masked_coarse_raycolor '


zero_one_loss_items='conf_coefficient' #regularize background to be either 0 or 1
zero_one_loss_weights=" 0.0001 "
sparse_loss_weight=0


bg_color="black" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"

cd run

python3 train_ft.py \
        --experiment $name \
        --scan $scan \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --maximum_step $maximum_step \
        --plr $plr \
        --lr $lr \
        --lr_policy $lr_policy \
        --lr_decay_iters $lr_decay_iters \
        --lr_decay_exp $lr_decay_exp \
        --gpu_ids $gpu_ids \
        --checkpoints_dir $checkpoints_dir \
        --save_iter_freq $save_iter_freq \
        --niter $niter \
        --niter_decay $niter_decay \
        --n_threads $n_threads \
        --pin_data_in_memory $pin_data_in_memory \
        --train_and_test $train_and_test \
        --test_num $test_num \
        --test_freq $test_freq \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --prob_freq $prob_freq \
        --prob_num_step $prob_num_step \
        --print_freq $print_freq \
        --bg_color $bg_color \
        --split $split \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --load_points $load_points \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --feature_init_method $feature_init_method \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --save_point_freq $save_point_freq  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --feedforward $feedforward \
        --trgt_id $trgt_id \
        --depth_vid $depth_vid \
        --ref_vid $ref_vid \
        --manual_depth_view $manual_depth_view \
        --pre_d_est $pre_d_est \
        --depth_occ $depth_occ \
        --manual_std_depth $manual_std_depth \
        --visual_items $visual_items \
        --appr_feature_str0 $appr_feature_str0 \
        --init_view_num $init_view_num \
        --feat_grad $feat_grad \
        --conf_grad $conf_grad \
        --dir_grad $dir_grad \
        --color_grad $color_grad \
        --depth_conf_thresh $depth_conf_thresh \
        --bgmodel $bgmodel \
        --vox_res $vox_res \
        --act_type $act_type \
        --geo_cnsst_num $geo_cnsst_num \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --normview $normview \
        --color_loss_weights $color_loss_weights \
        --zero_one_loss_items $zero_one_loss_items \
        --zero_one_loss_weights $zero_one_loss_weights \
        --default_conf $default_conf \
        --full_comb $full_comb \
        --prune_iter $prune_iter \
        --prune_thresh $prune_thresh \
        --uni_depth $uni_depth \
        --debug

