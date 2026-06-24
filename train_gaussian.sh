# Gaussian branch example.
# data_dirs必须是绝对路径！不然就会默认从data_components里开始拼相对路径，而不是根目录

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
  --domains=droid \
  --ptv3_size=small \
  --predictor_dim=128 \
  --data_dirs=/home/zhangrong/zhangrong-workspace/robot-wm/point-wm/PointWorld/restore_data/pointworld_droid_subset_restored_test_shard10/droid/wds \
  --norm_stats_path=stats/droid \
  --batch_size=1 \
  --eval_freq=30 \
  --save_freq=30 \
  --num_workers=16 \
  --eval_num_workers=5 \
  --exp_name=pointworld_pretrain_droid_shard10_gaussian_with_5x5mask \
  --enable_gaussian_splatting=true \
  --gaussian_loss_weight=1.0 \
  --gaussian_ssim_weight=0.2 \
  --gaussian_use_projection_mask=true \
  --gaussian_renderer_backend=diff_gaussian \
  --gaussian_znear=0.01 \
  --gaussian_zfar=100.0 \
  --gaussian_min_render_depth=0.05 \
  --gaussian_max_screen_radius=64.0 \
  --gaussian_patch_radius=2 \
  --gaussian_init_scale=0.01 \
  --gaussian_min_scale=0.0001 \
  --gaussian_max_scale=0.05 \
  --gaussian_init_opacity=0.1 \
  --gaussian_delta_mu_max=0.03 \
  --gaussian_train_save_freq=30 \
  --gaussian_eval_save=true \
  --gaussian_save_max_images=16
