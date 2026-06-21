# base --> outdim=256, small --> outdim=128,
# data_dirs必须是绝对路径！不然就会默认从data_components里开始拼相对路径，而不是根目录
CUDA_VISIBLE_DEVICES=1 python train.py \
  --domains=droid \
  --ptv3_size=base \
  --predictor_dim=256 \
  --data_dirs=/backup/zhangrong/data/workspace/robot-wm/point-wm/PointWorld/restore_data/pointworld_droid_subset_restored/droid/wds_8152clips \
  --norm_stats_path=stats/droid \
  --batch_size=22 \
  --eval_freq=300 \
  --save_freq=300 \
  --num_workers=16 \
  --eval_num_workers=5 \
  --exp_name=pointworld_pretrain_base_droid_8152clips_dino_full_layer_no_gripper \
  --robot_use_gripper_open_feature=false \
  --scene_use_dino=true \
  --scene_dino_layers=4,11,17,23