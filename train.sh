# base --> outdim=256, small --> outdim=128,
# data_dirs必须是绝对路径！不然就会默认从data_components里开始拼相对路径，而不是根目录
CUDA_VISIBLE_DEVICES=0 python train.py \
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
  --exp_name=pointworld_pretrain_droid_shard10_dino_11_23_layer \
  --robot_use_gripper_open_feature=true \
  --scene_use_dino=true \
  --scene_dino_layers=11,23