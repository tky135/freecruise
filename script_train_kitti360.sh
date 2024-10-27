#!/bin/bash

export PYTHONPATH=$(pwd)
# python tools/train.py --config_file configs/omnire_extended_cam.yaml --output_root tkyout --project test_proj --run_name 2_6cams dataset=nuscenes/6cams data.scene_idx=2 data.start_timestep=0 data.end_timestep=-1

python tools/train.py --config_file configs/omnire_neus_ts.yaml --output_root tkyout --project kitti360 --run_name exp6_lidar_loss dataset=kitti360/2cams data.scene_idx=999 data.start_timestep=0 data.end_timestep=-1


