export PYTHONPATH=.
python datasets/preprocess.py --data_root data/kitti360/raw/ --dataset kitti360 --split 2013_05_28 --scene_ids 0 --target_dir data/kitti360/processed/ --workers 1 --process_keys images lidar pose calib dynamic_masks objects
