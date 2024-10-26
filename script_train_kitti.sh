export PYTHONPATH=.

python tools/train.py --config_file configs/omnire_neus_ts.yaml --output_root tkyout --project kitti --run_name new_kitti_009 dataset=kitti/kitti_mot_nvs_075 data.start_timestep=0 data.end_timestep=100

