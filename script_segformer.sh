export PYTHONPATH=.
python datasets/tools/extract_masks.py \
    --data_root data/kitti360/processed \
    --segformer_path=SegFormer \
    --checkpoint=segformer.b5.1024x1024.city.160k.pth \
    --scene_ids 999 \
    --process_dynamic_mask
