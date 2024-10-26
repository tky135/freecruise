import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pykitti
from PIL import Image

from datasets.kitti.trackletparser import parseXML
from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.geometry import get_corners, project_camera_points_to_image
from utils.visualization import color_mapper, dump_3d_bbox_on_image
import shutil

KITTI_LABELS = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

KITTI_NONRIGID_DYNAMIC_CLASSES = [
    'Pedestrian', 'Person_sitting', 'Cyclist'
]

KITTI_RIGID_DYNAMIC_CLASSES = [
    'Car', 'Van', 'Truck', 'Tram'
]

KITTI_DYNAMIC_CLASSES = KITTI_NONRIGID_DYNAMIC_CLASSES + KITTI_RIGID_DYNAMIC_CLASSES

class Kitti360Processor(object):
    """Process KITTI dataset."""
    
    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        process_keys: List[str] = [
            "images",
            "lidar",
            "calib",
            "dynamic_masks",
            "objects"
        ],
        prefix: str = "2011_09_26",
        process_id_list: List[str] = None,
        workers: int = 64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        # self.HW = (375, 1242)
        print("will process keys: ", self.process_keys)

        self.cam_list = [
            "CAM_LEFT",     # "xxx_0.jpg"
            "CAM_RIGHT"     # "xxx_1.jpg"
        ]

        self.split_dir = os.path.join(load_dir, prefix) # load_dir: data/kitti360/raw
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")
        
    def get_kitti_data(self, basedir):
        date = basedir.split("/")[-2]
        drive = basedir.split("/")[-1].split('_')[-2]
        data = pykitti.raw(self.load_dir, date, drive)
        return data

    def convert_one(self, scene_name: str):
        """Convert action for single file."""
        # basedir = os.path.join(self.split_dir, scene_name)
        split_file = np.loadtxt(os.path.join(self.split_dir, "train_" + f'{scene_name:02d}.txt'), dtype=str).tolist()
        self.train_split = split_file
        self.test_split = []#np.loadtxt(os.path.join(self.split_dir, "test_" + f'{scene_name:02d}.txt'), dtype=str).tolist()
        # test_split_file = np.loadtxt(os.path.join(self.split_dir, "test_" + f'{scene_name:02d}.txt'), dtype=str).tolist()
        # split_file = train_split_file + test_split_file
        if "images" in self.process_keys:
            self.save_image(split_file, scene_name)
        if "calib" in self.process_keys:
            self.save_calib(scene_name)
        if "pose" in self.process_keys:
            self.save_pose(split_file, scene_name)
        if "lidar" in self.process_keys:
            self.save_lidar(split_file, scene_name)
            
        # tracklet_file = os.path.join(basedir, 'tracklet_labels.xml')
        # tracklets = parseXML(tracklet_file)
        # if "dynamic_masks" in self.process_keys:
        #     self.save_dynamic_mask(kitti_data, tracklets, scene_name, class_valid='all')
        #     self.save_dynamic_mask(kitti_data, tracklets, scene_name, class_valid='human')
        #     self.save_dynamic_mask(kitti_data, tracklets, scene_name, class_valid='vehicle')
        
        # # process annotated objects
        # if "objects" in self.process_keys:
        #     instances_info, frame_instances = self.save_objects(kitti_data, tracklets)
            
        #     # Save instances info and frame instances
        #     instances_info_save_path = f"{self.save_dir}/{str(scene_name).zfill(3)}/instances"
        #     os.makedirs(instances_info_save_path, exist_ok=True)
        #     with open(f"{instances_info_save_path}/instances_info.json", "w") as fp:
        #         json.dump(instances_info, fp, indent=4)
        #     with open(f"{instances_info_save_path}/frame_instances.json", "w") as fp:
        #         json.dump(frame_instances, fp, indent=4)
                
        #     if "objects_vis" in self.process_keys:
        #         objects_vis_path = f"{self.save_dir}/{str(scene_name).zfill(3)}/instances/debug_vis"
        #         if not os.path.exists(objects_vis_path):
        #             os.makedirs(objects_vis_path)
        #         self.visualize_dynamic_objects(kitti_data, scene_name, objects_vis_path, instances_info, frame_instances)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list) if self.process_id_list else 1

    def save_image(self, split_files, scene_name: str):
        for path in self.train_split:
            image_0 = Image.open(os.path.join(self.split_dir, path))
            image_1 = Image.open(os.path.join(self.split_dir, path.replace("image_00", "image_01")))
            frame_idx = int(path.split("/")[-1].split(".")[0])
            # Save left and right images
            image_0.save(f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_0.jpg")
            image_1.save(f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_1.jpg")
        for path in self.test_split:
            frame_idx = int(path.split("/")[-1].split(".")[0])
            image_0 = np.ones((375, 1242, 3), dtype=np.uint8) * 255
            image_1 = np.ones((375, 1242, 3), dtype=np.uint8) * 255
            # Save left and right images
            cv2.imwrite(f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_0.jpg", image_0)
            cv2.imwrite(f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_1.jpg", image_1)
    def save_calib(self, scene_name: str):
        # NOTE: assume ego vehicle is the same as velodyne frame 

        cam_00_to_velo = np.array([0.04307104361, -0.08829286498, 0.995162929, 0.8043914418, -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574, -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824]).reshape(3, 4)
        cam_00_to_velo = np.vstack([cam_00_to_velo, [0, 0, 0, 1]])
        
        R_rect_00 = np.array([0.999974, -0.007141, -0.000089, 0.007141, 0.999969, -0.003247, 0.000112, 0.003247, 0.999995]).reshape(3, 3)
        R_rect_01 = np.array([0.999778, -0.012115, 0.017222, 0.012059, 0.999922, 0.003351, -0.017261, -0.003143, 0.999846]).reshape(3, 3)
        R_rect_00_4x4 = np.eye(4)
        R_rect_00_4x4[:3, :3] = R_rect_00
        R_rect_01_4x4 = np.eye(4)
        R_rect_01_4x4[:3, :3] = R_rect_01
        
        cam_00_to_imu = np.array([0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039, 0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093, 0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000]).reshape(3, 4)
        cam_00_to_imu = np.vstack([cam_00_to_imu, [0, 0, 0, 1]])
        
        cam_01_to_imu = np.array([0.0194000864, -0.1051529641, 0.9942668106, 1.5977241400, 0.9997374956, -0.0100836652, -0.0205732716, 0.5981494900, 0.0121891942, 0.9944049345, 0.1049297370, -0.6488433108]).reshape(3, 4)
        cam_01_to_imu = np.vstack([cam_01_to_imu, [0, 0, 0, 1]])

        cam_01_to_velo = cam_00_to_velo @ np.linalg.inv(cam_00_to_imu) @ cam_01_to_imu
        np.savetxt(
            f"{self.save_dir}/{scene_name}/extrinsics/0.txt",
            cam_00_to_velo @ np.linalg.inv(R_rect_00_4x4)
        )
        np.savetxt(
            f"{self.save_dir}/{scene_name}/extrinsics/1.txt",
            cam_01_to_velo @ np.linalg.inv(R_rect_01_4x4)
        )
        
        cam2_Ks = np.array([552.554261, 0.000000, 682.049453, 0.000000, 0.000000, 552.554261, 238.769549, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]).reshape(3, 4)[:3, :3]
        cam3_Ks = np.array([552.554261, 0.000000, 682.049453, -328.318735, 0.000000, 552.554261, 238.769549, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]).reshape(3, 4)[:3, :3]
        # fx, fy, cx, cy, p1, p2, k1, k2, k3
        Ks_left = np.array([cam2_Ks[0, 0], cam2_Ks[1, 1], cam2_Ks[0, 2], cam2_Ks[1, 2], 0, 0, 0, 0, 0])
        Ks_right = np.array([cam3_Ks[0, 0], cam3_Ks[1, 1], cam3_Ks[0, 2], cam3_Ks[1, 2], 0, 0, 0, 0, 0])
        np.savetxt(
            f"{self.save_dir}/{scene_name}/intrinsics/0.txt",
            Ks_left
        )
        np.savetxt(
            f"{self.save_dir}/{scene_name}/intrinsics/1.txt",
            Ks_right
        )

    def save_pose(self, split_file, scene_name: str):
        # NOTE: we assume the ego pose is the same as the velodyne pose
        split_file = self.train_split + self.test_split
        ids = [int(path.split("/")[-1].split(".")[0]) for path in split_file]
        
        imu2w_item = np.loadtxt(os.path.join(self.split_dir, split_file[0].split("/")[0], "poses.txt"))[:, 1:].reshape(-1, 3, 4)
        imu2w_item = np.concatenate([imu2w_item, np.array([[[0, 0, 0, 1]]]).reshape(1, 1, 4).repeat(len(imu2w_item), axis=0)], axis=1)
        imu2w_key = np.loadtxt(os.path.join(self.split_dir, split_file[0].split("/")[0], "poses.txt"))[:, 0].astype(int)

        imu2w = {}
        for i in range(len(imu2w_key)):
            imu2w[imu2w_key[i]] = imu2w_item[i]
        # cam0_to_world = np.concatenate([cam0_to_world, np.array([[0, 0, 0, 1]]).reshape(1, 1, 4).repeat(len(cam0_to_world), axis=0)], axis=1)
        
        cam02velo = np.array([0.04307104361, -0.08829286498, 0.995162929, 0.8043914418, -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574, -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824]).reshape(3, 4)
        cam02velo = np.vstack([cam02velo, [0, 0, 0, 1]])
        velo2cam0 = np.linalg.inv(cam02velo)
        
        
        cam02imu = np.array([0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039, 0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093, 0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000]).reshape(3, 4)
        cam02imu = np.vstack([cam02imu, [0, 0, 0, 1]])
        
        # R_rect_00 = np.array([0.999974, -0.007141, -0.000089, 0.007141, 0.999969, -0.003247, 0.000112, 0.003247, 0.999995]).reshape(3, 3)
        # R_rect_01 = np.array([0.999778, -0.012115, 0.017222, 0.012059, 0.999922, 0.003351, -0.017261, -0.003143, 0.999846]).reshape(3, 3)
        # R_rect_00_4x4 = np.eye(4)
        # R_rect_00_4x4[:3, :3] = R_rect_00
        # R_rect_01_4x4 = np.eye(4)
        # R_rect_01_4x4[:3, :3] = R_rect_01
        
        
        rectcam02velo = np.loadtxt(f"{self.save_dir}/{scene_name}/extrinsics/0.txt")

        
        
        velo2imu = cam02imu @ velo2cam0
        
        for ids_idx in range(len(ids)):
            frame_idx = ids[ids_idx]
            velo2world = imu2w[frame_idx] @ velo2imu
            np.savetxt(
                f"{self.save_dir}/{scene_name}/ego_pose/{str(frame_idx).zfill(3)}.txt",
                velo2world
            )

    def save_lidar(self, split_file, scene_name: str):
        split_file = self.train_split + self.test_split
        frame_ids = [int(path.split("/")[-1].split(".")[0]) for path in split_file]
        for idx in range(len(frame_ids)):
            frame_idx = frame_ids[idx]
            lidar_load_path = os.path.join(self.split_dir, split_file[0].split('/')[0], "velodyne_points", "data", f"{frame_idx:010d}.bin")
            # Points are already in ego frame (velodyne frame), so we don't need to transform them
            # points = kitti_data.get_velo(frame_idx)
            
            # Save lidar points
            lidar_save_path = f"{self.save_dir}/{scene_name}/lidar/{str(frame_idx).zfill(3)}.bin"
            shutil.copyfile(lidar_load_path, lidar_save_path)
            # points.astype(np.float32).tofile(lidar_save_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for scene_name in id_list:
            os.makedirs(f"{self.save_dir}/{scene_name}/images", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/extrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/intrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/sky_masks", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_name}/lidar", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_name}/dynamic_masks/all", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_name}/dynamic_masks/human", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_name}/dynamic_masks/vehicle", exist_ok=True)