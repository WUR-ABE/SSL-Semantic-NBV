#!/usr/bin/env python3
from utils.paths import MATERIALS_DIR, real_plant_dir
from view_sampling.view_sampler import ViewSampler
from nbv_compute.nbv_computor import Nbvcomputor
import rospy
import time
import numpy as np
import random
import os
import torch
from geometry_msgs.msg import Pose
import cv2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class Environment_RW:
    def __init__(
        self,
        config,
    ):
        self.config = config

        assert config['experiment'] == 2, 'can only set experiment =2'

        self.data_save_dir = MATERIALS_DIR + "/%s" % (str(time.ctime()))

        # Demo mode (config['demo']): drive the simulated ABB arm to each view and publish the
        # images and point clouds for RViz. Needs the arm simulation running (abb_control) and
        # lifts everything by z_offset into the arm's workspace. Off for training and evaluation.
        self.demo = config.get('demo', False)
        if self.demo:
            from abb_control.arm_control_client import ArmControlClient

            self.arm_control = ArmControlClient()
        self.z_offset = 1.0 if self.demo else 0.0

        self.nbv_compute = Nbvcomputor(octomap_resulotion=config['octo_resolution'])
        self.scale_action_space = False

        self.bridge = CvBridge()
        self.rgb_pub = rospy.Publisher("rgb_image", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("mask_image", Image, queue_size=1)
        self.info_pub = rospy.Publisher("camera_info", CameraInfo, queue_size=1)

        self.big_plants = []
        self.small_plants = []

        self.experiment = config[
            'experiment'
        ]  ##0: normal plant; 1: heavily occluded plant; 2: real world

        self.view_range = {
            'num_col': 10,
            'num_row': 15,
        }

        self.x_range = [0]
        self.y_range = [0]
        self.z_range = [0]
        self.rotation_range = [0]

        self.train_plants = [31, 32, 34, 35, 37, 38, 40]  ##select the plants for training
        self.valid_plants = [33, 36, 39]

        self.all_plants = self.train_plants + self.valid_plants

        self.rois_bbox_pub = rospy.Publisher("rois_bbox_marker", Marker, queue_size=1)

        self.pc_size = config["pc_size"]

        self.set_seed()

    def get_viewpoints(self):
        viewpoints = np.load(
            os.path.join(real_plant_dir(self.plant), 'viewpoints.npy'), allow_pickle=True
        )
        viewpoints = np.array(viewpoints, dtype=np.float32)
        viewpoints = viewpoints[..., [0, 1, 2, 6, 3, 4, 5]]
        # Apply z_offset so viewpoints land in the arm's reachable workspace.
        # Raw npy z: 0.45–1.05 m.  Arm workspace z: 0.8–1.2 m (arm_control_test.py).
        # Adding 0.6 m shifts viewpoints to 1.05–1.65 m, within reach.
        viewpoints[..., 2] += self.z_offset
        return viewpoints

    def resample_pc(self, pcd, n):
        """
        Optimized version of point cloud resampling using voxel downsampling.

        Args:
            pcd (torch.Tensor): Input point cloud of shape (1, N, C)
            n (int): Target number of points

        Returns:
            torch.Tensor: Resampled point cloud of shape (1, n, C)
        """
        device = pcd.device
        points = pcd[0, :, :3]  # (N, 3)
        features = pcd[0, :, 3:]  # (N, 3)

        # Compute voxel size more efficiently
        bbox_min = points.min(dim=0).values  # torch.tensor([-0.8, -0.8, 0.0]).to(device)
        bbox_max = points.max(dim=0).values  # torch.tensor([0.8, 0.8, 2]).to(device)
        voxel_size = ((bbox_max - bbox_min).prod() / n) ** (1 / 3)

        # Faster voxel grid quantization
        voxel_indices = ((points - bbox_min) / voxel_size).long()

        # Convert to unique string key for faster unique operation
        voxel_keys = (
            voxel_indices[:, 0] * 100000000 + voxel_indices[:, 1] * 10000 + voxel_indices[:, 2]
        )
        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)

        # Use index_select for faster point selection
        if len(unique_keys) >= n:
            # If we have more voxels than needed, randomly select n voxels
            selected_voxels = unique_keys[torch.randperm(len(unique_keys), device=device)[:n]]
            mask = torch.zeros(len(voxel_keys), dtype=torch.bool, device=device)
            for voxel_id in selected_voxels:
                mask |= voxel_keys == voxel_id
            sampled_indices = mask.nonzero(as_tuple=True)[0][:n]
        else:
            # Take one point from each voxel
            sampled_indices = []
            for voxel_id in unique_keys:
                points_in_voxel = (voxel_keys == voxel_id).nonzero(as_tuple=True)[0]
                sampled_indices.append(points_in_voxel[0])
            sampled_indices = torch.tensor(sampled_indices, device=device)

            # Pad with random points if needed
            if len(sampled_indices) < n:
                remaining = n - len(sampled_indices)
                pad_indices = torch.randint(points.shape[0], (remaining,), device=device)
                sampled_indices = torch.cat((sampled_indices, pad_indices))

        # Gather points and features in one operation
        resampled_pcd = torch.cat(
            (points[sampled_indices], features[sampled_indices]), dim=-1
        ).unsqueeze(0)
        return resampled_pcd

    def update_mode(self, mode):
        self.mode = mode

    def get_pcd(self, plant, rotation, position: list, hw: list):
        pcd_path = os.path.join(
            real_plant_dir(plant), "pcd_sam3", "pcd_%s_%s.npy" % (str(hw[0]), str(hw[1]))
        )
        pcd = np.load(pcd_path)
        pcd[..., 2] += self.z_offset  # lift point clouds above collision objects
        return torch.tensor(pcd, dtype=torch.float32).cuda()  # , xyzrgbc

    def get_grid_info(self):  # need to check
        x_dim, y_dim, z_dim = 0.6, 0.6, 1
        grid_center = np.array([0.4, -0.8, 0.9])

        return np.array([x_dim, y_dim, z_dim]), grid_center

    def set_seed(self):
        random.seed(1004)

    def reset_global(self, planner, predefined_views=None):
        if planner == 'voxel_global_nbv':
            self.initial_bbox_location = True

        self.planner = planner

        if self.mode == "train":
            self.plant = random.choice(self.train_plants)
        else:
            self.plant = random.choice(self.valid_plants)

        self.viewpoints = self.get_viewpoints()

        self.max_step = self.config['max_step']

        self.rotation = random.choice(self.rotation_range)
        self.x, self.y, self.z = (
            random.choice(self.x_range),
            random.choice(self.y_range),
            random.choice(self.z_range),
        )

        self.viewpoints_1d = self.viewpoints.reshape(
            1, self.viewpoints.shape[0] * self.viewpoints.shape[1], 7
        )

        self.pcd_list = []
        for i in range(self.viewpoints.shape[0]):
            for j in range(self.viewpoints.shape[1]):
                pcd_part = self.get_pcd(self.plant, self.rotation, [self.x, self.y, self.z], [i, j])
                self.pcd_list.append(pcd_part)

        self.viewstate = -1 * torch.ones(1, self.viewpoints_1d.shape[1], dtype=torch.float32).cuda()
        self.nbv_compute.load_complete_pc(
            'plant',
            self.plant,
            self.config['recon_target'],
            plant_rotation=self.rotation,
            plant_position=[self.x, self.y, self.z + self.z_offset * 100],
            scale=1,
            visual_com=self.demo,  # publish complete pc to /camera/complete_point_cloud
            visual_roi=self.demo,  # publish ROI bounding boxes
        )

        valid_init_view = False
        self.step_count = 0
        while not valid_init_view:
            if predefined_views is None:
                current_view_index = random.randint(0, self.viewpoints_1d.shape[1] - 1)
            else:
                current_view_index = random.choice(predefined_views)

            pcd_acc = self.pcd_list[current_view_index]

            if pcd_acc.shape[1] > 10:
                valid_init_view = True

        # classify so pcd_acc has the semantic class channel (xyzrgbc)
        self.pcd_acc = self.nbv_compute.classify_points_use_bbox(
            pcd_acc, self.config['recon_target']
        )

        current_hw = np.unravel_index(
            current_view_index, (self.viewpoints.shape[0], self.viewpoints.shape[1])
        )

        # move the robot arm to the initial viewpoint
        if self.demo:
            success = self.arm_control.move_arm_to_pose(
                self.create_ros_pose(self.viewpoints[current_hw[0], current_hw[1], :])
            )
            if success:
                self.publish_images(self.plant, self.rotation, [self.x, self.y, self.z], current_hw)
                rospy.sleep(2.0)

        self.viewstate[0, current_view_index] = 1.0
        acc_pc_resampled = self.resample_pc(self.pcd_acc.clone(), self.pc_size).permute(0, 2, 1)

        return (
            acc_pc_resampled,
            self.viewstate.clone(),
            current_view_index,
            current_hw,
        )  # acc_pc_weighted.shape = (1,3,n)

    def step_global(self, nbv_index):
        assert self.step_count < self.max_step, "StepCount exceeded MaxStep=%s" % str(self.max_step)
        nbv_hw = np.unravel_index(nbv_index, (self.viewpoints.shape[0], self.viewpoints.shape[1]))

        # # Check if the view has already been visited
        if self.viewstate[0, nbv_index] == 1.0:
            print('repeat view')
        # generate a vector with all None values
        gt_ig_list = torch.full((1, self.viewstate.shape[-1]), torch.nan).cuda()
        gt_ig_list[0, self.viewstate[0, :] == 1.0] = 0.0

        pcd_acc = self.pcd_acc.clone()

        pcd_nbv = self.pcd_list[nbv_index]
        pcd_nbv = self.nbv_compute.classify_points_use_bbox(pcd_nbv, self.config['recon_target'])
        self.pcd_acc, _, _, gt_ig = self.nbv_compute.combine_pc(
            pcd_nbv, pcd_acc, self.planner, resaign_color=True, visual_pc=self.demo
        )

        gt_ig_list[0, nbv_index] = gt_ig
        self.viewstate[0, nbv_index] = 1.0

        # move the robot arm to the NBV viewpoint
        if self.demo:
            success = self.arm_control.move_arm_to_pose(
                self.create_ros_pose(self.viewpoints[nbv_hw[0], nbv_hw[1], :])
            )
            if success:
                self.publish_images(self.plant, self.rotation, [self.x, self.y, self.z], nbv_hw)
                rospy.sleep(5.0)

        # Resample the accumulated point cloud
        acc_pc_resampled = self.resample_pc(self.pcd_acc.clone(), self.pc_size).permute(0, 2, 1)

        self.step_count += 1
        done = self.step_count >= self.max_step

        if self.demo:
            # brief pause so RViz can render the updated point cloud
            rospy.sleep(0.5)

        return acc_pc_resampled, self.viewstate.clone(), gt_ig_list.cpu(), done, nbv_hw

    def create_ros_pose(self, vp):
        """Convert viewpoint [x, y, z, qw, qx, qy, qz] to geometry_msgs/Pose.

        We use a fixed orientation pointing towards world -Y (robot forward)
        to ensure MoveIt planning succeeds while facing the plant.
        Apply the same y-offset (-0.2) as the marker arrows in create_demo_rw_sam3.py.
        """
        pose = Pose()
        pose.position.x = float(vp[0])
        pose.position.y = float(vp[1]) - 0.2  # same y-shift as marker arrows
        pose.position.z = float(vp[2])
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = -0.7071
        pose.orientation.w = 0.7071
        return pose

    def publish_images(self, plant, rotation, position, hw):
        try:
            # Construct paths
            # Optional images, only for display in RViz
            plant_dir = real_plant_dir(plant)
            rgb_path = os.path.join(
                plant_dir, "raw", "rgb", "pcd_%s_%s_pcd_rgb.png" % (str(hw[0]), str(hw[1]))
            )

            mask_path = os.path.join(
                plant_dir, "raw", "masks", "pcd_%s_%s_pcd_rgb_debug.png" % (str(hw[0]), str(hw[1]))
            )

            # Load and publish RGB
            if os.path.exists(rgb_path):
                rgb_img = cv2.imread(rgb_path)
                if rgb_img is not None:
                    self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img, "bgr8"))

            # Load and publish Mask
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path)
                if mask_img is not None:
                    self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_img, "bgr8"))

            # Publish CameraInfo so RViz 'Camera' display works
            info = CameraInfo()
            info.header.stamp = rospy.Time.now()
            info.header.frame_id = "camera_color_optical_frame"
            # Standard L515 720p defaults (can be adjusted if known)
            info.width = 1280
            info.height = 720
            info.distortion_model = "plumb_bob"
            info.D = [0, 0, 0, 0, 0]
            info.K = [900, 0, 640, 0, 900, 360, 0, 0, 1]
            info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            info.P = [900, 0, 640, 0, 0, 900, 360, 0, 0, 0, 1, 0]
            self.info_pub.publish(info)
        except Exception as e:
            print(f"[Environment_RW] Error publishing images: {e}")

    def _viewpoint_to_pose(self, vp):
        """Alias kept for compatibility."""
        return self.create_ros_pose(vp)

    def get_nbv_pc(self, action: np.ndarray, current_hw):
        _, next_hw, _ = ViewSampler.update_view(self.viewpoints, current_hw, action)
        pcd_next = self.get_state(self.plant, self.rotation, [self.x, self.y], next_hw)
        return pcd_next

    def get_inbound_pcds(self, current_hw):
        in_bound_actions = []
        in_bound_pcds = []
        for action in range(self.action_space):
            _, next_hw, in_bound = ViewSampler.update_view(self.viewpoints, current_hw, action)
            if in_bound:
                pcd_next = self.get_pcd(self.plant, self.rotation, [self.x, self.y], next_hw)
                in_bound_actions.append(action)
                in_bound_pcds.append(pcd_next)
        return in_bound_actions, in_bound_pcds

    def update_roi_bbox(self, node_points):
        grid_size, grid_center = self.get_grid_info()

        if self.config['recon_target'] == 'PLANT':
            dim = 0.6
        elif self.config['recon_target'] == 'FRUIT':
            dim = 0.6
        elif self.config['recon_target'] == 'NODE':
            dim = 0.1

        if self.initial_bbox_location:
            if node_points.shape[0] > 5:
                x_center, y_center, z_center = node_points.mean(axis=0)
                x_min, y_min, z_min = (
                    x_center - grid_size[0] / 2,
                    y_center - grid_size[1] / 2,
                    z_center - grid_size[2] / 2,
                )
                x_max, y_max, z_max = (
                    x_center + grid_size[0] / 2,
                    y_center + grid_size[1] / 2,
                    z_center + grid_size[2] / 2,
                )

                self.node_bbox = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
                self.initial_bbox_location = False
            else:
                self.node_bbox = np.array(
                    [
                        grid_center[0] - grid_size[0] / 2,
                        grid_center[1] - grid_size[1] / 2,
                        grid_center[2] - grid_size[2] / 2,
                        grid_center[0] + grid_size[0] / 2,
                        grid_center[1] + grid_size[1] / 2,
                        grid_center[2] + grid_size[2] / 2,
                    ]
                )
        else:
            x_center, y_center, z_center = node_points.mean(axis=0)
            x_min, y_min, z_min = (
                x_center - grid_size[0] / 2,
                y_center - grid_size[1] / 2,
                z_center - grid_size[2] / 2,
            )
            x_max, y_max, z_max = (
                x_center + grid_size[0] / 2,
                y_center + grid_size[1] / 2,
                z_center + grid_size[2] / 2,
            )

            self.node_bbox = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

        # check the node_bbox, if it outside the grid, crop it to the grid
        self.node_bbox[0] = (
            grid_center[0] - grid_size[0] / 2
            if self.node_bbox[0] < grid_center[0] - grid_size[0] / 2
            else self.node_bbox[0]
        )
        self.node_bbox[1] = (
            grid_center[1] - grid_size[1] / 2
            if self.node_bbox[1] < grid_center[1] - grid_size[1] / 2
            else self.node_bbox[1]
        )
        self.node_bbox[2] = (
            grid_center[2] - grid_size[2] / 2
            if self.node_bbox[2] < grid_center[2] - grid_size[2] / 2
            else self.node_bbox[2]
        )
        self.node_bbox[3] = (
            grid_center[0] + grid_size[0] / 2
            if self.node_bbox[3] > grid_center[0] + grid_size[0] / 2
            else self.node_bbox[3]
        )
        self.node_bbox[4] = (
            grid_center[1] + grid_size[1] / 2
            if self.node_bbox[4] > grid_center[1] + grid_size[1] / 2
            else self.node_bbox[4]
        )
        self.node_bbox[5] = (
            grid_center[2] + grid_size[2] / 2
            if self.node_bbox[5] > grid_center[2] + grid_size[2] / 2
            else self.node_bbox[5]
        )
        self.visualize_roi_bbox()
        node_bbox = self.node_bbox.copy()
        return node_bbox

    def visualize_roi_bbox(self):
        position = [
            (self.node_bbox[0] + self.node_bbox[3]) / 2,
            (self.node_bbox[1] + self.node_bbox[4]) / 2,
            (self.node_bbox[2] + self.node_bbox[5]) / 2,
        ]
        size = [
            self.node_bbox[3] - self.node_bbox[0],
            self.node_bbox[4] - self.node_bbox[1],
            self.node_bbox[5] - self.node_bbox[2],
        ]

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = Marker.CUBE
        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.scale.x = size[0]
        marker.scale.y = size[1]
        marker.scale.z = size[2]
        # Set the pose of the marker
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.rois_bbox_pub.publish(marker)
