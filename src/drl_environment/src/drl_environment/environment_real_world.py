#!/usr/bin/env python3
from view_sampling.view_sampler import ViewSampler
from nbv_compute.nbv_computor import Nbvcomputor
import rospy
import time
import numpy as np
import random
import os
import torch
# from camera_control.camera_control import CameraControl
# from camera_control.camera_pose_broadcaster import CameraPoseBroadcaster
from visualization_msgs.msg import Marker

class Environment_RW:
    def __init__(
        self, config,
    ):
        self.config = config

        assert config['experiment']==2, 'can only set experiment =2'

        self.data_save_dir = (
            "/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/%s"
            % (str(time.ctime()))
        )

        # self.camera_control = CameraControl()
        # self.camera_pose_broadcast = CameraPoseBroadcaster()

        self.nbv_compute = Nbvcomputor(octomap_resulotion = config['octo_resolution'])
        self.scale_action_space = False
        
        self.big_plants = []
        self.small_plants = []

        self.experiment = config['experiment']##0: normal plant; 1: heavily occluded plant; 2: real world

        self.nbv_data_path = '/home/jianchao/dataset/nbv_data/paper4/data_smaller_fov_distance' 
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
        viewpoints = np.load(os.path.join(self.nbv_data_path, 'viewpoints_%s.npy'%str(self.plant)), allow_pickle=True)
        viewpoints = np.array(viewpoints, dtype=np.float32)
        viewpoints = viewpoints[...,[0,1,2,6,3,4,5]]
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
        points = pcd[0, :, :3] #(N, 3)
        features = pcd[0, :, 3:] #(N, 3)
        
        # Compute voxel size more efficiently
        bbox_min = points.min(dim=0).values#torch.tensor([-0.8, -0.8, 0.0]).to(device)
        bbox_max = points.max(dim=0).values#torch.tensor([0.8, 0.8, 2]).to(device)
        voxel_size = ((bbox_max - bbox_min).prod() / n) ** (1/3)
        
        # Faster voxel grid quantization
        voxel_indices = ((points - bbox_min) / voxel_size).long()
        
        # Convert to unique string key for faster unique operation
        voxel_keys = voxel_indices[:, 0] * 100000000 + voxel_indices[:, 1] * 10000 + voxel_indices[:, 2]
        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
        
        # Use index_select for faster point selection
        if len(unique_keys) >= n:
            # If we have more voxels than needed, randomly select n voxels
            selected_voxels = unique_keys[torch.randperm(len(unique_keys), device=device)[:n]]
            mask = torch.zeros(len(voxel_keys), dtype=torch.bool, device=device)
            for voxel_id in selected_voxels:
                mask |= (voxel_keys == voxel_id)
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
        resampled_pcd = torch.cat((points[sampled_indices], features[sampled_indices]), dim=-1).unsqueeze(0)
        # resampled_pcd = self.add_noise(resampled_pcd, noisy_level='complete')
        return resampled_pcd
    
    def update_mode(self, mode):
        self.mode = mode

    def get_pcd(self, plant, rotation, position: list, hw: list):
        pcd_path = os.path.join(
            self.nbv_data_path,
            "plant%s" % str(plant),
            "xyzr_%s_%s_%s_%s" % (str(position[0]), str(position[1]), str(position[2]), str(rotation)),
            'pcd',
            "pcd_%s_%s.npy" % (str(hw[0]), str(hw[1])),
        )
        pcd = np.load(pcd_path)
        return torch.tensor(pcd, dtype=torch.float32).cuda()  # , xyzrgb
    
    
    def get_grid_info(self): # need to check
        x_dim,y_dim,z_dim = 0.6, 0.6, 1
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
        self.x, self.y, self.z = random.choice(self.x_range), random.choice(self.y_range), random.choice(self.z_range)

        self.viewpoints_1d = self.viewpoints.reshape(1, self.viewpoints.shape[0]*self.viewpoints.shape[1], 7)

        self.pcd_list = []
        for i in range(self.viewpoints.shape[0]):
            for j in range(self.viewpoints.shape[1]):
                pcd_part = self.get_pcd(
                self.plant, self.rotation, [self.x, self.y, self.z], [i,j]
            )
                self.pcd_list.append(pcd_part)

        self.viewstate = -1*torch.ones(1, self.viewpoints_1d.shape[1], dtype=torch.float32).cuda()
        self.nbv_compute.load_complete_pc(
            'plant',
            self.plant,
            self.config['recon_target'],
            plant_rotation=self.rotation,
            plant_position=[self.x, self.y, self.z],
            scale = 1,
            visual_com = False,
            visual_roi = False
        )

        valid_init_view = False
        self.step_count = 0
        while not valid_init_view:
            if predefined_views is None:
                current_view_index = random.randint(0, self.viewpoints_1d.shape[1]-1)
            else:
                current_view_index = random.choice(predefined_views)

            pcd_acc = self.pcd_list[current_view_index]

            if pcd_acc.shape[1] > 10:
                valid_init_view = True

        self.pcd_acc = self.nbv_compute.classify_points_use_bbox(pcd_acc, self.config['recon_target']) # self.pcd_acc.shape = (1,n, 4)

        current_hw = np.unravel_index(current_view_index, (self.viewpoints.shape[0], self.viewpoints.shape[1]))

        self.viewstate[0, current_view_index] = 1.0
        acc_pc_resampled = self.resample_pc(self.pcd_acc.clone(), self.pc_size).permute(0, 2, 1)

        return (
            acc_pc_resampled,
            self.viewstate.clone(),
            current_view_index,
            current_hw,
        ) 

    def step_global(self, nbv_index):
        assert self.step_count < self.max_step, "StepCount exceeded MaxStep=%s" % str(self.max_step)
        nbv_hw = np.unravel_index(nbv_index, (self.viewpoints.shape[0], self.viewpoints.shape[1]))

        if self.viewstate[0, nbv_index] == 1.0:
            print('repeat view')

        gt_ig_list = torch.full((1, self.viewstate.shape[-1]), torch.nan).cuda()
        gt_ig_list[0, self.viewstate[0, :] == 1.0] = 0.0

        pcd_acc = self.pcd_acc.clone()

        pcd_nbv = self.pcd_list[nbv_index]
        pcd_nbv = self.nbv_compute.classify_points_use_bbox(pcd_nbv, self.config['recon_target'])
        self.pcd_acc, _, _, gt_ig = self.nbv_compute.combine_pc(pcd_nbv, pcd_acc, self.planner, visual_pc=False)

        gt_ig_list[0, nbv_index] = gt_ig
        self.viewstate[0, nbv_index] = 1.0

        acc_pc_resampled = self.resample_pc(self.pcd_acc.clone(), self.pc_size).permute(0, 2, 1)

        self.step_count += 1
        done = self.step_count >= self.max_step
        return acc_pc_resampled, self.viewstate.clone(), gt_ig_list.cpu(), done, nbv_hw
    
    def get_nbv_pc(self, action: np.ndarray, current_hw):
        _, next_hw, _ = ViewSampler.update_view(
            self.viewpoints, current_hw, action
        )
        pcd_next = self.get_state(self.plant, self.rotation, [self.x, self.y], next_hw)
        return pcd_next

    def get_inbound_pcds(self, current_hw):
        in_bound_actions = []
        in_bound_pcds = []
        for action in range(self.action_space):
            _, next_hw, in_bound = ViewSampler.update_view(
                self.viewpoints, current_hw, action
            )
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
                x_min, y_min, z_min = x_center-grid_size[0]/2, y_center-grid_size[1]/2, z_center-grid_size[2]/2
                x_max, y_max, z_max = x_center+grid_size[0]/2, y_center+grid_size[1]/2, z_center+grid_size[2]/2

                self.node_bbox = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
                self.initial_bbox_location = False
            else:
                self.node_bbox = np.array([grid_center[0]-grid_size[0]/2, grid_center[1]-grid_size[1]/2, grid_center[2]-grid_size[2]/2, grid_center[0]+grid_size[0]/2, grid_center[1]+grid_size[1]/2, grid_center[2]+grid_size[2]/2])
        else:
            x_center, y_center, z_center = node_points.mean(axis=0)
            x_min, y_min, z_min = x_center-grid_size[0]/2, y_center-grid_size[1]/2, z_center-grid_size[2]/2
            x_max, y_max, z_max = x_center+grid_size[0]/2, y_center+grid_size[1]/2, z_center+grid_size[2]/2
            
            self.node_bbox = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

        # check the node_bbox, if it outside the grid, crop it to the grid
        self.node_bbox[0] = grid_center[0]-grid_size[0]/2 if self.node_bbox[0] < grid_center[0]-grid_size[0]/2 else self.node_bbox[0]
        self.node_bbox[1] = grid_center[1]-grid_size[1]/2 if self.node_bbox[1] < grid_center[1]-grid_size[1]/2 else self.node_bbox[1]
        self.node_bbox[2] = grid_center[2]-grid_size[2]/2 if self.node_bbox[2] < grid_center[2]-grid_size[2]/2 else self.node_bbox[2]
        self.node_bbox[3] = grid_center[0]+grid_size[0]/2 if self.node_bbox[3] > grid_center[0]+grid_size[0]/2 else self.node_bbox[3]
        self.node_bbox[4] = grid_center[1]+grid_size[1]/2 if self.node_bbox[4] > grid_center[1]+grid_size[1]/2 else self.node_bbox[4]
        self.node_bbox[5] = grid_center[2]+grid_size[2]/2 if self.node_bbox[5] > grid_center[2]+grid_size[2]/2 else self.node_bbox[5]
        self.visualize_roi_bbox()   
        node_bbox = self.node_bbox.copy()
        return node_bbox
    
    def visualize_roi_bbox(self):
        position = [(self.node_bbox[0]+self.node_bbox[3])/2, (self.node_bbox[1]+self.node_bbox[4])/2, (self.node_bbox[2]+self.node_bbox[5])/2]
        size = [self.node_bbox[3]-self.node_bbox[0], self.node_bbox[4]-self.node_bbox[1], self.node_bbox[5]-self.node_bbox[2]]

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


