#!/usr/bin/env python3

import torch
import numpy as np
import rospy
import ros_numpy
import tf2_ros
import os
import open3d as o3d
from networks.chamfer import chamfer_distance
import std_msgs.msg as std_msgs
# import PyKDL
from sensor_msgs.msg import PointField
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from scipy.spatial.transform import Rotation as R
# import copy
# import time
from torch_scatter import scatter_mean
from visualization_msgs.msg import Marker, MarkerArray
import pandas as pd
import time

class Nbvcomputor:
    ref_frame = "world"
    max_depth = 1.5
    def __init__(self, octomap_resulotion, ref_frame=ref_frame, max_depth=max_depth):
        # Parameters.
        self.ref_frame = ref_frame
        self.octomap_resulotion = octomap_resulotion
        self.max_depth = max_depth

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # self.point_cloud_stream = None
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.previous_ids = set()  # Track previously published marker IDs

        self.complete_plant_dir = '/home/jianchao/dataset/nbv_data/paper4/gt_pcd'
        self.roi_path = '/home/jianchao/dataset/nbv_data/paper4/gt_nodes'

        rospy.Subscriber("/camera/color/image_rect_color",
                         Image,
                         self.color_callback,
                         queue_size=2)
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info",
            CameraInfo,
            self.info_callback,
            queue_size=2,
        )
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=2,
        )
        
        self.current_pc_publisher = rospy.Publisher(
            "/camera/current_point_cloud", PointCloud2, queue_size=1)
        
        self.complete_pc_publisher = rospy.Publisher(
            "/camera/complete_point_cloud", PointCloud2, queue_size=1)
        
        self.acc_pc_publisher = rospy.Publisher(
            "/camera/acc_point_cloud", PointCloud2, queue_size=1)
        
        self.rois_pub = rospy.Publisher("rois_marker", MarkerArray, queue_size=1)

    def color_callback(self, msg):
        # Subscriber callback for color image.
        data = ros_numpy.numpify(msg)
        self.color_image = data[:, :, ::-1]

    def depth_callback(self, msg):
        # Subscriber callback for depth image.
        data = ros_numpy.numpify(msg)
        self.depth_image = data.astype('float32')# /1000 in real world

    def info_callback(self, msg):
        # Subscriber callback for camera info.
        self.camera_info = msg

    def generate_cloud_data_common(self, bgr_img_, depth_img_):
        """
        Do depth registration, suppose that rgb_img and depth_img has the same intrinsic
        \param bgr_img (numpy array bgr8)
        \param depth_img (numpy array float32 2d)
        [x, y, Z] = [X, Y, Z] * intrinsic.T
        """
        bgr_img = bgr_img_.copy()
        depth_img = depth_img_.copy()
        
        width, height = self.camera_info.width, self.camera_info.height
        x_index = np.array([[i for i in range(width)] * height], dtype="<f4")
        y_index = np.array([[i] * width for i in range(height)],
                           dtype="<f4").ravel()

        xy_index = np.vstack((x_index, y_index)).T  # x,y

        bgr0_vect = np.zeros([width * height, 4], dtype="<f4")  # bgr0
        xyd_vect = np.zeros([width * height, 3], dtype="<f4")  # x,y,depth
        XYZ_vect = np.zeros([width * height, 3],
                            dtype="<f4")  # real world coord
        ros_data = np.ones(
            [width * height, 8], dtype="<f4"
        )  # [x,y,z,0,bgr0,0,0,0] or [x,y,z,0,bgr0,semantics,confidence,0]

        bgr_img = bgr_img.view("<u1")
        # Bring nan/inf points to max_range

        depth_img[np.isnan(depth_img)] = (self.max_depth)
        depth_img = depth_img.view("<f4")

        # Camera intrinsic matrix
        fx = self.camera_info.K[0]  # fx
        fy = self.camera_info.K[4]  # fy
        cx = self.camera_info.K[2]  # cx
        cy = self.camera_info.K[5]  # cy
        intrinsic = np.matrix([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                              dtype=np.float32)
        # Add depth information

        # print(xy_index.shape, depth_img.reshape(-1, 1).shape)
        xyd_vect[:, 0:2] = xy_index * depth_img.reshape(-1, 1)
        xyd_vect[:, 2:3] = depth_img.reshape(-1, 1)
        XYZ_vect = xyd_vect.dot(intrinsic.I.T)
        # Convert to ROS point cloud message in a vectorialized manner
        # ros msg data: [x,y,z,0,bgr0,0,0,0,color0,color1,color2,0,confidence0,confidence1,confidenc2,0] (little endian float32)
        # Transform color
        bgr0_vect[:, 0:1] = bgr_img[:, :, 2].reshape(-1, 1)
        bgr0_vect[:, 1:2] = bgr_img[:, :, 1].reshape(-1, 1)
        bgr0_vect[:, 2:3] = bgr_img[:, :, 0].reshape(-1, 1)
        # Concatenate data
        ros_data[:, 0:3] = XYZ_vect
        ros_data[:, 3:7] = bgr0_vect/255.0
        return ros_data
    
    # def generate_cloud_data_common(self, bgr_img, depth_img):
    #     """
    #     NEED TO TEST
    #     Do depth registration, assuming that rgb_img and depth_img have the same intrinsic
    #     \param bgr_img (numpy array bgr8)
    #     \param depth_img (numpy array float32 2d)
    #     [x, y, Z] = [X, Y, Z] * intrinsic.T
    #     """
    #     # Get image width and height from camera info
    #     width, height = self.camera_info.width, self.camera_info.height

    #     # Create indices for x and y (no need for specific dtype)
    #     x_index = np.array([i for i in range(width)] * height, dtype=np.float32)
    #     y_index = np.array([[i] * width for i in range(height)], dtype=np.float32).ravel()

    #     # Stack x and y indices together to form pixel coordinates
    #     xy_index = np.vstack((x_index, y_index)).T  # x,y
    #     print("xy_index shape:", xy_index.shape)

    #     # Prepare arrays for depth and color information
    #     bgr0_vect = np.zeros([width * height, 4], dtype=np.float32)  # bgr0
    #     xyd_vect = np.zeros([width * height, 3], dtype=np.float32)  # x,y,depth
    #     XYZ_vect = np.zeros([width * height, 3], dtype=np.float32)  # real world coord
    #     ros_data = np.ones([width * height, 8], dtype=np.float32)  # [x,y,z,0,bgr0,semantics,confidence,0]

    #     # Ensure bgr_img is in the correct format (np.uint8)
    #     bgr_img = bgr_img.astype(np.uint8)

    #     # Handle NaN values in the depth image
    #     depth_img[np.isnan(depth_img)] = self.max_depth
    #     depth_img = depth_img.astype(np.float32)  # Ensure depth image is float32

    #     # Camera intrinsic matrix
    #     fx = self.camera_info.K[0]  # fx
    #     fy = self.camera_info.K[4]  # fy
    #     cx = self.camera_info.K[2]  # cx
    #     cy = self.camera_info.K[5]  # cy

    #     # Build intrinsic matrix using NumPy
    #     intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    #     # Calculate x, y, depth (Z) values
    #     xyd_vect[:, 0] = (x_index - cx) * depth_img.ravel() / fx  # X = (x - cx) * Z / fx
    #     xyd_vect[:, 1] = (y_index - cy) * depth_img.ravel() / fy  # Y = (y - cy) * Z / fy
    #     xyd_vect[:, 2] = depth_img.ravel()  # Z (depth)

    #     # Calculate 3D coordinates by multiplying with the inverse of the intrinsic matrix
    #     XYZ_vect = np.dot(xyd_vect, np.linalg.inv(intrinsic).T)

    #     # Transform color data from BGR image
    #     bgr0_vect[:, 0] = bgr_img[:, :, 2].reshape(-1)  # Blue
    #     bgr0_vect[:, 1] = bgr_img[:, :, 1].reshape(-1)  # Green
    #     bgr0_vect[:, 2] = bgr_img[:, :, 0].reshape(-1)  # Red

    #     # Concatenate real-world coordinates and color data
    #     ros_data[:, 0:3] = XYZ_vect  # 3D coordinates
    #     ros_data[:, 3:7] = bgr0_vect / 255.0  # Normalize BGR color values
    #     return ros_data


    def point_cloud_down_sampling(self, points, voxel_size):
        # Convert points to tensors and move them to GPU if available
        points = torch.tensor(points, dtype=torch.float32)
        points = points.cuda()

        num_channels = points.shape[1]  # Get the number of channels

        # Apply voxel grid downsampling
        voxel_indices = torch.floor(points[:, :3] / voxel_size)
        unique_indices, inverse_indices = torch.unique(voxel_indices, return_inverse=True, dim=0)

        downsampled_points = scatter_mean(points, inverse_indices, dim=0)

        # Adjust the number of channels based on the input point cloud
        if num_channels == 3:
            downsampled_points = downsampled_points[:, :3]  # Keep only the XYZ coordinates
        elif num_channels == 6:
            downsampled_points = downsampled_points[:, :6]  # Keep all the XYZ coordinates and additional channels

        return downsampled_points#.cpu().numpy()

    def to_tensor(self, points, to_cuda = True):
        if to_cuda:
            return torch.tensor(points).to('cuda')
        else:
            return torch.tensor(points)

    def get_chamfer_dists(self, pcd_x, pcd_y):
        # We compare the two sets of pointclouds by computing the chamfer loss.
        _, _, dist_x, dist_y = chamfer_distance(pcd_x, pcd_y)
        dist_x = dist_x.cpu().detach()
        dist_y = dist_y.cpu().detach()

        return dist_x, dist_y

    def load_complete_pc(self, plant, plant_idx, recon_target, plant_rotation, plant_position = [0,0,0], scale = 0.4, visual_com = False, visual_roi = False):
        print('loading complete point cloud of [tomato%s]' %
              (str(plant_idx) + '_r(%d)' % int(plant_rotation) + '_t(%d_%d_%d)'%(int(plant_position[0]), int(plant_position[1]), int(plant_position[2]))))
        plant_path = os.path.join(self.complete_plant_dir,
                                  plant + str(plant_idx) + '.pcd')

        pcd = o3d.io.read_point_cloud(plant_path)
        pcd.scale(scale, center=(0.0, 0.0, 0.0))

        T = np.eye(4)
        T[:3,:3] = pcd.get_rotation_matrix_from_xyz((0, 0, plant_rotation)) 
        T[0:3,3] = np.array(plant_position)/100

        pcd = pcd.transform(T)
        points = np.asarray(pcd.points).astype(np.float32)
        points = self.point_cloud_down_sampling(points, self.octomap_resulotion).reshape(1, -1, 3).to('cuda')

        self.fruit_poses, self.fruit_sizes, self.node_poses, self.node_sizes = \
            self.get_rios(plant_idx, plant_rotation, np.array(plant_position)/100)
        
        self.fruit_poses, self.fruit_sizes, self.node_poses, self.node_sizes = \
            torch.tensor(self.fruit_poses), torch.tensor(self.fruit_sizes).reshape(-1, 1), torch.tensor(self.node_poses), torch.tensor(self.node_sizes).reshape(-1, 1)

        if visual_roi:
            self.visual_roi(self.fruit_poses, self.fruit_sizes, self.node_poses, self.node_sizes)

        if visual_com:
            pcd_com_msg = np_to_pointcloud2_msg(points.cpu().numpy())
            self.complete_pc_publisher.publish(pcd_com_msg)

        self.complete_pc = self.classify_points_use_bbox(points, recon_target)
        
        self.xyz_rgb_buffer = []
    
    def get_rios(self, plant_idx, rotation, position):
        fruit_poses = []
        sizes_fruits = []
        node_poses = []
        sizes_nodes = []

        rois_fruits = pd.read_csv(self.roi_path + '/plant%s_export_fruit.csv'%str(plant_idx))
        rois_nodes = pd.read_csv(self.roi_path + '/plant%s_export_internode.csv'%str(plant_idx))

        if len(rois_fruits)>0:
            for roi in range(len(rois_fruits)):
                #read the fruits one by one and convert the pose using rotation and position and add to fruits list
                fruit = np.array([rois_fruits['loc_x'][roi], rois_fruits['loc_y'][roi], rois_fruits['loc_z'][roi]])
                #rotate the fruit
                fruit = np.dot(np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), fruit)
                #translate the fruit
                fruit_pose = fruit + np.array([position[0], position[1], position[2]])
                fruit_poses.append(fruit_pose)
                sizes_fruits.append(rois_fruits['radius'][roi])
        else:
            fruit_poses = [np.array([0,0,0])]
            sizes_fruits = [0.001]

        if plant_idx > 30:
            for roi in range(len(rois_nodes)):
                node = np.array([rois_nodes['loc_x'][roi], rois_nodes['loc_y'][roi], rois_nodes['loc_z'][roi]])
                #rotate the fruit
                node = np.dot(np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), node)
                #translate the fruit
                node_pose = node + np.array([position[0], position[1], position[2]])
                node_poses.append(node_pose)
                sizes_nodes.append(0.03)
        else:
            for roi in range(2, len(rois_nodes)-2):
                node = np.array([rois_nodes['loc_x'][roi], rois_nodes['loc_y'][roi], rois_nodes['loc_z'][roi]])
                #rotate the fruit
                node = np.dot(np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), node)
                #translate the fruit
                node_pose = node + np.array([position[0], position[1], position[2]])
                node_poses.append(node_pose)
                sizes_nodes.append(0.015)

        return np.array(fruit_poses), np.array(sizes_fruits), np.array(node_poses), np.array(sizes_nodes)

    def visual_roi(self, fruits, fruit_sizes, node_poses, node_sizes):
        # combine fruits and nodes
        rois_poses = torch.cat([fruits, node_poses], dim=0)
        rois_sizes = torch.cat([fruit_sizes, node_sizes], dim=0)

        RoiArray = MarkerArray() 
        current_ids = set()

        for id, (f, r) in enumerate(zip(rois_poses, rois_sizes)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker.type = Marker.CUBE
            marker.id = id
            # Set the scale of the marker
            marker.scale.x = r*2
            marker.scale.y = r*2
            marker.scale.z = r*2
            # Set the color
            color = [1,0,0,0.6] if id < len(fruits) else [1,0,1,0.6]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # Set the pose of the marker
            marker.pose.position.x = f[0]
            marker.pose.position.y = f[1]
            marker.pose.position.z = f[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            RoiArray.markers.append(marker)
            current_ids.add(id)

        # Handle marker deletion
        delete_markers = [
            Marker(id=marker_id, action=Marker.DELETE)
            for marker_id in self.previous_ids - current_ids
        ]

        self.previous_ids = current_ids

        # Publish the updated markers
        marker_array = MarkerArray(markers=delete_markers + RoiArray.markers)
        self.rois_pub.publish(marker_array)

    def get_point_cloud(self, visual_pc=False):  # [1,n, 3]
        color_image = self.color_image
        depth_image = self.depth_image

        point_cloud_raw = self.generate_cloud_data_common(
            color_image, depth_image)[...,0:6]
        xyz = point_cloud_raw[:, 0:3]
        rgb = point_cloud_raw[:, 3:6]

        rotation_matrix, translation_matrix = self.convert_matrix(
            self.tf_buffer.lookup_transform(self.ref_frame,
                                            'camera_depth_optical_frame',
                                            self.camera_info.header.stamp,
                                            rospy.Duration(1.0)))
                
        remove_thre = self.max_depth
        valid_label = xyz[..., 2]<remove_thre # (1, n)

        valid_xyz = xyz[valid_label]
        valid_rgb = rgb[valid_label]

        ####point cloud downsampling

        valid_xyz = np.asarray(valid_xyz).astype(np.float32).reshape(1, -1, 3)
        valid_rgb = np.asarray(valid_rgb).astype(np.float32).reshape(1, -1, 3) 

        ###transform to world frame
        valid_xyz[0,:,:] = (np.dot(rotation_matrix, valid_xyz[0].T)+np.array(translation_matrix).reshape(-1,1)).T
        valid_xyzrgb = np.concatenate((valid_xyz,valid_rgb), axis= 2) #[1,n, 6]

        valid_xyzrgb = np.zeros((1,1,6)) if valid_xyzrgb.shape[1]==0 else valid_xyzrgb
        valid_xyz = np.zeros((1,1,3)) if valid_xyz.shape[1]==0 else valid_xyz

        valid_xyzrgb = self.point_cloud_down_sampling(valid_xyzrgb[0], self.octomap_resulotion).unsqueeze(0)
        valid_xyz = self.point_cloud_down_sampling(valid_xyz[0], self.octomap_resulotion).unsqueeze(0)

        if visual_pc:
            pc2_msg_world_frame = np_to_pointcloud2_msg(
                valid_xyzrgb.cpu().numpy(),
                ref_frame=self.ref_frame,
                stamp=self.camera_info.header.stamp)
            self.current_pc_publisher.publish(pc2_msg_world_frame)
            rospy.sleep(1.0)

        return color_image, depth_image, valid_xyzrgb, valid_xyz  ##both on cuda

    def convert_matrix(self, t):
        rotation_matrix = R.from_quat([t.transform.rotation.x, 
                                        t.transform.rotation.y, 
                                        t.transform.rotation.z, 
                                        t.transform.rotation.w]).as_matrix()

        translation_matrix = np.array([t.transform.translation.x, t.transform.translation.y, 
                                        t.transform.translation.z]).reshape(-1,1)
        return rotation_matrix, translation_matrix

    def combine_pc(self, pc1, pc2, planner, visual_pc=False):
        """Add pc1 to pc2 using batched processing to reduce memory usage.
        Returns:
            tensor[1,n,3] on cuda: combined point cloud
        """
        if pc2.shape[1] > 0:
            if not pc1.is_cuda:
                pc1 = pc1.to('cuda')
            if not pc2.is_cuda:
                pc2 = pc2.to('cuda')
            
            if pc1.shape[1] == 0:
                return pc2, None, None, torch.tensor(0, dtype=torch.float32, device='cuda')

            dist1, _ = self.get_chamfer_dists(pc1[..., 0:3], pc2[..., 0:3])

            pc1_new_mask = dist1 > self.octomap_resulotion
            pc1_new = pc1[pc1_new_mask]
            
            # Efficient concatenation
            pc_output = torch.cat([pc2, pc1_new.unsqueeze(0)], dim=1)
            gt_ig = self.compute_roi_ig(pc1_new.unsqueeze(0), pc1, planner=planner)
            
            if visual_pc:
                pc2_msg_world_frame = np_to_pointcloud2_msg(
                    pc_output.cpu().numpy()[...,0:6],
                    ref_frame=self.ref_frame,
                    stamp=rospy.Time.now())
                self.acc_pc_publisher.publish(pc2_msg_world_frame)
            return pc_output, pc1_new.unsqueeze(0), pc1_new.shape, gt_ig.item()
        else:
            gt_ig = self.compute_roi_ig(pc1, pc1, planner=planner)
            return pc1, None, None, gt_ig.item()
    
    def cal_ig(self, pc1, pc2):
        """Add pc1 to pc2 using batched processing to reduce memory usage.
        Returns:
            tensor[1,n,3] on cuda: combined point cloud
        """
        if pc2 is not None:
            if not pc1.is_cuda:
                pc1 = pc1.to('cuda')
            if not pc2.is_cuda:
                pc2 = pc2.to('cuda')
            
            if pc1.shape[1] == 0:
                return pc2, None, None, torch.log(torch.tensor(1, dtype=torch.float32, device='cuda'))

            dist1, _ = self.get_chamfer_dists(pc1[..., 0:3], pc2[..., 0:3])

            pc1_new_mask = dist1 > self.octomap_resulotion
            pc1_new = pc1[pc1_new_mask]
            
            # Efficient concatenation
            gt_ig = self.compute_roi_ig(pc1_new.unsqueeze(0), pc1)
            return gt_ig.item()
        else:
            gt_ig = self.compute_roi_ig(pc1, pc1)
            return gt_ig.item()
    
    def cal_coverage(self, acc_pc):
        '''
        Calculate coverage ratio between accumulated point cloud and complete point cloud
        Args:
            acc_pc: tensor[1,N,4] - accumulated point cloud with 4 channels (xyzc)
        Returns:
            str: coverage ratio as string 'covered_points/total_points'
            float: coverage ratio as float
        '''
        if not acc_pc.is_cuda:
            acc_pc = acc_pc.to('cuda')

        complete_pc_roi = self.complete_pc[:, self.complete_pc[0,:,-1]==1, :]
        acc_pc_roi = acc_pc[:, acc_pc[0,:,-1]==1, :]

        # Handle case where there are no ROI points in accumulated point cloud
        if acc_pc_roi.shape[1] == 0:
            return '0/%d' % complete_pc_roi.shape[1], 0.0

        _, dist2 = self.get_chamfer_dists(acc_pc_roi[..., 0:3], complete_pc_roi[..., 0:3])

        covered_points = dist2 < self.octomap_resulotion
        cov_ratio = torch.sum(covered_points).item() / complete_pc_roi.shape[1]
        
        return '%d/%d' % (torch.sum(covered_points).item(), 
                          complete_pc_roi.shape[1]), cov_ratio
    
    def cal_pc_center(self, points):
        return points.mean(dim=-1, keepdim=True)
    
    def cal_point_weights(self, pcd, view_pose):
        center = self.cal_pc_center(pcd)
        p_view = center.squeeze()-view_pose
        p_xy = center-pcd
        weights = (torch.matmul(p_xy.transpose(1, 2), p_view)+1)/2
        return weights.unsqueeze(1)
    
    def get_weigted_pcd(self, pcd, view_pose):
        weights = self.cal_point_weights(pcd, view_pose)
        weighted_pcd = torch.concat((pcd, weights), dim=1)
        return weighted_pcd
    
    def compute_roi_ig(self, pcd_new, pcd_partial, planner):
        if pcd_new.shape[1]<=3:
            return torch.tensor(0, dtype=torch.float32, device='cuda')
        else:
            pcd_new_num = pcd_new.shape[1]
            roi_points_num_new = pcd_new[0,:,-1].sum()
            pcd_partial_num = pcd_partial.shape[1]

            if planner.startswith('ssl_semantic_nbv'):
                if 'NOIG' in planner.split('_'):
                    return torch.tensor(pcd_new_num/pcd_partial_num)
                else:
                    return torch.log(roi_points_num_new+1) + torch.log(torch.tensor(pcd_new_num)+1)/10
            
            elif planner=='ssl_nbv' or planner=='predefined_global' or planner=='random_global' or planner=='voxel_global_nbv':
                return torch.tensor(pcd_new_num/pcd_partial_num)
            
            #paper 1 ig: roi_points_num/pcd_partial.shape[1]
        #paper 4 ig: self.symlog(roi_points_num, a=1)+self.symlog(torch.tensor(pcd_new.shape[1]), a=0.1)

        #(1) self.symlog(roi_points_num/pcd_partial.shape[1], a=1)+self.symlog(roi_points_num, a=1) kind of the best
        #(2) self.symlog(roi_points_num, a=1)
        #(3) self.symlog(roi_points_num/pcd_partial.shape[1], a=1)+self.symlog(roi_points_num, a=1)+self.symlog(torch.tensor(pcd_new.shape[1]), a=1)
        #(4) self.symlog(roi_points_num, a=1)+self.symlog(torch.tensor(pcd_new.shape[1]), a=0.1) # kind of the best
        #(5) self.symlog(roi_points_num/pcd_partial.shape[1], a=1) # worse
        #(6) roi_points_num/pcd_partial.shape[1] bad
        #(7) torch.tensor(pcd_new.shape[1]/pcd_partial.shape[1]).cuda() paper 3 method
        #(8) self.symlog(roi_points_num_new, a=1) + self.symlog(roi_points_num_partial, a=0.2) + self.symlog(torch.tensor(pcd_new.shape[1]), a=0.1)
        #(9) torch.log(roi_points_num_new+1)*torch.log(torch.tensor(pcd_new_num)+1) + torch.log(torch.tensor(pcd_new_num)+1)
        #### (10) torch.log(roi_points_num_new+1)
        #(11) torch.log(torch.tensor(pcd_new_num)+1)
        ####(12) torch.log(roi_points_num_new+1) + torch.log(torch.tensor(pcd_new_num)+1)/10
        #(13) torch.log(roi_points_num_new + non_roi_points_num_new/10+1)

    @staticmethod
    def symlog(x, a=1):
        return torch.sign(x) * torch.log(torch.abs(x)*a + 1.0)
            
    def classify_points_use_bbox(self, point_cloud, recon_target):
        """
        Optimized version using vectorized operations
        """
        if recon_target is None:
            return point_cloud
        elif recon_target == 'PLANT':
            return torch.cat([
                point_cloud, 
                torch.ones((1, point_cloud.shape[1], 1), device='cuda')
            ], dim=-1)
            
        # Vectorized bbox calculations
        center_poses = (self.fruit_poses if recon_target == 'FRUIT' else self.node_poses).to('cuda')
        sizes = (self.fruit_sizes if recon_target == 'FRUIT' else self.node_sizes).to('cuda')
        
        points = point_cloud[0, :, :3]
        
        # Compute all distances at once
        points_expanded = points.unsqueeze(1)
        center_expanded = center_poses.unsqueeze(0)
        sizes_expanded = sizes.unsqueeze(0)
        
        # Vectorized bounds check
        in_bounds = torch.all(
            torch.abs(points_expanded - center_expanded) <= sizes_expanded,
            dim=-1
        )
        
        # Combine results
        mask = torch.any(in_bounds, dim=1).unsqueeze(0).unsqueeze(-1)
        return torch.cat([point_cloud, mask], dim=-1)

def np_to_pointcloud2_msg(points, ref_frame="world", stamp=None):

    if len(points.shape) == 3:
        points = points[0]

    if points.shape[-1] == 3:
        field = 'xyz'
        step = 3
    elif points.shape[-1] == 6:
        field = 'xyzrgb'
        step = 6

    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  #itemsize==4
    data = points.astype(dtype).ravel().tobytes()[:]
    fields = [
        PointField(name=n,
                   offset=i * itemsize,
                   datatype=PointField.FLOAT32,
                   count=1) for i, n in enumerate(field)
    ]

    stamp = rospy.Time.now() if stamp == None else stamp
    header = std_msgs.Header(frame_id=ref_frame, stamp=stamp)

    return PointCloud2(header=header,
                       height=1,
                       width=points.shape[0],
                       is_dense=True,
                       is_bigendian=False,
                       fields=fields,
                       point_step=(itemsize * step),
                       row_step=(itemsize * step * points.shape[0]),
                       data=data)
