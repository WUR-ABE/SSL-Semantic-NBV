#!/usr/bin/env python3
from view_sampling.view_sampler import ViewSampler
from nbv_compute.nbv_computor import Nbvcomputor
import rospy
import numpy as np
from camera_control.camera_control import CameraControl
from camera_control.camera_pose_broadcaster import CameraPoseBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2
import std_msgs.msg as std_msgs
import os
import torch
from geometry_msgs.msg import Quaternion, Point

class Visualization:    
    def __init__(self): 
        self.viewsampler = ViewSampler(distribution_type="plane")
        self.nbv_compute = Nbvcomputor(octomap_resulotion=0.006)
        self.camera_control = CameraControl()
        self.camera_pose_broadcaster = CameraPoseBroadcaster()

        ### sumulation#####################################
        # self.config_middle = {
        #         'center_x': -0.6, 'center_y': 0,
        #         'num_col': 9,
        #         'num_row': 9,
        #         'interval_h': 0.15,
        #         'interval_w': 0.15,
        #     }
        # self.viewpoints = self.viewsampler.sample_views(**self.config_middle)
        # self.pcd_dir = '/home/jianchao/dataset/nbv_data/paper4/data_smaller_fov_distance'
        # self.plant_idx = '3'
        ##############################################################################

        ######real world#######################################################
        self.pcd_dir = '/home/jianchao/dataset/nbv_data/paper4/data_smaller_fov_distance'
        self.plant_idx = '35'
        self.viewpoints = np.load(os.path.join(self.pcd_dir, 'viewpoints_%s.npy'%str(self.plant_idx)), allow_pickle=True) # real world
        self.viewpoints = self.viewpoints[...,[0,1,2,6,3,4,5]]
        ##################################################################################
        self.hw = [5,5]
        self.viewspace_color = [1.0,0.5,0.0]
        self.viewpoints_color = [1.0,0.5,0.0]
        self.viewspace_alpha = 0.2  
        self.box_color = [0.0,1.0,0.0]
        self.box_alpha = 0.5

        self.box_marker_publisher = rospy.Publisher('box_marker', Marker, queue_size=10)
        self.pcd_publisher = rospy.Publisher("/camera/acc_point_cloud", PointCloud2, queue_size=1)
        self.predefined_viewpoints_publisher = rospy.Publisher('predefined_viewpoints', MarkerArray, queue_size=10)

    def draw_viewpoints(self):
        self.viewsampler.visual_candidates(self.viewpoints, color=self.viewpoints_color)# change color to light orange

    def draw_predefined_viewpoints(self):
        viewpoints_1d = self.viewpoints.reshape(-1, 7)
        # predefined_views = np.array([10, 13, 16, 22, 28, 31, 34, 40, 46, 49, 52, 58, 64, 67, 70]) # simulation
        predefined_views = np.array([8, 11, 28, 31, 48, 51, 68, 71, 88, 91, 108, 111, 128, 131, 148]) # real world

        # predefined_views = np.array([11, 13, 15, 22, 29, 31, 33, 40, 47, 49, 51, 58, 65, 67, 69])
        predefined_views_1d = viewpoints_1d[predefined_views]

        viewpoints = MarkerArray() 
        for id, pose in enumerate(predefined_views_1d):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.id = id
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.scale.x = 0.03
            marker.scale.y = 0.015
            marker.scale.z = 0.015
            marker.color.a = 1.0
            # set color to light blue
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 1
            marker.pose.orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6])
            marker.pose.position = Point(pose[0], pose[1], pose[2])
            viewpoints.markers.append(marker)
        self.predefined_viewpoints_publisher.publish(viewpoints)
    
    def draw_viewspace(self):
        self.viewsampler.visual_plane(color=self.viewspace_color, alpha=self.viewspace_alpha)
    
    def move_camera_2_viewpoint(self):
        view = np.asarray([self.viewpoints[self.hw[0], self.hw[1],:]])
        self.camera_control.move_camera_to_pose(view[0])
        self.camera_pose_broadcaster.broadcast(view[0])
    
    def draw_box(self, box_position = [0, 0, 0.1], box_size = [0.4, 0.4, 0.2]):
        box_marker = Marker()
        box_marker.header.frame_id = "world"
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        box_marker.pose.position.x = box_position[0]
        box_marker.pose.position.y = box_position[1]
        box_marker.pose.position.z = box_position[2]
        box_marker.scale.x = box_size[0]
        box_marker.scale.y = box_size[1]
        box_marker.scale.z = box_size[2]
        box_marker.color.r = self.box_color[0]
        box_marker.color.g = self.box_color[1]
        box_marker.color.b = self.box_color[2]
        box_marker.color.a = self.box_alpha
        self.box_marker_publisher.publish(box_marker)
    
    def draw_plant_pcd(self):
        pcd_path = os.path.join(self.pcd_dir, 'plant%s' % str(self.plant_idx), 'xyzr_0_0_0_0', 'pcd')
        pcd_list = os.listdir(pcd_path)
        pcd_list = [f for f in pcd_list if f.endswith('.npy')]
        pcd_msg = []
        for pcd_file in pcd_list:
            pcd = np.load(os.path.join(pcd_path, pcd_file))
            pcd_msg.append(pcd)
        pcd_msg = np.concatenate(pcd_msg, axis=1)
        pcd_msg = self.np_to_pointcloud2_msg(pcd_msg[..., 0:6], ref_frame="world", stamp=rospy.Time.now())
        self.pcd_publisher.publish(pcd_msg)
    
    def np_to_pointcloud2_msg(self, points, ref_frame="world", stamp=None):
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
    
    def draw_rois(self):
        fruits, fruit_sizes, node_poses, node_sizes = self.nbv_compute.get_rios(int(self.plant_idx), 0, [0, 0, 0])
        fruits, fruit_sizes, node_poses, node_sizes = \
            torch.tensor(fruits), torch.tensor(fruit_sizes).reshape(-1, 1), torch.tensor(node_poses), torch.tensor(node_sizes).reshape(-1, 1)
        self.nbv_compute.visual_roi(fruits, fruit_sizes, node_poses, node_sizes)
    
if __name__ == "__main__":
    rospy.init_node("draw_figures", anonymous=True)
    visualization = Visualization()
    while not rospy.is_shutdown():
        visualization.draw_viewpoints()
        # visualization.draw_viewspace()
        visualization.move_camera_2_viewpoint()
        visualization.draw_box()
        visualization.draw_plant_pcd()
        visualization.draw_predefined_viewpoints()
        visualization.draw_rois()
        rospy.sleep(0.1)
    
    
