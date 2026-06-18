#!/usr/bin/env python3

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Quaternion, Vector3,PoseArray
import rospy
from typing import Union
from std_msgs.msg import ColorRGBA
import torch
from scipy.spatial.transform import Rotation
from view_sampling.position_ember import get_embedder
from camera_control.camera_control import CameraControl
from camera_control.camera_pose_broadcaster import CameraPoseBroadcaster
import random

class ViewSampler:
    def __init__(self, distribution_type:str):
        if distribution_type not in ('plane', 'cylinder'):
            raise ValueError("Input must be 'plane_distribtion' or 'cylinder_distribtion'")
        
        self.ref_frame = "world"
        self.distribution_type = distribution_type
        self.embedder, self.out_dim = get_embedder(multires=10, i=0, input_dims=3)
        self.interv_h = 0.05
        self.interv_ang = 10
        self.interv_w = 0.05

        self.markerarr_pub = rospy.Publisher(
            "visual_candidates", MarkerArray, queue_size=50)
        self.marker_pub = rospy.Publisher(
            "visual_currentview", MarkerArray, queue_size=50)
        self.plane_pub = rospy.Publisher('plane_marker', Marker, queue_size=1)
    
    def sample_views(self, **kwargs) -> np.array:
        if self.distribution_type == 'plane':
            if 'interval_h' in kwargs:
                self.interv_h = kwargs['interval_h'] 
            if 'interval_w' in kwargs:
                self.interv_w = kwargs['interval_w']
            
            args = {
                    "center": {"x": kwargs['center_x'], "y": kwargs['center_y'], "z": self.interv_h*(kwargs['num_row']-1)/2},
                    "plane_views": {
                    "interval": {"dh": self.interv_h, "dw": self.interv_w},
                    "columns": kwargs['num_col'],
                    "rows": kwargs['num_row'],
                    "plane_size": {"height": self.interv_h*(kwargs['num_row']-1), "width": self.interv_w*(kwargs['num_col']-1)}
                },
            }
            self.center = [args['center']['x'], args['center']['y'], args['center']['z']]

        elif self.distribution_type == 'cylinder':
            if 'interval_h' in kwargs:
                self.interv_h = kwargs['interval_h']
            if 'interval_ang' in kwargs:
                self.interv_ang = kwargs['interval_ang']
            
            args = {
                "cylinder_views": {
                "height": self.interv_h*(kwargs['num_row']-1),
                "radius": kwargs['radius'], # radius_span=0.6 in the paper3
                "columns": kwargs['num_col'],
                "rows": kwargs['num_row'],
                "phi": [-((kwargs['num_col']-1)*self.interv_ang)/2, ((kwargs['num_col']-1)*self.interv_ang)/2],
                },
            }
        
        self.kwargs = args
            
        if self.distribution_type == 'plane':
            return self.plane_distribtion()
        elif self.distribution_type == 'cylinder':
            return self.cylinder_distribtion()
        
    def plane_distribtion(self) -> np.array:
        self.plane_center = np.array([self.center[0], 
                                      self.center[1], 
                                      self.center[2]])
        self.plane_height = self.kwargs['plane_views']['plane_size']['height']
        self.plane_width = self.kwargs['plane_views']['plane_size']['width']
        inter_height = self.kwargs['plane_views']['interval']['dh']
        inter_width = self.kwargs['plane_views']['interval']['dw']

        h = np.arange(0, self.plane_height+(1E-5), inter_height)
        w = np.arange(0, self.plane_width+(1E-5), inter_width)[::-1]

        w, h = np.meshgrid(w,h)
        wh = np.stack([w,h], axis=-1)
        
        wh = wh - np.array([self.plane_width/2, self.plane_height/2])
        wh = wh + np.array([self.plane_center[1], self.plane_center[2]])
        xyz = np.concatenate((np.ones((wh.shape[0], wh.shape[1]))[:,:, np.newaxis]*self.plane_center[0], wh), axis=-1)

        rpyw = np.array([0,0,0,-1])
        xyzrpyw = np.concatenate((xyz, np.tile(rpyw, xyz.shape[0]*xyz.shape[1]).reshape(xyz.shape[0], xyz.shape[1], -1)), axis=-1)
        return xyzrpyw
    
    def cylinder_distribtion(self) -> np.array:
        height = self.kwargs['cylinder_views']['height']
        radius = self.kwargs['cylinder_views']['radius']
        columns = self.kwargs['cylinder_views']['columns']
        rows = self.kwargs['cylinder_views']['rows']
        phi = self.kwargs['cylinder_views']['phi']

        phis = np.linspace(self.degree2radians(phi[0]), self.degree2radians(phi[1]), columns)+np.pi
        heights = np.linspace(0, height, rows).reshape(-1,1)

        x = radius * np.cos(phis)
        y = radius * np.sin(phis)
        xy = np.stack((x, y), axis=-1)[np.newaxis, ...]
        xy = xy.repeat(rows, axis=0)

        z = (heights*np.ones((xy.shape[0], xy.shape[1])))[..., np.newaxis]
        xyz = np.concatenate((xy, z), axis=-1)

        rotations = Rotation.from_euler('z', phis-np.pi).as_quat()[np.newaxis,...].repeat(rows, axis=0)
        xyzrpyw = np.concatenate((xyz, rotations), axis=-1)
        return xyzrpyw

    def np2markerarray(self, xyzrpyw:np.array, color:list, alpha: float):
        viewpoints = MarkerArray() 
        for id, pose in enumerate(xyzrpyw):
            marker = Marker()
            marker.header.frame_id = self.ref_frame
            marker.id = id
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.scale.x = 0.03
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = alpha
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.pose.orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6])
            marker.pose.position = Point(pose[0], pose[1], pose[2])
            viewpoints.markers.append(marker)
        return viewpoints
    
    def plane_boundary(self) -> np.array:
        plane_bottom_right = [self.plane_center[0], self.plane_center[1] - self.plane_width/2, self.plane_center[2] - self.plane_height/2]
        plane_bottom_left = [self.plane_center[0],  self.plane_center[1]+ self.plane_width/2, self.plane_center[2] - self.plane_height/2]
        plane_top_right = [self.plane_center[0],  self.plane_center[1]- self.plane_width/2, self.plane_center[2] + self.plane_height/2]
        plane_top_left = [self.plane_center[0],  self.plane_center[1]+ self.plane_width/2, self.plane_center[2] + self.plane_height/2]
        corners = np.array([plane_bottom_left, plane_bottom_right, plane_top_right, plane_top_left])
        return corners
    
    def degree2radians(self, degree:Union[np.ndarray, torch.Tensor, float, int]):
        return degree * np.pi / 180
    
    def radians2degree(self, radians:Union[np.ndarray,torch.Tensor, float, int]):
        return radians * 180 / np.pi
    
    def random_view(self, viewpoints: np.array):
        h = random.choice(range(viewpoints.shape[0]))
        w = random.choice(range(viewpoints.shape[1]))
        return np.asarray([viewpoints[h,w,:]]), [h,w]
    
    def quan2rpy(self, view_quan):
        position = view_quan[...,:3]
        quaternion = view_quan[...,3:]
        rot_vector = Rotation.from_quat(quaternion).as_rotvec() ##radians
        view_rpy = np.concatenate([position, rot_vector], axis=-1)
        return torch.tensor(view_rpy, dtype=torch.float32)[0], \
                torch.tensor(position, dtype=torch.float32)[0], \
                torch.tensor(rot_vector, dtype=torch.float32)[0]

    def position_embeding(self, position: torch.Tensor):
        embeded_position = self.embedder(position.unsqueeze(0))
        return embeded_position.unsqueeze(0) # or embeded_position.unsqueeze(0) if not pfrl

    def visual_candidates(self, candidates: Union[MarkerArray, np.array], color:list):
        if not isinstance(candidates, MarkerArray):
            if len(candidates.shape)==3:
                candidates = candidates.reshape(candidates.shape[0]*candidates.shape[1], -1)
            alpha = 0.5 if len(candidates)>1 else 1.0
            candidates = self.np2markerarray(candidates, color, alpha=alpha)
        if len(candidates.markers)>1:
            self.markerarr_pub.publish(candidates)
        else:
            self.marker_pub.publish(candidates)

    def visual_plane(self, color = [0.0, 1.0, 0.0], alpha = 0.5):
        marker = Marker()
        marker.header.frame_id = self.ref_frame
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = marker.ADD
        marker.scale = Vector3(0.02, self.plane_width, self.plane_height)
        marker.color = ColorRGBA(color[0], color[1], color[2], alpha)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.pose.position = Point(*self.plane_center)
        self.plane_pub.publish(marker)

    @classmethod
    def update_view(cls, viewpoints:np.ndarray, hw:list, action:int, simulation=True, check_in_bound=True):
        if action == 0:
            h_next = hw[0]+1
            w_next = hw[1]
        elif action == 1:
            h_next = hw[0]-1
            w_next = hw[1]
        elif action == 2:
            h_next = hw[0]
            w_next = hw[1]-1
        elif action == 3:
            h_next = hw[0]
            w_next = hw[1]+1
        elif action == 4:
            h_next = hw[0]+3
            w_next = hw[1]
        elif action == 5:
            h_next = hw[0]-3
            w_next = hw[1]
        elif action == 6:
            h_next = hw[0]
            w_next = hw[1]-3
        elif action == 7:
            h_next = hw[0]
            w_next = hw[1]+3
            
        if check_in_bound:
            inbound = cls.check_in_bound(viewpoints, h_next, w_next, simulation)
            if inbound:
                return np.asarray([viewpoints[h_next,w_next,:]]), [h_next,w_next], inbound
            else:
                return None, [h_next,w_next], inbound
        else:
            return np.asarray([viewpoints[h_next,w_next,:]]), [h_next,w_next], True
        
    @staticmethod
    def check_in_bound(viewpoints, h_next, w_next, simulation):
        if simulation:
            if 0<= h_next <= (viewpoints.shape[0]-1) and 0<= w_next <= (viewpoints.shape[1]-1):
                return True
            return False
        else:
            if h_next < 0 or h_next >= viewpoints.shape[0] or w_next < 0 or w_next >= viewpoints.shape[1]:
                return False
            else:
                if np.isnan(viewpoints[h_next,w_next, :]).any():
                    return False
            return True
  
class RvizVisualizer:
    def __init__(self):
        self.ref_frame = "world"
        self.poses_with_covariance_pub = rospy.Publisher(
            "poses_with_covariance", MarkerArray, queue_size=1
        )
        self.poses_pub = rospy.Publisher(
            "pose_estimation/poses", PoseArray, queue_size=1
        )
        self.curve_pub = rospy.Publisher("curve", Marker, queue_size=1)

    def visualize_poses_with_covariance(self, objects) -> None:
        """
        Visualize the poses with covariance in Rviz as a MarkerArray message
        :param objects: List of objects with poses and covariances
        """
        marker_array = MarkerArray()
        obj_poses = PoseArray()
        obj_poses.header.frame_id = self.world_frame_id
        for i, obj in enumerate(objects):
            pose = obj["pose"]
            obj_poses.poses.append(pose)
            variance = obj["var"]
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "poses_with_covariance"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(*np.sqrt(variance))
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
            marker_array.markers.append(marker)
        self.poses_pub.publish(obj_poses)
        self.poses_with_covariance_pub.publish(marker_array)
    
if __name__ == '__main__':
    rospy.init_node('view_samplint')
    config_cylinder = {
                'num_col': 19, 
                'num_row':10,
                'radius': 1,
                'interval_h': 0.2,
                'interval_ang': 10}
    
    config_plane = {'center_x': -0.6, 'center_y': 0,
                'num_col': 9,
                'num_row': 9,
                'interval_h': 0.15,
                'interval_w': 0.15,
            }

    distribution_type = 'plane'  #cylinder, plane
    viewsampler = ViewSampler(distribution_type)
    viewpoints = viewsampler.sample_views(**config_plane)

    viewput = viewsampler.position_embeding(torch.tensor(1))
    print(viewput.shape)
    print(viewput)

    for i in range(1000): 
        viewsampler.visual_candidates(viewpoints, color=[1,0,0]) 
        viewsampler.visual_plane()

    # print(viewpoints.shape)
    camera_control = CameraControl()
    camera_pose_broadcast = CameraPoseBroadcaster()
    hw = [5,5]
    # view, hw  = viewsampler.random_view(viewpoints)
    view = np.asarray([viewpoints[hw[0], hw[1],:]])

    camera_control.move_camera_to_pose(view[0])
    camera_pose_broadcast.broadcast(view[0])

    # while True:
    #     hw = [9,0]
    #     # view, hw  = viewsampler.random_view(viewpoints)
    #     view = np.asarray([viewpoints[hw[0], hw[1],:]])
    #     in_bound = True 
    #     while in_bound:
    #         # viewsampler.visual_plane()
    #         viewsampler.visual_candidates(viewpoints, color=[1,0,0]) 
    #         viewsampler.visual_candidates(view, color=[0,0,1])

    #         camera_control.move_camera_to_pose(view[0])
    #         camera_pose_broadcast.broadcast(view[0])

    #         rospy.sleep(50)
    #         view, hw, in_bound = ViewSampler.update_view(viewpoints, hw, action=3)


        
