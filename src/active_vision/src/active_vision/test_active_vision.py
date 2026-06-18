#!/usr/bin/env python3
import numpy as np
import random
from active_vision.volumetric_planner import ActivePlanner
from matplotlib import pyplot as plt
import rospy
import torch

class TestActivePlanner:
    def __init__(self):
        
        self.viewpoints = np.load('/home/jianchao/ros_workspace/deep_learning_nbv/src/nbv_data_collection/materials/poses.pkl', allow_pickle=True).poses
        self.viewpoints_wxyz = self.get_viewpoint_wxyz(self.viewpoints)

        self.viewpoints_pc_dir = '/home/jianchao/dataset/nbv_data/pc_on_different_viewpoints/viewpoints'
        self.test_plants = [3, 8]

        self.planner = ActivePlanner()

    def get_viewpoints_pc(self, plant, rotation, position):
        viewpoints_pc_list = []
        for view in range(33):
            pc_path = self.viewpoints_pc_dir + '/plant%s/xyr_%s_%s_%s/%s_pc.npy'%(str(plant), str(position[0]), str(position[1]), str(rotation), str(view))
            pcd = torch.tensor(np.load(pc_path)).cuda()
            viewpoints_pc_list.append(pcd)

        return viewpoints_pc_list
        
    def get_viewpoint_wxyz(self, views):
        viewpoints_wxyz = []
        for view in views:
            view = np.array([view.position.x, 
                                view.position.y, 
                                view.position.z, 
                                view.orientation.w,
                                view.orientation.x, 
                                view.orientation.y, 
                                view.orientation.z
                                ])
            viewpoints_wxyz.append(view)
        return np.array(viewpoints_wxyz)

    
    def start_test(self):
        plant = random.choice(self.test_plants)
        rotation = random.choice(np.arange(0, 360+20, 20))
        x, y, z = random.choice(np.arange(-10, 10+2, 2)), random.choice(np.arange(-10, 10+2, 2)), 0
        position = [x, y, z]
        viewpoint_pcs = self.get_viewpoints_pc(plant, rotation, position)
        view_index = random.randint(0, 32)

        for i in range(2):
            for step in range(10):
                view_wxyz = self.viewpoints_wxyz[view_index]
                point_cloud_global = viewpoint_pcs[view_index][..., 0:3]
                print(point_cloud_global.shape)
                depth_image = self.planner.convert_pc2depth(point_cloud_global, view_wxyz)

                self.planner.depth_to_voxels(depth_image, torch.empty_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1,1,2), view_wxyz)
                self.planner.visualize()
                _, nbv_idx, gain, _ = self.planner.active_view(self.viewpoints_wxyz)
                view_index = nbv_idx
            self.planner.voxel_grid.reset_grid()


if __name__ == '__main__':
    rospy.init_node("test_active_planner")
    test_active_planner = TestActivePlanner()
    test_active_planner.start_test()








    

    

         
        






    




