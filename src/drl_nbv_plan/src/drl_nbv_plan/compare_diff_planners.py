#!/usr/bin/env python3

from drl_environment.environment import Environment
import rospy
import time
import os
from agent import AgentLocal, AgentGlobal, AgentGlobalWork1
import csv
import numpy as np
# import random
from baseline_planners import GlobalRandom, PredefinedGlobal
from active_vision.volumetric_planner import ActivePlanner
import torch


class Evaluator_compare_planners:
    def __init__(self):
        self.data_save_dir = (
            "/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/prilinary_test/%s"
            % (str(time.ctime()))
        )

        if not os.path.exists(self.data_save_dir):
            os.makedirs(self.data_save_dir)

        # self.modules_test = ['positionemb', 'recurrent', 'weightedpc', 'tanhig']
        self.max_test  = 25
        self.total_iteration_count = 0
        self.planners = ['ssl_semantic_nbv', 'ssl_nbv', 'predefined_global', 'random_global', 'voxel_global_nbv']
        
        self.recon_file = open(os.path.join(self.data_save_dir, 'results.csv'), 'a+')
        self.recon_writer = csv.writer(self.recon_file)
        self.max_allowed_step = 20
        cov = ['cov_%s'% step for step in range(self.max_allowed_step)]
        curr_view = ['curr_hw_%s'% step for step in range(self.max_allowed_step)]
        columns = ['planner', 'valid_modules','plant_xyzr','test_num'] + cov+['auc']+curr_view+['total_distance']+['step_time']
        self.recon_writer.writerow(columns)
        
    def update_mode(self, mode):
        # self.agent_local.update_mode(mode=mode)
        self.agent_ssl_semantic_nbv.update_mode(mode=mode)
        self.agent_ssl_nbv.update_mode(mode=mode)
        self.environment.update_mode(mode=mode)
    
    def check_modules(self):
        including_module = []
        for key in self.config.keys():
            if key in self.modules_test:
                if self.config[key] == True:
                    including_module.append(key)
        return '_'.join(including_module)
    
    def config_env_model(self):
        self.config = {
            "num_actions": 8,
            "recurrent": True, ##True or False
            "weightedpc": False, ##True or False
            "positionemb":True, ## True or False
            "tanhig": False,
            "pc_size": 512,
            'experiment': 0,
            'octo_resolution': 0.003,
            'planners': self.planners,
            'recon_target': 'NODE', #'PLANT','FRUIT','NODE'
            'max_step': 15
            }
        
        self.environment = Environment(self.config)
        # self.viewpoints = self.environment.viewpoints

        # self.agent_local = AgentLocal(self.config)
        # self.valid_module = self.check_modules()
        # self.agent_local.config_evaluate_only()
        # self.agent_local.load_checkpoint('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/NEWDATA_LOCAL_PVCNN_FRUIT_NORMALBUFFER_500_IG4/ssl_local_nbv_best_model')
        
        self.agent_ssl_semantic_nbv  = AgentGlobal(self.environment.config_middle['num_col']*self.environment.config_middle['num_row'], planner='ssl_semantic_nbv')
        self.agent_ssl_semantic_nbv.config_evaluate_only()
        self.agent_ssl_semantic_nbv.load_checkpoint('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q1/work4/%s/ssl_semantic_nbv_model_epi_2000' % self.config['recon_target'])

        self.agent_ssl_nbv = AgentGlobalWork1(self.environment.config_middle['num_col']*self.environment.config_middle['num_row'])
        self.agent_ssl_nbv.config_evaluate_only()
        self.agent_ssl_nbv.load_checkpoint('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q1/work2/PLANT/ssl_nbv_best_model')
        
        self.update_mode(mode="valid")

        # self.localrandom = LocalRandom()
        self.globalrandom = GlobalRandom()
        self.predefinedglobal = PredefinedGlobal()
        # self.localgt = LocalGt()
        # self.voxel_local = ActivePlanner()
        self.voxel_global = ActivePlanner()

    def run(self):
        print(
            "========================================Start running========================================"
        )
        self.config_env_model()
        for planner in self.planners:
            self.planner = planner
            self.environment.set_seed()

            average_auc = 0
            for test_count in range(self.max_test):
                print("===========Validating:{}-{}=========".format(planner, test_count))
                coverage_list, auc, hw_list, traj_distance, step_time = self.valid_once()

                #fill nan in the coverage list to have a length==self.max_step
                coverage_list = coverage_list + [np.nan]*(self.max_allowed_step-len(coverage_list))
                hw_list = hw_list + [np.nan]*(self.max_allowed_step-len(hw_list))

                self.recon_writer.writerow([planner, 'recurrent_positionemb', '%s_%s_%s_%s_%s'%(self.environment.plant, self.environment.x, self.environment.y, self.environment.z, self.environment.rotation), str(test_count), *coverage_list, str(auc), *hw_list, traj_distance, step_time])
                average_auc+=auc
            print("Average AUC for {}:{}".format(planner, average_auc/self.max_test))

    def euclidean_distance(self, xyz1, xyz2):
        xyz1 = np.array(xyz1)
        xyz2 = np.array(xyz2)
        return np.linalg.norm(xyz1 - xyz2)

    def valid_once(self):
        # predefined_views = np.array([11, 13, 15, 22, 29, 31, 33, 40, 47, 49, 51, 58, 65, 67, 69])
        predefined_views = np.array([10, 13, 16, 22, 28, 31, 34, 40, 46, 49, 52, 58, 64, 67, 70])

        done = False
        recurrent_state = None
        step_time = 0
        
        if self.planner == 'ssl_semantic_nbv' or self.planner == 'voxel_global_nbv' or self.planner == 'random_global' or self.planner == 'ssl_nbv' or self.planner == 'predefined_global':
            current_acc, current_view_state, current_view_index, current_hw = self.environment.reset_global(planner=self.planner, predefined_views=predefined_views)
            self.viewpoints_1d = self.environment.viewpoints_1d

            if self.planner == 'voxel_global_nbv':
                grid_size, grid_center = self.environment.get_grid_info()
                self.viewpoints_1d = self.viewpoints_1d[...,[0,1,2,6,3,4,5]] #change to xyzwrpy
                self.voxel_global.init_voxel_grid(
                        grid_size = grid_size,
                        voxel_size = np.array([self.config['octo_resolution']]),
                        grid_center = grid_center)
                
                view_xyzwrpy = self.viewpoints_1d[0, current_view_index, :]
                ########################################
                pcd = self.environment.pcd_list[current_view_index][..., 0:3]
                # depth_image = self.voxel_global.convert_pc2depth(self.environment.pcd_list[current_view_index][..., 0:3], view_xyzwrpy, mode = 'xyz')
                # self.voxel_global.depth_to_voxels(depth_image, torch.ones_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1,1,2), view_xyzwrpy)
                #self.voxel_global.visualize()
                ##############################
                pcd = self.environment.pcd_list[current_view_index]
                pcd_target = self.environment.nbv_compute.classify_points_use_bbox(pcd, self.config['recon_target'])
                ####
                pcd_node = self.environment.nbv_compute.classify_points_use_bbox(pcd, 'NODE').cpu().numpy()
                pcd_node = pcd_node[pcd_node[..., -1]==1, 0:3]
                current_node_bbox = self.environment.update_roi_bbox(pcd_node)
                #####
                depth_image, semantics_image = self.voxel_global.convert_pc2depth(pcd_target, view_xyzwrpy, mode = 'xyzrgbc')
                self.voxel_global.depth_to_voxels(depth_image, torch.tensor(semantics_image), view_xyzwrpy)
                self.voxel_global.visualize()
                ###############################
        else:
            current_acc, current_view, current_hw = self.environment.reset(planner=self.planner)
            self.viewpoints = self.environment.viewpoints

            if self.planner == 'voxel_local_nbv':
                self.viewpoints = self.viewpoints[...,[0,1,2,6,3,4,5]] #change to xyzwrpy
                grid_size, grid_center = self.environment.get_grid_info()
                self.voxel_local.init_voxel_grid(
                        grid_size = grid_size,
                        voxel_size = np.array([self.config['octo_resolution']]),
                        grid_center = grid_center)
                
                view_xyzwrpy = self.viewpoints[current_hw[0],current_hw[1],:]
                depth_image = self.voxel_local.convert_pc2depth(self.environment.pcd_acc[..., 0:3], view_xyzwrpy)
                self.voxel_local.depth_to_voxels(depth_image, torch.ones_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1,1,2), view_xyzwrpy)
                # self.voxel_local.visualize()

        coverage_list = []
        hw_list = []
        while not done:
            
            _, cr = self.environment.nbv_compute.cal_coverage(
                self.environment.pcd_acc
            )
            print('current_view_index',current_view_index, 'current_coverage:', cr)

            coverage_list.append(cr)
            hw_list.append(current_hw)
            if self.planner == 'ssl_local_nbv':
                time_start = time.time()
                _, action, recurrent_state_ = self.agent_local.act(
                    0,
                    current_acc,
                    current_view,
                    recurrent_state,
                    current_hw,
                    self.viewpoints,
                )
                time_end = time.time()

            elif self.planner == 'ssl_semantic_nbv':
                time_start = time.time()
                _, nbv_index = self.agent_ssl_semantic_nbv.act(
                0,
                current_acc,
                current_view_state,
            )
                time_end = time.time()
                recurrent_state_ = None
            
            elif self.planner == 'ssl_nbv':
                time_start = time.time()
                _, nbv_index = self.agent_ssl_nbv.act(
                0,
                current_acc,
                current_view_state,
            )
                time_end = time.time()
                recurrent_state_ = None

            elif self.planner == 'random_local':
                time_start = time.time()
                action = self.localrandom.act(current_hw, self.viewpoints)
                time_end = time.time()
                recurrent_state_ = None
            
            elif self.planner == 'random_global':
                time_start = time.time()
                nbv_index = self.globalrandom.act(current_view_state, allow_repeat_view=True)
                time_end = time.time()
                recurrent_state_ = None

            elif self.planner == 'predefined_global':
                # remove the current view from the predefined views   
                predefined_views = np.delete(predefined_views, np.where(predefined_views == current_view_index))
                time_start = time.time()   

                if len(predefined_views) > 0:
                    nbv_index = self.predefinedglobal.act(predefined_views)
                else: 
                    nbv_index = current_view_index

                time_end = time.time()
                recurrent_state_ = None

            elif self.planner == 'localgt':
                in_bound_actions, in_bound_pcds=self.environment.get_inbound_pcds(current_hw)
                time_start = time.time()
                action, max_ig = self.localgt.act(in_bound_actions, in_bound_pcds, self.environment.pcd_acc)
                time_end = time.time()
                recurrent_state_ = None
                
            elif self.planner == 'voxel_local_nbv':
                time_start = time.time()
                in_bound_actions, in_bound_viewpoints=self.environment.get_inbound_actions(current_hw)
                _, nbv, gain, _ = self.voxel_local.active_view(in_bound_viewpoints)
                action = in_bound_actions[nbv]
                time_end = time.time()
                recurrent_state_ = None
                
                nbv_xyzwrpy = in_bound_viewpoints[nbv]
                nbv_pc = self.environment.get_nbv_pc(action, current_hw)[..., 0:3] 
                depth_image = self.voxel_local.convert_pc2depth(nbv_pc, nbv_xyzwrpy)

                self.voxel_local.depth_to_voxels(depth_image, torch.empty_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1,1,2), nbv_xyzwrpy)
                # self.voxel_local.visualize()
            
            elif self.planner == 'voxel_global_nbv':
                self.voxel_global.voxel_grid.crop_grid(current_node_bbox)
                time_start = time.time()
                _, nbv_index, gain, _ = self.voxel_global.active_view(self.viewpoints_1d[0])
                time_end = time.time()
                nbv_xyzwrpy = self.viewpoints_1d[0, nbv_index, :]
                #######################################
                #nbv_pc = self.environment.pcd_list[nbv_index][..., 0:3]
                # depth_image = self.voxel_global.convert_pc2depth(nbv_pc, nbv_xyzwrpy, mode = 'xyz')
                # self.voxel_global.depth_to_voxels(depth_image, torch.ones_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1,1,2), nbv_xyzwrpy)
                #self.voxel_global.visualize()
                ######################
                nbv_pc = self.environment.pcd_list[nbv_index]
                nbv_pc_target = self.environment.nbv_compute.classify_points_use_bbox(nbv_pc, self.config['recon_target'])
                ###
                pcd_node = self.environment.nbv_compute.classify_points_use_bbox(nbv_pc, 'NODE').cpu().numpy()
                pcd_node = pcd_node[pcd_node[..., -1]==1, 0:3]
                current_node_bbox = self.environment.update_roi_bbox(pcd_node)
                ###
                depth_image, semantics_image = self.voxel_global.convert_pc2depth(nbv_pc_target, nbv_xyzwrpy, mode = 'xyzrgbc')
                self.voxel_global.depth_to_voxels(depth_image, torch.tensor(semantics_image), nbv_xyzwrpy)
                self.voxel_global.visualize()
                ###############################################

                recurrent_state_ = None

            step_time+=(time_end - time_start)

            if self.planner == 'ssl_semantic_nbv' or self.planner == 'voxel_global_nbv' or self.planner == 'random_global' or self.planner == 'ssl_nbv' or self.planner == 'predefined_global':
                (
                next_acc,
                next_view_state,
                _,
                done,
                next_hw,
            ) = self.environment.step_global(nbv_index)
                
            else:
                (
                    next_acc,
                    next_view,
                    _,
                    done,
                    _,
                    next_hw,
                ) = self.environment.step(action, current_hw)

            current_acc = next_acc

            if self.planner=='ssl_semantic_nbv' or self.planner == 'voxel_global_nbv' or self.planner == 'random_global' or self.planner == 'ssl_nbv' or self.planner == 'predefined_global':
                current_view_state = next_view_state
                current_view_index = nbv_index
            else:
                current_view = next_view

            recurrent_state = recurrent_state_
            current_hw = next_hw

        total_distance = 0
        for j in range(len(hw_list)-1):
            xyz1 = self.environment.viewpoints[hw_list[j][0],hw_list[j][1], 0:3]
            xyz2 = self.environment.viewpoints[hw_list[j+1][0],hw_list[j+1][1], 0:3]
            distance = self.euclidean_distance(xyz1,xyz2)
            total_distance+=distance
        return coverage_list, np.trapz(np.array(coverage_list))/(self.environment.max_step-1), hw_list, total_distance, step_time/(self.environment.max_step-1)

if __name__ == "__main__":
    rospy.init_node("Evaluator_SS1")
    trainer = Evaluator_compare_planners()
    trainer.run()
