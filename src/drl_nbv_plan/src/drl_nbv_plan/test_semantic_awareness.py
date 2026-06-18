#!/usr/bin/env python3

from drl_environment.environment import Environment
import rospy
import time
import os
from agent import AgentGlobal
import csv
import numpy as np
import torch


class Evaluator_compare_planners:
    def __init__(self):
        self.data_save_dir = (
            "/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/prilinary_test/"
        )

        if not os.path.exists(self.data_save_dir):
            os.makedirs(self.data_save_dir)

        self.modules_test = ['positionemb', 'recurrent', 'weightedpc', 'tanhig']
        self.max_test  = 25
        self.total_iteration_count = 0
        self.planners = ['ssl_semantic_nbv']#,'ssl_nbv','ssl_semantic_nbv','random_global']#['predefined_global','ssl_local_nbv', 'ssl_semantic_nbv', 'voxel_local_nbv', 'voxel_global_nbv','random_local', 'random_global']#['ssl_local_nbv', 'ssl_semantic_nbv', 'voxel_local_nbv', 'voxel_global_nbv','random_local', 'random_global','localgt']
        
        self.train_target = 'NODE'
        self.recon_target = 'PLANT' #'PLANT','FRUIT','NODE'

        self.recon_file = open(os.path.join(self.data_save_dir, 'test_semantic_aware', 'train_%s_recon_%s.csv' % (self.train_target, self.recon_target)), 'a+')
        self.recon_writer = csv.writer(self.recon_file)
        self.max_step = 20
        cov = ['cov_%s'% step for step in range(self.max_step)]
        curr_view = ['curr_hw_%s'% step for step in range(self.max_step)]
        columns = ['planner', 'valid_modules','plant_xyzr','test_num'] + cov+['auc']+curr_view+['total_distance']+['step_time']
        self.recon_writer.writerow(columns)
        
    def update_mode(self, mode):
        # self.agent_local.update_mode(mode=mode)
        self.agent_global.update_mode(mode=mode)
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
            'train_target': self.train_target,
            'recon_target': self.recon_target, #'PLANT','FRUIT','NODE'
            'max_step': 15
            }
        
        self.environment = Environment(self.config)

        self.agent_global  = AgentGlobal(self.environment.config_middle['num_col']*self.environment.config_middle['num_row'], planner='ssl_semantic_nbv')
        self.agent_global.config_evaluate_only()
        self.agent_global.load_checkpoint('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q1/work4/%s/ssl_semantic_nbv_model_epi_2000' % self.config['train_target'])

        self.update_mode(mode="valid")

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
                coverage_list = coverage_list + [np.nan]*(self.max_step-len(coverage_list))
                hw_list = hw_list + [np.nan]*(self.max_step-len(hw_list))

                self.recon_writer.writerow([planner, 'recurrent_positionemb', '%s_%s_%s_%s_%s'%(self.environment.plant, self.environment.x, self.environment.y, self.environment.z, self.environment.rotation), str(test_count), *coverage_list, str(auc), *hw_list, traj_distance, step_time])
                average_auc+=auc
            print("Average AUC for {}:{}".format(planner, average_auc/self.max_test))

    def euclidean_distance(self, xyz1, xyz2):
        xyz1 = np.array(xyz1)
        xyz2 = np.array(xyz2)
        return np.linalg.norm(xyz1 - xyz2)

    def valid_once(self):
        done = False
        recurrent_state = None
        step_time = 0
        
        current_acc, current_view_state, current_view_index, current_hw = self.environment.reset_global(planner=self.planner)
        self.viewpoints_1d = self.environment.viewpoints_1d

        coverage_list = []
        hw_list = []
        while not done:
            print('current_view_index',current_view_index)
            _, cr = self.environment.nbv_compute.cal_coverage(
                self.environment.pcd_acc
            )

            coverage_list.append(cr)
            hw_list.append(current_hw)

            time_start = time.time()
            _, nbv_index = self.agent_global.act(
            0,
            current_acc,
            current_view_state,
        )
            time_end = time.time()
            recurrent_state_ = None
            
            step_time+=(time_end - time_start)

            (
            next_acc,
            next_view_state,
            _,
            done,
            next_hw,
        ) = self.environment.step_global(nbv_index)
            
            current_acc = next_acc

            current_view_state = next_view_state
            current_view_index = nbv_index

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
    rospy.init_node("Evaluator_Semantic_Awareness")
    trainer = Evaluator_compare_planners()
    trainer.run()
