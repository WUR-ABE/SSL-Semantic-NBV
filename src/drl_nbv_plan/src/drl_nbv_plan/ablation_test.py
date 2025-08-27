#!/usr/bin/env python3

from drl_environment.environment import Environment
import rospy
import time
import os
from agent import AgentGlobal
import csv
import numpy as np


class Evaluator_compare_planners:
    def __init__(self):
        self.data_save_dir = (
            "/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/prilinary_test/%s"
            % (str(time.ctime()))
        )

        if not os.path.exists(self.data_save_dir):
            os.makedirs(self.data_save_dir)

        self.max_test  = 25
        self.planners = ['ssl_semantic_nbv']
        self.recon_targets = ['NODE'] #['PLANT','FRUIT','NODE']
        self.max_allowed_steps = 15
        self.num_steps = 15
        self.ckpt_dir = '/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q3'
        self.ckpts = np.arange(0, 2001, 200)
        
    def update_mode(self, mode):
        self.agent_ssl_semantic_nbv.update_mode(mode=mode)
        self.environment.update_mode(mode=mode)
    
    def config_env(self):
        self.config = {
            "pc_size": 512,
            'experiment': 0,
            'octo_resolution': 0.003,
            'planners': self.planner,
            'recon_target': self.recon_target, #'PLANT','FRUIT','NODE'
            'max_step': self.num_steps
            }
        self.environment = Environment(self.config)
    
    def create_agent(self):
        self.agent_ssl_semantic_nbv  = AgentGlobal(self.environment.config_middle['num_col']*self.environment.config_middle['num_row'], planner=self.planner)
        self.agent_ssl_semantic_nbv.config_evaluate_only()
        #print ckp key(s)
        self.agent_ssl_semantic_nbv.load_checkpoint('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q3/%s/%s/%s_model_epi_%s' % (self.planner, self.recon_target, self.planner, self.ckpt))        
        self.update_mode(mode="valid")

    def run(self):
        print(
            "========================================Start running========================================"
        )
        for recon_target in self.recon_targets:
            self.recon_target = recon_target
            for planner in self.planners:
                self.planner = planner
                self.config_env()
                self.environment.set_seed()

                # create a new file for each planner and target
                self.recon_file = open(os.path.join(self.data_save_dir, '%s_%s.csv' % (self.planner, self.recon_target)), 'a+')
                self.recon_writer = csv.writer(self.recon_file)
                
                cov = ['cov_%s'% step for step in range(self.max_allowed_steps)]
                curr_view = ['curr_hw_%s'% step for step in range(self.max_allowed_steps)]
                columns = ['planner', 'target', 'ckpt', 'plant_xyzr','test_num'] + cov+['auc']+curr_view+['total_distance']+['step_time']
                self.recon_writer.writerow(columns)

                for ckpt in self.ckpts:
                    self.ckpt = ckpt
                    self.create_agent()
                    
                    average_auc = 0
                    for test_count in range(self.max_test):
                        print("===========Validating:{}-{}-{}-{}=========".format(self.planner, self.recon_target, self.ckpt, test_count))
                        
                        coverage_list, auc, hw_list, traj_distance, step_time = self.valid_once()

                        #fill nan in the coverage list to have a length==self.max_step
                        coverage_list = coverage_list + [np.nan]*(self.max_allowed_steps-len(coverage_list))
                        hw_list = hw_list + [np.nan]*(self.max_allowed_steps-len(hw_list))

                        self.recon_writer.writerow([self.planner, self.recon_target, self.ckpt, '%s_%s_%s_%s_%s'%(self.environment.plant, self.environment.x, self.environment.y, self.environment.z, self.environment.rotation), str(test_count), *coverage_list, str(auc), *hw_list, traj_distance, step_time])
                        average_auc+=auc
                    print("Average AUC for {}-{}-{}:{}".format(self.planner, self.recon_target, self.ckpt, average_auc/self.max_test))

    def euclidean_distance(self, xyz1, xyz2):
        xyz1 = np.array(xyz1)
        xyz2 = np.array(xyz2)
        return np.linalg.norm(xyz1 - xyz2)

    def valid_once(self):
        done = False
        step_time = 0
        
        current_acc, current_view_state, current_view_index, current_hw = self.environment.reset_global(planner=self.planner)
        self.viewpoints_1d = self.environment.viewpoints_1d

        coverage_list = []
        hw_list = []
        while not done:
            
            _, cr = self.environment.nbv_compute.cal_coverage(
                self.environment.pcd_acc
            )
            print('current_view_index',current_view_index, 'current_coverage:', cr)

            coverage_list.append(cr)
            hw_list.append(current_hw)
            
           
            time_start = time.time()
            _, nbv_index = self.agent_ssl_semantic_nbv.act(
            0,
            current_acc,
            current_view_state,
            )
            time_end = time.time()
        
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

            current_hw = next_hw

        total_distance = 0
        for j in range(len(hw_list)-1):
            xyz1 = self.environment.viewpoints[hw_list[j][0],hw_list[j][1], 0:3]
            xyz2 = self.environment.viewpoints[hw_list[j+1][0],hw_list[j+1][1], 0:3]
            distance = self.euclidean_distance(xyz1,xyz2)
            total_distance+=distance
        return coverage_list, np.trapz(np.array(coverage_list))/(self.environment.max_step-1), hw_list, total_distance, step_time/(self.environment.max_step-1)

if __name__ == "__main__":
    rospy.init_node("Evaluator_ablation")
    trainer = Evaluator_compare_planners()
    trainer.run()
