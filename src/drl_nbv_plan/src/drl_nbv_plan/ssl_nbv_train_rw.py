#!/usr/bin/env python3
from drl_environment.environment_real_world import Environment_RW
import rospy
import time
from torch.utils.tensorboard import SummaryWriter
import os
from agent import AgentGlobal, AgentGlobalWork1
from tqdm import tqdm
import torch
import numpy as np
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Trainer:
    def __init__(self):
        self.data_save_dir = (
            "/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/%s"
            % (str(time.ctime()))
        )
        self.config = {
        "pc_size": 512,
        "planners":['ssl_semantic_nbv'],#['ssl_semantic_nbv','ssl_semantic_nbv_NOPE', 'ssl_semantic_nbv_NOIG', 'ssl_semantic_nbv_NOLF', 'ssl_nbv'],
        'octo_resolution': 0.006,
        'experiment': 2, # similated normal plants, 1: simulated heavy occlusion plants, 2: real plants
        'recon_target': 'NODE', #'PLANT','FRUIT','NODE'
        'max_step': 15,
    }
        
        self.environment = Environment_RW(self.config)
        
        self.max_episode = 2000 # 2000 for ssl_semantic_nbv, 1000 for ssl_nbv
        self.total_iteration_count = 0
        self.update_interval = self.config['max_step']  # update interval of iterations
        self.batch_size = 32
        self.buffer_size = 500
        self.update_number = 10 # 10 for ssl_semantic_nbv, 1 for ssl_nbv
        self.episodic_loss = 1E7
        self.start_training = False

        self.log_writer = SummaryWriter(self.data_save_dir + "/log")
        if not os.path.exists(self.data_save_dir + "/log"):
            os.makedirs(self.data_save_dir + "log")

    def update_mode(self, mode):
        self.agent.update_mode(mode=mode)
        self.environment.update_mode(mode=mode)
    
    def create_agent(self):    
        if self.planner.startswith('ssl_semantic_nbv'):
            self.agent = AgentGlobal(self.environment.view_range['num_col']*self.environment.view_range['num_row'], self.planner)
            self.agent.config_train(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                update_number=self.update_number,)
            self.agent.load_encoder_only('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q1/work4/%s/ssl_semantic_nbv_model_epi_2000'%(self.config['recon_target']))
                 
        elif self.planner.startswith('ssl_nbv'):
            self.agent = AgentGlobalWork1(self.environment.view_range['num_col']*self.environment.view_range['num_row'])
            self.agent.config_train(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                update_number=self.update_number,)
            # self.agent.load_checkpoint('/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/train/Q1/work2/PLANT/ssl_nbv_model_epi_10000')

    def run(self):
        print(
            "========================================Start running========================================"
        )
        for planner in self.config['planners']:
            self.planner = planner
            self.log_name = 'ssl_semantic_nbv' if self.planner.startswith('ssl_semantic_nbv') else 'ssl_nbv'
            self.create_agent()

            for i in tqdm(range(self.max_episode+1)):
                self.episode_count = i
                print("\nEpisode:{}".format(self.episode_count))
                print(
                    "Current learning rate : {}".format(
                        self.agent.optimizer.state_dict()["param_groups"][0]["lr"]
                    )
                )
                print("Current exploration rate: {}".format(self.agent.get_epsilon()))
                self.update_mode(mode="train")
            
                train_loss, start_training =  self.train_one_traj_global()

                if self.episode_count % 100 == 0: # 100 for ssl_semantic_nbv and for ssl_nbv
                    print("===========Validating=========")
                    self.update_mode(mode="valid")
                    valid_loss = self.valid_n_times(10)
                    self.log_writer.add_scalar("%s/Valid Loss"%self.log_name, valid_loss, self.episode_count)
                
                if start_training:
                    self.log_writer.add_scalar("%s/Train Loss"%self.log_name, train_loss, self.episode_count)

                if self.episode_count % 200 == 0: # 200 for ssl_semantic_nbv, 2000 for ssl_nbv
                    self.agent.save_model("{}/{}_model_epi_{}".format(self.data_save_dir, self.planner, self.episode_count))
                
                if valid_loss<self.episodic_loss and self.episode_count>500: # 1000 for ssl_semantic_nbv and for ssl_nbv
                    self.agent.save_model("{}/{}_best_model".format(self.data_save_dir, self.planner))
                    self.episodic_loss = valid_loss

    def valid_once_global(self):
        done = False
        # recurrent_rgb = torch.zeros(1, 1, 128).cuda()
        current_acc, current_viewstate, _, _ = self.environment.reset_global(planner=self.planner)
        self.viewpoints_1d = self.environment.viewpoints_1d

        valid_loss_one_traj = 0
        coverage_list = []

        while not done:
            _, cr = self.environment.nbv_compute.cal_coverage(
                self.environment.pcd_acc,
            )
            print(cr)

            coverage_list.append(cr)

            ig_value, nbv_index = self.agent.act(
                self.episode_count,
                current_acc,
                current_viewstate,
            )
            print(nbv_index)
            (
                next_acc,
                next_viewstate,
                reward,
                done,
                _,
            ) = self.environment.step_global(nbv_index)

            if self.planner.startswith('ssl_semantic_nbv'):
                if 'NOLF' in self.planner.split('_'):
                    loss = ((ig_value[0, nbv_index]-reward[0, nbv_index])**2)
                else:
                    reward_filled = torch.where(torch.isnan(reward), torch.tensor(ig_value), reward).detach()
                    loss = torch.sum((reward_filled-torch.tensor(ig_value))**2)

            elif self.planner== 'ssl_nbv':
                loss = ((ig_value[0, nbv_index]-reward[0, nbv_index])**2)

            valid_loss_one_traj += loss.item()

            current_acc = next_acc
            current_viewstate = next_viewstate
            # current_rgb = next_rgb
            # recurrent_rgb = recurrent_rgb_tp1
        return valid_loss_one_traj, np.trapz(np.array(coverage_list))/(self.environment.max_step-1)
    
    def train_one_traj_global(self):
        done = False
        loss_per_traj = 0

        # Initialize recurrent state
        # recurrent_rgb = torch.zeros(1, 1, 128, device='cuda')

        # Get initial state
        current_acc, current_viewstate, _, _ = self.environment.reset_global(planner=self.planner)
        
        while not done:
            pre_ig, nbv_index = self.agent.act(
                self.episode_count,
                current_acc,
                current_viewstate,
            )
            # self.agent.select_informative_views(pre_ig, current_viewstate, nbv_index)
            # Take action in environment
            next_acc, next_viewstate, reward, done, _ = self.environment.step_global(nbv_index)

            self.agent.observe(
                pcd=current_acc,
                viewstate=current_viewstate,
                nbv_index=nbv_index,
                reward=reward,
                done=done,
            )

            # Update current state
            current_acc = next_acc
            current_viewstate = next_viewstate

        # Update network if enough samples
        if (done and len(self.agent.replay_buffer.buffer) > self.batch_size):
            loss_per_traj = self.agent.update_network(self.episode_count)
            self.start_training = True

        return loss_per_traj, self.start_training
    
    def valid_n_times(self, n):
        valid_loss = 0
        auc = 0
        for i in range(n):
            print("===========Validating:{}=========".format(i))
            valid_loss_one_traj, auc_traj = self.valid_once_global()
            valid_loss += valid_loss_one_traj
            auc += auc_traj

        self.log_writer.add_scalar("%s/AUC"%self.log_name, auc / n, self.episode_count)

        return valid_loss/n/self.config['max_step']

if __name__ == "__main__":
    rospy.init_node("ssl_nbv_train")
    trainer = Trainer()
    trainer.run()
