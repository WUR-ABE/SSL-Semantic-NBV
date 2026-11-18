#!/usr/bin/env python3
"""Step 3 of the pipeline: self-supervised training in simulation.

Checkpoints are written to $NBV_MODELS/<experiment>_<recon_target>_<planner>/
(<planner>_model_epi_<N> every `save_interval` episodes and <planner>_best_model),
which is where the evaluation scripts look for them.

Settings (private ROS param `~name` or environment variable `NAME`):
    planners       comma-separated, e.g. ssl_semantic_nbv_NOCOLOR,ssl_nbv     [ssl_semantic_nbv_NOCOLOR]
    experiment     0 = normal plants, 1 = heavy-occlusion plants             [0]
    recon_target   PLANT, FRUIT or NODE                                       [PLANT]
    max_episode    [2000]   valid_interval [100]   valid_episodes [10]
    save_interval  [200]    best_after     [1000]  max_step [15]
    batch_size     [32]     buffer_size    [500]   update_number [10]
    out_dir        override the checkpoint folder (single planner only)
    overwrite      allow writing into a folder that already has checkpoints   [false]
"""

from utils.config import get_param
from utils.paths import model_dir
from drl_environment.environment import Environment
import rospy
from torch.utils.tensorboard import SummaryWriter
import os
from agent import AgentGlobal, AgentGlobalWork1
from tqdm import tqdm
import torch
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Trainer:
    def __init__(self):
        self.config = {
            "pc_size": 512,
            # 'ssl_semantic_nbv' variants (+ _NOPE/_NOLF/_NOIG ablations, _NOCOLOR = xyz + class only) or 'ssl_nbv'
            "planners": get_param("planners", ["ssl_semantic_nbv_NOCOLOR"]),
            'octo_resolution': 0.003,
            'experiment': get_param(
                "experiment", 0
            ),  # 0: simulated normal plants, 1: simulated heavy occlusion plants
            'recon_target': get_param("recon_target", "PLANT"),  # 'PLANT','FRUIT','NODE'
            'max_step': get_param("max_step", 15),
        }
        self.environment = Environment(self.config)

        self.max_episode = get_param("max_episode", 2000)
        self.valid_interval = get_param("valid_interval", 100)
        self.valid_episodes = get_param("valid_episodes", 10)
        self.save_interval = get_param("save_interval", 200)
        self.best_after = get_param("best_after", 1000)
        self.out_dir = get_param("out_dir", "")
        self.total_iteration_count = 0
        self.update_interval = self.config['max_step']  # update interval of iterations
        self.batch_size = get_param("batch_size", 32)
        self.buffer_size = get_param("buffer_size", 500)
        self.update_number = get_param("update_number", 10)
        self.start_training = False

    def update_mode(self, mode):
        self.agent.update_mode(mode=mode)
        self.environment.update_mode(mode=mode)

    def create_agent(self):
        if self.planner.startswith('ssl_semantic_nbv'):
            self.agent = AgentGlobal(
                self.environment.config_middle['num_col']
                * self.environment.config_middle['num_row'],
                self.planner,
            )
            self.agent.config_train(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                update_number=self.update_number,
            )

        elif self.planner.startswith('ssl_nbv'):
            self.agent = AgentGlobalWork1(
                self.environment.config_middle['num_col']
                * self.environment.config_middle['num_row']
            )
            self.agent.config_train(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                update_number=self.update_number,
            )

    def run(self):
        print(
            "========================================Start running========================================"
        )
        for planner in self.config['planners']:
            self.planner = planner
            self.log_name = (
                'ssl_semantic_nbv' if self.planner.startswith('ssl_semantic_nbv') else 'ssl_nbv'
            )
            if self.out_dir and len(self.config['planners']) == 1:
                self.data_save_dir = os.path.expanduser(self.out_dir)
            else:
                self.data_save_dir = model_dir(
                    self.config['experiment'], self.config['recon_target'], self.planner
                )
            existing = (
                [f for f in os.listdir(self.data_save_dir) if f.startswith(self.planner + "_")]
                if os.path.isdir(self.data_save_dir)
                else []
            )
            if existing and not get_param("overwrite", False):
                raise RuntimeError(
                    "%s already contains checkpoints (%s). Set overwrite:=true or choose another "
                    "out_dir / NBV_MODELS." % (self.data_save_dir, existing[0])
                )
            os.makedirs(self.data_save_dir, exist_ok=True)
            print("Saving checkpoints to %s" % self.data_save_dir)
            self.log_writer = SummaryWriter(os.path.join(self.data_save_dir, "log"))
            self.episodic_loss = float("inf")
            self.start_training = False
            valid_loss = float("inf")
            self.create_agent()

            for i in tqdm(range(self.max_episode + 1)):
                self.episode_count = i
                print("\nEpisode:{}".format(self.episode_count))
                print(
                    "Current learning rate : {}".format(
                        self.agent.optimizer.state_dict()["param_groups"][0]["lr"]
                    )
                )
                print("Current exploration rate: {}".format(self.agent.get_epsilon()))
                self.update_mode(mode="train")

                train_loss, start_training = self.train_one_traj_global()

                if self.episode_count % self.valid_interval == 0:
                    print("===========Validating=========")
                    self.update_mode(mode="valid")
                    valid_loss = self.valid_n_times(self.valid_episodes)
                    self.log_writer.add_scalar(
                        "%s/Valid Loss" % self.log_name, valid_loss, self.episode_count
                    )

                if start_training:
                    self.log_writer.add_scalar(
                        "%s/Train Loss" % self.log_name, train_loss, self.episode_count
                    )

                if self.episode_count % self.save_interval == 0:
                    self.agent.save_model(
                        "{}/{}_model_epi_{}".format(
                            self.data_save_dir, self.planner, self.episode_count
                        )
                    )

                if valid_loss < self.episodic_loss and self.episode_count > self.best_after:
                    self.agent.save_model(
                        "{}/{}_best_model".format(self.data_save_dir, self.planner)
                    )
                    self.episodic_loss = valid_loss

    def valid_once_global(self):
        done = False
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
                    loss = (ig_value[0, nbv_index] - reward[0, nbv_index]) ** 2
                else:
                    reward_filled = torch.where(
                        torch.isnan(reward), torch.tensor(ig_value), reward
                    ).detach()
                    loss = torch.sum((reward_filled - torch.tensor(ig_value)) ** 2)

            elif self.planner == 'ssl_nbv':
                loss = (ig_value[0, nbv_index] - reward[0, nbv_index]) ** 2

            valid_loss_one_traj += loss.item()

            current_acc = next_acc
            current_viewstate = next_viewstate
        return valid_loss_one_traj, np.trapz(np.array(coverage_list)) / (
            self.environment.max_step - 1
        )

    def train_one_traj_global(self):
        done = False
        loss_per_traj = 0

        # Get initial state
        current_acc, current_viewstate, _, _ = self.environment.reset_global(planner=self.planner)

        while not done:
            pre_ig, nbv_index = self.agent.act(
                self.episode_count,
                current_acc,
                current_viewstate,
            )
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
        if done and len(self.agent.replay_buffer.buffer) > self.batch_size:
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

        self.log_writer.add_scalar("%s/AUC" % self.log_name, auc / n, self.episode_count)

        return valid_loss / n / self.config['max_step']


if __name__ == "__main__":
    rospy.init_node("ssl_nbv_train")
    trainer = Trainer()
    trainer.run()
