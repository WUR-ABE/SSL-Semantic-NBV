#!/usr/bin/env python3
"""Ablation study: evaluate SSL-Semantic-NBV variants over training checkpoints.

Writes <out_dir>/<experiment>_<target>/ablation_results.csv.

Settings (private ROS param `~name` or environment variable `NAME`):
    planners      [ssl_semantic_nbv_NOCOLOR,ssl_semantic_nbv_NOPE_NOCOLOR,
                   ssl_semantic_nbv_NOPE_NOLF_NOCOLOR,ssl_semantic_nbv_NOPE_NOLF_NOIG_NOCOLOR]
    targets       [PLANT,FRUIT,NODE]
    experiment    [0]
    ckpts         checkpoint episodes to evaluate     [0,200,...,2000]
    max_test [25]  max_step [15]  max_allowed_step [15]
    models_dir    [$NBV_MODELS]
    out_dir       [$NBV_MATERIALS/results/ablation_test/<timestamp>]
"""

from utils.config import get_param
from utils.paths import MODELS_DIR, model_dir, results_dir
from drl_environment.environment import Environment
import rospy
import time
import os
from agent import AgentGlobal
import csv
import numpy as np


class Evaluator_ablation:
    def __init__(self):
        self.experiment = get_param("experiment", 0)
        self.planners = get_param(
            "planners",
            [
                "ssl_semantic_nbv_NOCOLOR",
                "ssl_semantic_nbv_NOPE_NOCOLOR",
                "ssl_semantic_nbv_NOPE_NOLF_NOCOLOR",
                "ssl_semantic_nbv_NOPE_NOLF_NOIG_NOCOLOR",
            ],
        )
        self.recon_targets = get_param("targets", ["PLANT", "FRUIT", "NODE"])
        self.max_test = get_param("max_test", 25)
        self.max_allowed_steps = get_param("max_allowed_step", 15)
        self.num_steps = get_param("max_step", 15)
        self.ckpts = get_param("ckpts", list(range(0, 2001, 200)))
        self.models_dir = os.path.expanduser(get_param("models_dir", MODELS_DIR))
        self.data_save_root = os.path.expanduser(get_param("out_dir", results_dir("ablation_test")))

    def update_mode(self, mode):
        self.agent_ssl_semantic_nbv.update_mode(mode=mode)
        self.environment.update_mode(mode=mode)

    def config_env(self):
        self.config = {
            "pc_size": 512,
            'experiment': self.experiment,
            'octo_resolution': 0.003,
            'planners': self.planner,
            'recon_target': self.recon_target,  #'PLANT','FRUIT','NODE'
            'max_step': self.num_steps,
        }
        self.environment = Environment(self.config)

    def create_agent(self):
        self.agent_ssl_semantic_nbv = AgentGlobal(
            self.environment.config_middle['num_col'] * self.environment.config_middle['num_row'],
            planner=self.planner,
        )
        self.agent_ssl_semantic_nbv.config_evaluate_only()
        ckpt_path = os.path.join(
            model_dir(self.experiment, self.recon_target, self.planner, self.models_dir),
            '%s_model_epi_%s' % (self.planner, self.ckpt),
        )
        print("Loading %s" % ckpt_path)
        self.agent_ssl_semantic_nbv.load_checkpoint(ckpt_path)
        self.update_mode(mode="valid")

    def run(self):
        print(
            "========================================Start running========================================"
        )
        for recon_target in self.recon_targets:
            self.recon_target = recon_target
            target_save_dir = os.path.join(
                self.data_save_root, '%s_%s' % (self.experiment, self.recon_target)
            )
            os.makedirs(target_save_dir, exist_ok=True)
            recon_file_path = os.path.join(target_save_dir, 'ablation_results.csv')
            write_header = (
                not os.path.exists(recon_file_path) or os.path.getsize(recon_file_path) == 0
            )
            self.recon_file = open(recon_file_path, 'a+')
            self.recon_writer = csv.writer(self.recon_file)
            if write_header:
                cov = ['cov_%s' % step for step in range(self.max_allowed_steps)]
                curr_view = ['curr_hw_%s' % step for step in range(self.max_allowed_steps)]
                columns = (
                    ['planner', 'target', 'ckpt', 'plant_xyzr', 'test_num']
                    + cov
                    + ['auc']
                    + curr_view
                    + ['total_distance']
                    + ['step_time']
                )
                self.recon_writer.writerow(columns)

            for planner in self.planners:
                self.planner = planner
                self.config_env()
                self.environment.set_seed()

                for ckpt in self.ckpts:
                    self.ckpt = ckpt
                    self.create_agent()

                    average_auc = 0
                    for test_count in range(self.max_test):
                        print(
                            "===========Validating:{}-{}-{}-{}=========".format(
                                self.planner, self.recon_target, self.ckpt, test_count
                            )
                        )

                        coverage_list, auc, hw_list, traj_distance, step_time = self.valid_once()

                        # fill nan in the coverage list to have a length==self.max_step
                        coverage_list = coverage_list + [np.nan] * (
                            self.max_allowed_steps - len(coverage_list)
                        )
                        hw_list = hw_list + [np.nan] * (self.max_allowed_steps - len(hw_list))

                        self.recon_writer.writerow(
                            [
                                self.planner,
                                self.recon_target,
                                self.ckpt,
                                '%s_%s_%s_%s_%s'
                                % (
                                    self.environment.plant,
                                    self.environment.x,
                                    self.environment.y,
                                    self.environment.z,
                                    self.environment.rotation,
                                ),
                                str(test_count),
                                *coverage_list,
                                str(auc),
                                *hw_list,
                                traj_distance,
                                step_time,
                            ]
                        )
                        self.recon_file.flush()
                        average_auc += auc
                    print(
                        "Average AUC for {}-{}-{}:{}".format(
                            self.planner, self.recon_target, self.ckpt, average_auc / self.max_test
                        )
                    )
            self.recon_file.close()

    def euclidean_distance(self, xyz1, xyz2):
        xyz1 = np.array(xyz1)
        xyz2 = np.array(xyz2)
        return np.linalg.norm(xyz1 - xyz2)

    def valid_once(self):
        done = False
        step_time = 0

        current_acc, current_view_state, current_view_index, current_hw = (
            self.environment.reset_global(planner=self.planner)
        )
        self.viewpoints_1d = self.environment.viewpoints_1d

        coverage_list = []
        hw_list = []
        while not done:
            _, cr = self.environment.nbv_compute.cal_coverage(self.environment.pcd_acc)
            print('current_view_index', current_view_index, 'current_coverage:', cr)

            coverage_list.append(cr)
            hw_list.append(current_hw)

            time_start = time.time()
            _, nbv_index = self.agent_ssl_semantic_nbv.act(
                0,
                current_acc,
                current_view_state,
            )
            time_end = time.time()

            step_time += time_end - time_start

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
        for j in range(len(hw_list) - 1):
            xyz1 = self.environment.viewpoints[hw_list[j][0], hw_list[j][1], 0:3]
            xyz2 = self.environment.viewpoints[hw_list[j + 1][0], hw_list[j + 1][1], 0:3]
            distance = self.euclidean_distance(xyz1, xyz2)
            total_distance += distance
        return (
            coverage_list,
            np.trapz(np.array(coverage_list)) / (self.environment.max_step - 1),
            hw_list,
            total_distance,
            step_time / (self.environment.max_step - 1),
        )


if __name__ == "__main__":
    rospy.init_node("Evaluator_ablation")
    trainer = Evaluator_ablation()
    trainer.run()
