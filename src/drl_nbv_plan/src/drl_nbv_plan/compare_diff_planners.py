#!/usr/bin/env python3
"""Step 4 of the pipeline: compare NBV planners in simulation.

Writes results.csv (coverage per step, AUC, trajectory length, planning time) to `out_dir`.

Settings (private ROS param `~name` or environment variable `NAME`):
    planners      comma-separated                 [ssl_semantic_nbv_NOCOLOR,ssl_nbv,predefined_global,random_global]
                  (also available: voxel_global_nbv)
    experiment    0 = normal plants, 1 = heavy-occlusion plants                [0]
    recon_target  PLANT, FRUIT or NODE                                         [PLANT]
    max_test      number of test episodes per planner                          [25]
    max_step      views per episode                                            [15]
    max_allowed_step  number of coverage columns in the csv                    [15]
    models_dir    folder with <experiment>_<target>_<planner>/ checkpoints     [$NBV_MODELS]
    ssl_semantic_nbv_ckpt  checkpoint file   [<models_dir>/<exp>_<target>_<planner>/<planner>_model_epi_2000]
    ssl_nbv_ckpt           checkpoint file   [<models_dir>/<exp>_<target>_ssl_nbv/ssl_nbv_best_model]
    out_dir       output folder   [$NBV_MATERIALS/results/compare_diff_planners/<timestamp>]
"""

from utils.config import get_param
from utils.paths import MODELS_DIR, model_dir, results_dir
from drl_environment.environment import Environment
import rospy
import time
import os
from agent import AgentGlobal, AgentGlobalWork1
import csv
import numpy as np
from baseline_planners import GlobalRandom, PredefinedGlobal
from active_vision.volumetric_planner import ActivePlanner
import torch


class Evaluator_compare_planners:
    def __init__(self):
        self.planners = get_param(
            "planners",
            ["ssl_semantic_nbv_NOCOLOR", "ssl_nbv", "predefined_global", "random_global"],
        )
        self.max_test = get_param("max_test", 25)
        self.max_allowed_step = get_param("max_allowed_step", 15)
        self.data_save_dir = os.path.expanduser(
            get_param("out_dir", results_dir("compare_diff_planners"))
        )
        os.makedirs(self.data_save_dir, exist_ok=True)
        self.total_iteration_count = 0

        self.recon_file = open(os.path.join(self.data_save_dir, 'results.csv'), 'a+')
        self.recon_writer = csv.writer(self.recon_file)
        cov = ['cov_%s' % step for step in range(self.max_allowed_step)]
        curr_view = ['curr_hw_%s' % step for step in range(self.max_allowed_step)]
        columns = (
            ['planner', 'valid_modules', 'plant_xyzr', 'test_num']
            + cov
            + ['auc']
            + curr_view
            + ['total_distance']
            + ['step_time']
        )
        self.recon_writer.writerow(columns)

    @staticmethod
    def internal_name(planner):
        # all SSL-Semantic-NBV variants (e.g. ssl_semantic_nbv_NOCOLOR) share the same code path
        return 'ssl_semantic_nbv' if planner.startswith('ssl_semantic_nbv') else planner

    def update_mode(self, mode):
        for agent in (self.agent_ssl_semantic_nbv, self.agent_ssl_nbv):
            if agent is not None:
                agent.update_mode(mode=mode)
        self.environment.update_mode(mode=mode)

    def config_env_model(self):
        experiment = get_param("experiment", 0)
        recon_target = get_param("recon_target", "PLANT")
        self.config = {
            "pc_size": 512,
            'experiment': experiment,
            'octo_resolution': 0.003,
            'planners': self.planners,
            'recon_target': recon_target,  #'PLANT','FRUIT','NODE'
            'max_step': get_param("max_step", 15),
        }

        self.environment = Environment(self.config)
        num_views = (
            self.environment.config_middle['num_col'] * self.environment.config_middle['num_row']
        )
        models_dir = os.path.expanduser(get_param("models_dir", MODELS_DIR))

        self.agent_ssl_semantic_nbv = None
        semantic_variants = [p for p in self.planners if p.startswith('ssl_semantic_nbv')]
        if semantic_variants:
            variant = semantic_variants[
                0
            ]  # e.g. ssl_semantic_nbv_NOCOLOR; defines the network input
            ckpt = get_param(
                "ssl_semantic_nbv_ckpt",
                os.path.join(
                    model_dir(experiment, recon_target, variant, models_dir),
                    '%s_model_epi_2000' % variant,
                ),
            )
            print("Loading %s from %s" % (variant, ckpt))
            self.agent_ssl_semantic_nbv = AgentGlobal(num_views, planner=variant)
            self.agent_ssl_semantic_nbv.config_evaluate_only()
            self.agent_ssl_semantic_nbv.load_checkpoint(ckpt)

        self.agent_ssl_nbv = None
        if 'ssl_nbv' in self.planners:
            ckpt = get_param(
                "ssl_nbv_ckpt",
                os.path.join(
                    model_dir(experiment, recon_target, 'ssl_nbv', models_dir), 'ssl_nbv_best_model'
                ),
            )
            print("Loading ssl_nbv from %s" % ckpt)
            self.agent_ssl_nbv = AgentGlobalWork1(num_views)
            self.agent_ssl_nbv.config_evaluate_only()
            self.agent_ssl_nbv.load_checkpoint(ckpt)

        self.update_mode(mode="valid")

        self.globalrandom = GlobalRandom()
        self.predefinedglobal = PredefinedGlobal()
        self.voxel_global = ActivePlanner()

    def run(self):
        print(
            "========================================Start running========================================"
        )
        self.config_env_model()
        print(
            "Experiment: %s | Target: %s | Results: %s"
            % (self.config['experiment'], self.config['recon_target'], self.data_save_dir)
        )
        for planner in self.planners:
            self.planner = self.internal_name(planner)
            self.environment.set_seed()

            average_auc = 0
            for test_count in range(self.max_test):
                print("===========Validating:{}-{}=========".format(planner, test_count))
                coverage_list, auc, hw_list, traj_distance, step_time = self.valid_once()

                # fill nan in the coverage list to have a length==self.max_step
                coverage_list = coverage_list + [np.nan] * (
                    self.max_allowed_step - len(coverage_list)
                )
                hw_list = hw_list + [np.nan] * (self.max_allowed_step - len(hw_list))

                self.recon_writer.writerow(
                    [
                        planner,
                        'recurrent_positionemb',
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
            print("Average AUC for {}:{}".format(planner, average_auc / self.max_test))

    def euclidean_distance(self, xyz1, xyz2):
        xyz1 = np.array(xyz1)
        xyz2 = np.array(xyz2)
        return np.linalg.norm(xyz1 - xyz2)

    def valid_once(self):
        predefined_views = np.array([10, 13, 16, 22, 28, 31, 34, 40, 46, 49, 52, 58, 64, 67, 70])

        done = False
        recurrent_state = None
        step_time = 0

        if (
            self.planner == 'ssl_semantic_nbv'
            or self.planner == 'voxel_global_nbv'
            or self.planner == 'random_global'
            or self.planner == 'ssl_nbv'
            or self.planner == 'predefined_global'
        ):
            current_acc, current_view_state, current_view_index, current_hw = (
                self.environment.reset_global(
                    planner=self.planner, predefined_views=predefined_views
                )
            )
            self.viewpoints_1d = self.environment.viewpoints_1d

            if self.planner == 'voxel_global_nbv':
                grid_size, grid_center = self.environment.get_grid_info()
                self.viewpoints_1d = self.viewpoints_1d[
                    ..., [0, 1, 2, 6, 3, 4, 5]
                ]  # change to xyzwrpy
                self.voxel_global.init_voxel_grid(
                    grid_size=grid_size,
                    voxel_size=np.array([self.config['octo_resolution']]),
                    grid_center=grid_center,
                )

                view_xyzwrpy = self.viewpoints_1d[0, current_view_index, :]
                pcd = self.environment.pcd_list[current_view_index][..., 0:3]
                pcd = self.environment.pcd_list[current_view_index]
                pcd_target = self.environment.nbv_compute.classify_points_use_bbox(
                    pcd, self.config['recon_target']
                )
                pcd_node = (
                    self.environment.nbv_compute.classify_points_use_bbox(pcd, 'NODE').cpu().numpy()
                )
                pcd_node = pcd_node[pcd_node[..., -1] == 1, 0:3]
                current_node_bbox = self.environment.update_roi_bbox(pcd_node)
                depth_image, semantics_image = self.voxel_global.convert_pc2depth(
                    pcd_target, view_xyzwrpy, mode='xyzrgbc'
                )
                self.voxel_global.depth_to_voxels(
                    depth_image, torch.tensor(semantics_image), view_xyzwrpy
                )
                self.voxel_global.visualize()
        else:
            current_acc, current_view, current_hw = self.environment.reset(planner=self.planner)
            self.viewpoints = self.environment.viewpoints

            if self.planner == 'voxel_local_nbv':
                self.viewpoints = self.viewpoints[..., [0, 1, 2, 6, 3, 4, 5]]  # change to xyzwrpy
                grid_size, grid_center = self.environment.get_grid_info()
                self.voxel_local.init_voxel_grid(
                    grid_size=grid_size,
                    voxel_size=np.array([self.config['octo_resolution']]),
                    grid_center=grid_center,
                )

                view_xyzwrpy = self.viewpoints[current_hw[0], current_hw[1], :]
                depth_image = self.voxel_local.convert_pc2depth(
                    self.environment.pcd_acc[..., 0:3], view_xyzwrpy
                )
                self.voxel_local.depth_to_voxels(
                    depth_image,
                    torch.ones_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1, 1, 2),
                    view_xyzwrpy,
                )

        coverage_list = []
        hw_list = []
        while not done:
            _, cr = self.environment.nbv_compute.cal_coverage(self.environment.pcd_acc)
            print('current_view_index', current_view_index, 'current_coverage:', cr)

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
                predefined_views = np.delete(
                    predefined_views, np.where(predefined_views == current_view_index)
                )
                time_start = time.time()

                if len(predefined_views) > 0:
                    nbv_index = self.predefinedglobal.act(predefined_views)
                else:
                    nbv_index = current_view_index

                time_end = time.time()
                recurrent_state_ = None

            elif self.planner == 'localgt':
                in_bound_actions, in_bound_pcds = self.environment.get_inbound_pcds(current_hw)
                time_start = time.time()
                action, max_ig = self.localgt.act(
                    in_bound_actions, in_bound_pcds, self.environment.pcd_acc
                )
                time_end = time.time()
                recurrent_state_ = None

            elif self.planner == 'voxel_local_nbv':
                time_start = time.time()
                in_bound_actions, in_bound_viewpoints = self.environment.get_inbound_actions(
                    current_hw
                )
                _, nbv, gain, _ = self.voxel_local.active_view(in_bound_viewpoints)
                action = in_bound_actions[nbv]
                time_end = time.time()
                recurrent_state_ = None

                nbv_xyzwrpy = in_bound_viewpoints[nbv]
                nbv_pc = self.environment.get_nbv_pc(action, current_hw)[..., 0:3]
                depth_image = self.voxel_local.convert_pc2depth(nbv_pc, nbv_xyzwrpy)

                self.voxel_local.depth_to_voxels(
                    depth_image,
                    torch.empty_like(torch.tensor(depth_image)).unsqueeze(-1).repeat(1, 1, 2),
                    nbv_xyzwrpy,
                )

            elif self.planner == 'voxel_global_nbv':
                self.voxel_global.voxel_grid.crop_grid(current_node_bbox)
                time_start = time.time()
                _, nbv_index, gain, _ = self.voxel_global.active_view(self.viewpoints_1d[0])
                time_end = time.time()
                nbv_xyzwrpy = self.viewpoints_1d[0, nbv_index, :]
                nbv_pc = self.environment.pcd_list[nbv_index]
                nbv_pc_target = self.environment.nbv_compute.classify_points_use_bbox(
                    nbv_pc, self.config['recon_target']
                )
                pcd_node = (
                    self.environment.nbv_compute.classify_points_use_bbox(nbv_pc, 'NODE')
                    .cpu()
                    .numpy()
                )
                pcd_node = pcd_node[pcd_node[..., -1] == 1, 0:3]
                current_node_bbox = self.environment.update_roi_bbox(pcd_node)
                depth_image, semantics_image = self.voxel_global.convert_pc2depth(
                    nbv_pc_target, nbv_xyzwrpy, mode='xyzrgbc'
                )
                self.voxel_global.depth_to_voxels(
                    depth_image, torch.tensor(semantics_image), nbv_xyzwrpy
                )
                self.voxel_global.visualize()

                recurrent_state_ = None

            step_time += time_end - time_start

            if (
                self.planner == 'ssl_semantic_nbv'
                or self.planner == 'voxel_global_nbv'
                or self.planner == 'random_global'
                or self.planner == 'ssl_nbv'
                or self.planner == 'predefined_global'
            ):
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

            if (
                self.planner == 'ssl_semantic_nbv'
                or self.planner == 'voxel_global_nbv'
                or self.planner == 'random_global'
                or self.planner == 'ssl_nbv'
                or self.planner == 'predefined_global'
            ):
                current_view_state = next_view_state
                current_view_index = nbv_index
            else:
                current_view = next_view

            recurrent_state = recurrent_state_
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
    rospy.init_node("Evaluator_SS1")
    trainer = Evaluator_compare_planners()
    trainer.run()
