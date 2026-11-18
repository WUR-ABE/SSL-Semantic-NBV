#!/usr/bin/env python3
"""Render the per-view point clouds used for training and evaluation in simulation.

For every plant, plant pose (x, y, z, rotation) and candidate viewpoint, the plant is
placed in Gazebo, the simulated L515 camera is moved to the viewpoint and the
resulting point cloud (world frame, xyz + rgb) is stored as

    <out_dir>/plant<i>/xyzr_<x>_<y>_<z>_<r>/pcd/pcd_<row>_<col>.npy   (1, N, 6) float32
    <out_dir>/plant<i>/xyzr_<x>_<y>_<z>_<r>/view/view_<row>_<col>.npy (7,) camera pose xyz + quat(xyzw)

which is the layout read by drl_environment/environment.py.

Settings (private ROS param `~name`, or environment variable `NAME`):
    experiment        0 = normal plants (1-10), 1 = heavy-occlusion plants (11-20)   [0]
    plants            comma-separated plant ids, overrides `experiment`
    out_dir           output folder                     [$NBV_DATA_ROOT/simulation/views]
    x_range, y_range  plant offsets in cm               [-20,-10,0,10,20]
    z_range           plant offsets in cm               [0,20]
    rotations         plant rotations about z           [0,90,180,270]
    view_settle_time  seconds to wait after moving the camera            [10.0]
    pose_settle_time  seconds to wait after moving the plant             [5.0]
    save_images       also store raw rgb/ and depth/ images              [false]
    skip_existing     skip views whose pcd file already exists          [true]
"""

import os
import time

import numpy as np
import rospy
from tqdm import tqdm

from camera_control.camera_control import CameraControl
from camera_control.camera_pose_broadcaster import CameraPoseBroadcaster
from nbv_compute.nbv_computor import Nbvcomputor
from plant_creation.plant_spawner import PlantSpawner
from utils.config import get_param
from utils.paths import VIEW_DATA_DIR
from view_sampling.view_sampler import ViewSampler

PLANTS_PER_EXPERIMENT = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}


class DataCollection:
    def __init__(self):
        experiment = get_param("experiment", 0)
        self.all_plants = get_param("plants", PLANTS_PER_EXPERIMENT[experiment])
        self.data_save_dir = os.path.expanduser(get_param("out_dir", VIEW_DATA_DIR))

        # Must match `config_middle` / the pose ranges in drl_environment/environment.py.
        self.config_plane = {
            "center_x": -0.6,
            "center_y": 0,
            "num_col": 9,
            "num_row": 9,
            "interval_h": 0.15,
            "interval_w": 0.15,
        }
        self.x_range = get_param("x_range", [-20, -10, 0, 10, 20])
        self.y_range = get_param("y_range", [-20, -10, 0, 10, 20])
        self.z_range = get_param("z_range", [0, 20])
        self.rotation_range = get_param("rotations", [0, 90, 180, 270])

        self.view_settle_time = get_param("view_settle_time", 10.0)
        self.pose_settle_time = get_param("pose_settle_time", 5.0)
        self.save_images = get_param("save_images", False)
        self.skip_existing = get_param("skip_existing", True)

        self.plant_spawner = PlantSpawner()
        self.plant_spawner.set_model_name("plant")
        self.nbv_compute = Nbvcomputor(octomap_resulotion=0.003)
        self.camera_control = CameraControl()
        self.camera_pose_broadcast = CameraPoseBroadcaster()
        self.view_sampler = ViewSampler(distribution_type="plane")

    def get_viewpoints(self):
        return self.view_sampler.sample_views(**self.config_plane)

    def pose_dir(self, plant, x, y, z, rotation):
        return os.path.join(
            self.data_save_dir,
            "plant%s" % plant,
            "xyzr_%s_%s_%s_%s" % (x, y, z, rotation),
        )

    @staticmethod
    def _save(folder, name, array):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, name), array)

    def save_view(self, pose_dir, row, col, pcd, view, rgb=None, depth=None):
        suffix = "%s_%s.npy" % (row, col)
        self._save(os.path.join(pose_dir, "pcd"), "pcd_" + suffix, pcd)
        self._save(os.path.join(pose_dir, "view"), "view_" + suffix, view)
        if rgb is not None:
            self._save(os.path.join(pose_dir, "rgb"), "rgb_" + suffix, rgb)
        if depth is not None:
            self._save(os.path.join(pose_dir, "depth"), "depth_" + suffix, depth)

    def wait_for_camera(self, timeout=60.0):
        # wall-clock timeout: Gazebo runs faster than real time (sim time is not a good measure here)
        start = time.time()
        while not rospy.is_shutdown():
            nc = self.nbv_compute
            if (
                nc.color_image is not None
                and nc.depth_image is not None
                and nc.camera_info is not None
            ):
                return
            if time.time() - start > timeout:
                raise RuntimeError(
                    "No camera images received within %.0f s. Is Gazebo running with the camera "
                    "(roslaunch offline_experiments collect_view_data.launch)?" % timeout
                )
            time.sleep(0.5)

    def run(self):
        rospy.loginfo("[DataCollection] plants=%s -> %s", self.all_plants, self.data_save_dir)
        viewpoints = self.get_viewpoints()
        self.wait_for_camera()

        for plant in tqdm(self.all_plants, desc="plants"):
            self.plant_spawner.spawn_plant(plant, 0, [0, 0, 0])
            rospy.sleep(self.pose_settle_time)

            for x in self.x_range:
                for y in self.y_range:
                    for z in self.z_range:
                        for rotation in self.rotation_range:
                            if rospy.is_shutdown():
                                return
                            pose_dir = self.pose_dir(plant, x, y, z, rotation)
                            todo = [
                                (r, c)
                                for r in range(viewpoints.shape[0])
                                for c in range(viewpoints.shape[1])
                                if not (
                                    self.skip_existing
                                    and os.path.exists(
                                        os.path.join(pose_dir, "pcd", "pcd_%s_%s.npy" % (r, c))
                                    )
                                )
                            ]
                            if not todo:
                                continue

                            # NOTE: `rotation` is passed as-is (in the same unit) to both Gazebo
                            # and load_complete_pc(), so the rendered views and the ground truth
                            # used during training stay consistent.
                            self.plant_spawner.set_pose(plant, rotation, [x, y, z])
                            self.nbv_compute.load_complete_pc(
                                "plant",
                                plant,
                                recon_target=None,
                                plant_rotation=rotation,
                                plant_position=[x, y, z],
                                scale=1,
                                visual_com=True,
                                visual_roi=True,
                            )
                            rospy.sleep(self.pose_settle_time)

                            for row, col in todo:
                                view = np.asarray(viewpoints[row, col, :])
                                self.view_sampler.visual_candidates(viewpoints, color=[1, 0, 0])
                                self.camera_control.move_camera_to_pose(view)
                                self.camera_pose_broadcast.broadcast(view)
                                rospy.sleep(self.view_settle_time)
                                color_image, depth_image, valid_xyzrgb, _ = (
                                    self.nbv_compute.get_point_cloud(visual_pc=True)
                                )
                                self.save_view(
                                    pose_dir,
                                    row,
                                    col,
                                    pcd=valid_xyzrgb.cpu().numpy().astype(np.float32),
                                    view=view,
                                    rgb=color_image if self.save_images else None,
                                    depth=depth_image if self.save_images else None,
                                )

            self.plant_spawner.delete_plants(plant)


if __name__ == "__main__":
    rospy.init_node("collect_view_data")
    DataCollection().run()
