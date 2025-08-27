#!/usr/bin/env python3
import rospy
import time
import os
from view_sampling.view_sampler import ViewSampler
import numpy as np
from plant_creation.plant_spawner import PlantSpawner
from nbv_compute.nbv_computor import Nbvcomputor
from camera_control.camera_control import CameraControl
from camera_control.camera_pose_broadcaster import CameraPoseBroadcaster
from tqdm import tqdm


class DataCollection:
    def __init__(self):
        self.data_save_dir = (
            "/home/jianchao/ros_workspace/paper4_semantic_nbv/src/offline_experiments/materials/view_data_collection/%s"
            % (str(time.ctime()))
        )

        self.plant_spawner = PlantSpawner()
        self.plant_spawner.set_model_name("plant")
        self.nbv_compute = Nbvcomputor(octomap_resulotion = 0.003)
        self.camera_control = CameraControl()
        self.camera_pose_broadcast = CameraPoseBroadcaster()

        self.all_plants = [19]  ##select the plants for training
        ### for plane view sampling
        self.config_plane = {
            'center_x': -0.6, 'center_y': 0,
            'num_col': 9,
            'num_row': 9,
            'interval_h': 0.15,
            'interval_w': 0.15,
        }
    ### for cylinder view sampling
        self.config_cylinder = {
                'num_col': 10, 
                'num_row':10,
                'radius': 0.8,
                'interval_h': 0.20,
                'interval_ang': 20}

        self.x_range = np.arange(-20, 20 + 1, 10)
        self.y_range = np.arange(-20, 20 + 1, 10)
        self.z_range = np.arange(0, 20 + 1, 20)
        self.rotation_range = np.arange(0, 360, 90)

        ##data for normal data when block is not used
        # self.x_range = np.arange(-20, 20 + 1, 10)
        # self.y_range = np.arange(-20, 20 + 1, 10)
        # self.rotation_range = np.arange(0, 360, 90)

        self.pc_size = 512
        self.view_sampler = ViewSampler(
            distribution_type="plane"
        )  # plane, cylinder

    def resampel_pc(self, pcd, n):
        idx = np.random.permutation(pcd.shape[1])
        if idx.shape[0] < n:
            idx = np.concatenate(
                [idx, np.random.randint(pcd.shape[1], size=n - pcd.shape[1])]
            )
        return pcd[0, idx[:n]].unsqueeze(0)
    
    def get_viewpoints(self):
        viewpoints = self.view_sampler.sample_views(
                **self.config_plane
            )
        return viewpoints

    def save_data_views(
        self,
        plant,
        x,
        y,
        z,
        rotation,
        height=None,
        angle=None,
        pcd=None,
        depth=None,
        rgb=None,
        view_xyzxyzw=None,
        complete_model=False,
    ):
        if complete_model == False:
            pcd = pcd.cpu().numpy()
            plant_path = os.path.join(
                self.data_save_dir,
                "plant%s" % str(plant),
                "xyzr_%s_%s_%s_%s" % (str(x), str(y), str(z), str(rotation)),
            )

            if depth is not None:
                depth_path = os.path.join(plant_path, "depth") 
                if os.path.exists(depth_path) == False:
                    os.makedirs(depth_path)
                np.save(os.path.join(depth_path, "depth_%s_%s.npy" % (str(height), str(angle))),
                depth,
            )

            if rgb is not None:
                rgb_path = os.path.join(plant_path, "rgb")
                if os.path.exists(rgb_path) == False:
                    os.makedirs(rgb_path)
                np.save(os.path.join(rgb_path, "rgb_%s_%s.npy" % (str(height), str(angle))),
                rgb,
            )
            if view_xyzxyzw is not None:
                view_path = os.path.join(plant_path, "view")
                if os.path.exists(view_path) == False:
                    os.makedirs(view_path)
                np.save(os.path.join(view_path, "view_%s_%s.npy" % (str(height), str(angle))),
                view_xyzxyzw,
                )
            if pcd is not None:
                pcd_path = os.path.join(plant_path, "pcd")
                if os.path.exists(pcd_path) == False:
                    os.makedirs(pcd_path)
                np.save(os.path.join(pcd_path, "pcd_%s_%s.npy" % (str(height), str(angle))),
                pcd,
                )
        else:
            pcd = pcd.cpu().numpy()
            path = os.path.join(
                self.data_save_dir, "complete_model", "plant%s" % str(plant)
            )  # , 'xyr_%s_%s_%s'%(str(x), str(y), str(rotation)))
            if os.path.exists(path) == False:
                os.makedirs(path)
            np.save(
                os.path.join(
                    path, "complete_pcd_%s_%s_%s.npy" % (str(x), str(y), str(rotation))
                ),
                pcd,
            )

    def run(self):
        print(
            "========================================Start data collection========================================"
        )
        for plant in tqdm(self.all_plants):  ##select a plant
            self.plant_spawner.spawn_plant(plant, 0, [0, 0, 0])
            # self.plant_spawner.spawn_block()
            rospy.sleep(5.0)

            self.viewpoints = self.get_viewpoints()

            for x in self.x_range:  ##select a distance in x axis
                for y in self.y_range:  ##select a distance in y axis
                    for z in self.z_range:
                        for rotation in self.rotation_range:  ##select a rotation
                            self.plant_spawner.set_pose(
                                plant, rotation, [x, y, z]
                            )  ##spawn the plant model with pose
                            self.nbv_compute.load_complete_pc(
                                'plant',
                                plant,
                                recon_target = None,
                                plant_rotation=rotation,
                                plant_position=[x, y, z],
                                scale = 1,
                                visual_com = True,
                                visual_roi = True,
                            )
                            rospy.sleep(5.0)
                            # self.save_data_views(
                            #     plant,
                            #     x,
                            #     y,
                            #     rotation,
                            #     pcd=self.nbv_compute.complete_pc,
                            #     complete_model=True,
                            # )

                            for height in range(
                                self.viewpoints.shape[0]
                            ):  ##collect data from each viewpoints
                                for angle in range(self.viewpoints.shape[1]):
                                    self.view_sampler.visual_candidates(
                                        self.viewpoints, color=[1, 0, 0]
                                    )

                                    view = np.asarray([self.viewpoints[height, angle, :]])
                                    self.camera_control.move_camera_to_pose(view[0])
                                    self.camera_pose_broadcast.broadcast(view[0])
                                    rospy.sleep(10.0)
                                    color_image, depth_image, valid_xyzrgb, valid_xyz = self.nbv_compute.get_point_cloud(
                                        visual_pc=True
                                        )

                                    # valid_xyz = valid_xyzrgb[:, :, 0:3]
                                    self.save_data_views(
                                        plant,
                                        x,
                                        y,
                                        z,
                                        rotation,
                                        height,
                                        angle,
                                        pcd=valid_xyzrgb,
                                        # depth=depth_image,
                                        rgb=color_image,
                                        view_xyzxyzw=view[0],
                                        complete_model=False,
                                    )

            self.plant_spawner.delete_plants(plant)

if __name__ == "__main__":
    rospy.init_node("collect_nbv_data")
    trainer = DataCollection()
    trainer.run()
