import torch
import torch.nn as nn
import numpy as np
import time

from active_vision.voxel_grid import VoxelGrid
from utils.rviz_visualizer import RvizVisualizer
from utils.utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import look_at_rotation


class OptimalPlanner(nn.Module):
    """
    Class to plan a locally optimal viewpoint
    """

    def __init__(
        self,
        start_pose: np.array,
        grid_size: np.array = np.array([0.3, 0.3, 0.7]),
        voxel_size: np.array = np.array([0.002]),
        grid_center: np.array = np.array([0.9, 0.0, 1.15]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array(
            [
                [685.5028076171875, 0.0, 485.35955810546875],
                [0.0, 685.6409912109375, 270.7330627441406],
                [0.0, 0.0, 1.0],
            ],
        ),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        save_path: str = None,
        target_params: np.array = np.array([0.9, 0.0, 1.15]),
    ) -> None:
        """
        Initialize the planner
        :param grid_size: size of the voxel grid in meters
        :param voxel_size: size of the voxels in meters
        :param grid_center: center of the voxel grid in meters
        :param image_size: size of the image in pixels
        :param num_pts_per_ray: number of points sampled per ray
        :param num_features: number of features per voxel
        """
        super(OptimalPlanner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.optimization_params(start_pose, target_params)
        self.voxel_grid = VoxelGrid(
            grid_size=grid_size,
            voxel_size=voxel_size,
            grid_center=grid_center,
            # target_params=None,
            target_params=self.target_params,
            # target_params=self.camera_params[3:],
            image_size=image_size,
            intrinsics=intrinsics,
            num_pts_per_ray=num_pts_per_ray,
            num_features=num_features,
            device=self.device,
            save_path=save_path,
        )
        self.rviz_visualizer = RvizVisualizer()
        self.num_samples = num_samples

    def optimization_params(
        self, start_pose: np.array, target_params: np.array
    ) -> None:
        """
        Initialize the optimization parameters
        """
        self.camera_params = nn.Parameter(
            torch.tensor(
                # [start_pose[0], start_pose[1], start_pose[2]],
                # [start_pose[0], start_pose[1], start_pose[2], 0.9, 0.0, 1.15],
                [
                    start_pose[0],
                    start_pose[1],
                    start_pose[2],
                    target_params[0],
                    target_params[1],
                    target_params[2],
                ],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        )
        self.target_params = torch.tensor(
            target_params,
            dtype=torch.float32,
            device=self.device,
        )
        self.camera_bounds = torch.tensor(
            # [[0.5, -0.35, 1.0], [0.7, 0.35, 1.3]],
            # [[0.5, -0.35, 1.0, 0.85, -0.35, 1.0], [0.7, 0.35, 1.3, 0.95, 0.35, 1.3]],
            # [[0.5, -0.35, 1.0, 0.85, -0.35, 1.0], [0.7, 0.35, 1.3, 0.95, 0.35, 1.3]],
            [
                [
                    start_pose[0] - 0.1,
                    start_pose[1] - 0.05,
                    start_pose[2] - 0.1,
                    target_params[0] - 0.1,
                    target_params[1] - 0.05,
                    target_params[2] - 0.1,
                ],
                [
                    start_pose[0] + 0.1,
                    start_pose[1] + 0.1,
                    start_pose[2] + 0.1,
                    target_params[0] + 0.1,
                    target_params[1] + 0.1,
                    target_params[2] + 0.1,
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.025)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer,
        #     base_lr=0.005,
        #     max_lr=0.05,
        #     step_size_up=3,
        #     cycle_momentum=False,
        # )
        torch.autograd.set_detect_anomaly(False)

    def depth_to_voxels(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth image and insert it into a voxel grid
        :param depth_image: depth image (H, W)
        :param semantics: confidence scores and class ids (H, W, 2)
        :param viewpoint: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(
            viewpoint[3:], dtype=torch.float32, device=self.device
        )
        coverage = self.voxel_grid.insert_point_cloud(
            depth_image, semantics, position, orientation
        )
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        return coverage

    def loss(self) -> torch.tensor:
        """
        Compute the loss for the current viewpoint
        :return: loss
        """
        # loss = self.voxel_grid.compute_gain(self.camera_params, self.target_params)
        # t0 = time.time()
        loss = self.voxel_grid.compute_gain(
            self.camera_params[:3], self.camera_params[3:]
        )
        # print("gain time", time.time() - t0)
        return loss

    def next_best_view(self) -> np.array:
        """
        Compute the next best viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        counter = 0
        prev_loss = 0.0
        for _ in range(self.num_samples):
            self.optimizer.zero_grad()
            loss, semantic_gain, weighted_mean = self.loss()
            counter += 1
            loss.backward()
            diff = abs(prev_loss - loss.item())
            prev_loss = loss.item()
            if diff < 1e-3:
                break
            # print("loss: ", loss.item())
            # print("camera params gradients: ", self.camera_params.grad)
            self.optimizer.step()
            # print("camera params: ", self.camera_params.data)
            self.camera_params.data = torch.clamp(
                self.camera_params.data, self.camera_bounds[0], self.camera_bounds[1]
            )
            # self.scheduler.step()

        weighted_mean = weighted_mean.detach().cpu().numpy()
        self.rviz_visualizer.visualize_semantic_mean(weighted_mean)

        viewpoint = self.get_viewpoint()
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        loss = loss.detach().cpu().numpy()
        return viewpoint, loss, counter

    def get_viewpoint(self) -> np.array:
        """
        Get the current viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        # quat = look_at_rotation(self.camera_params, self.target_params)
        quat = look_at_rotation(self.camera_params[:3], self.camera_params[3:])
        quat = quat.detach().cpu().numpy()
        viewpoint = np.zeros(7)
        viewpoint[:3] = self.camera_params.detach().cpu().numpy()[:3]
        viewpoint[3:] = quat
        return viewpoint

    def visualize(self):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        voxel_points, semantic_points = self.voxel_grid.get_occupied_points()
        voxel_points = voxel_points.cpu().numpy()
        semantic_points = semantic_points.cpu().numpy()
        self.rviz_visualizer.visualize_voxels(voxel_points, semantic_points)
        # Visualize target
        # target = self.target_params.cpu().numpy()
        target = self.camera_params.detach().cpu().numpy()[3:]
        rois = np.array([[*target, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        # Visualize camera bounds
        # camera_bounds = self.camera_bounds.cpu().numpy()
        camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds)
