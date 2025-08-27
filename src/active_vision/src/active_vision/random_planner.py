import torch
import numpy as np

from viewpoint_sampler.viewpoint_sampler import ViewpointSampler
from active_vision.voxel_grid import VoxelGrid
from utils.rviz_visualizer import RvizVisualizer
from utils.utils import numpy_to_pose, numpy_to_pose_array


class RandomPlanner:
    """
    Class to plan random viewpoints
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
        num_samples: int = 10,
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.random_params(start_pose, target_params)
        self.voxel_grid = VoxelGrid(
            grid_size=grid_size,
            voxel_size=voxel_size,
            grid_center=grid_center,
            target_params=self.target_params,
            image_size=image_size,
            intrinsics=intrinsics,
            num_pts_per_ray=num_pts_per_ray,
            num_features=num_features,
            device=self.device,
            save_path=save_path,
        )
        self.num_samples = num_samples
        self.view_sampler = ViewpointSampler(num_samples)
        self.viewpoint = start_pose
        self.target_position = target_params
        self.rviz_visualizer = RvizVisualizer()

    def random_params(self, start_pose: np.array, target_params: np.array) -> None:
        self.target_params = torch.tensor(
            target_params,
            dtype=torch.float32,
            device=self.device,
        )
        # self.camera_bounds = np.array([[0.5, -0.35, 1.0], [0.7, 0.35, 1.3]])
        self.camera_bounds = np.array(
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
            ]
        )

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

    def random_view(self, seed=None) -> np.array:
        """
        Sample a viewpoint at random
        """
        view_samples = self.view_sampler.random_neighbour_sampler(
            self.viewpoint[:3],
            self.target_position,
            camera_limits=self.camera_bounds[:, :3],
            target_limits=self.camera_bounds[:, 3:],
            seed=seed,
        )
        self.rviz_visualizer.visualize_view_samples(
            numpy_to_pose_array(view_samples[:, :7])
        )
        random_index = np.random.randint(self.num_samples)
        viewpoint = view_samples[random_index, :7]
        self.target_position = view_samples[random_index, 7:]
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.viewpoint = viewpoint
        return self.viewpoint

    def visualize(self):
        """
        Visualize the voxel grid as a point cloud in rviz
        """
        voxel_points, semantic_points = self.voxel_grid.get_occupied_points()
        voxel_points = voxel_points.cpu().numpy()
        semantic_points = semantic_points.cpu().numpy()
        self.rviz_visualizer.visualize_voxels(voxel_points, semantic_points)
        self.rviz_visualizer.visualize_camera_bounds(self.camera_bounds[:, :3])
