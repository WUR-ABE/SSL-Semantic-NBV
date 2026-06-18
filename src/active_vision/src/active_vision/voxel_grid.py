import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pytransform3d import rotations as py3dr
from active_vision.ray_sampler import RaySampler
from utils.torch_utils import (
    look_at_rotation,
    transform_from_rotation_translation,
    # unravel_index,
    # gaussian_kernel,
)

# from active_vision.clustering import Clusterer


class VoxelGrid:
    """
    3D representation to store occupancy information and other features (e.g. semantics) over
    multiple viewpoints
    """

    def __init__(
        self,
        grid_size: torch.tensor,
        voxel_size: torch.tensor,
        grid_center: torch.tensor,
        target_params: torch.tensor = None,
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
        eps: torch.float32 = 1e-7,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        save_path: str = None,
    ) -> None:
        """
        :param eps: epsilon value for numerical stability
        """
        self.device = device
        self.grid_size = grid_size
        self.dims = (grid_size / voxel_size).to(torch.long)
        self.voxel_size = voxel_size
        self.grid_center = grid_center
        self.origin = grid_center - grid_size / 2.0
        self.min_bound = self.origin
        self.max_bound = self.origin + grid_size
        self.eps = eps
        self.save_path = save_path

        # rois, occ_prob, sem_conf, sem_class
        self.grid = torch.zeros(
            (self.dims[0], self.dims[1], self.dims[2], num_features),
            dtype=torch.float32,
            device=self.device,
        )
        self.grid[..., 1] = 0.4  # 0.1 # weight for occupancy
        self.grid[..., 2] = 0.4  # 1e-7 # weight for semantic
        self.grid[..., 3] = -1 # weight for class

        # grid coordinates
        self.grid_coords = torch.meshgrid(
            [
                torch.arange(
                    0.0, self.dims[0], device=self.device, dtype=torch.float32
                ).to(self.device),
                torch.arange(
                    0.0, self.dims[1], device=self.device, dtype=torch.float32
                ).to(self.device),
                torch.arange(
                    0.0, self.dims[2], device=self.device, dtype=torch.float32
                ).to(self.device),
            ]
        )
        self.grid_coords = torch.stack(self.grid_coords, dim=-1).view(-1, 3)

        # # Define region of interest around the target
        # self.target_bounds = None
        # if target_params is not None:
        #     # print("Target params: ", target_params)
        #     # print("Origin: ", self.origin)
        #     target_coords = torch.div(
        #         target_params - self.origin, self.voxel_size, rounding_mode="floor"
        #     ).to(torch.long)
        #     # print("Target coords: ", target_coords)
        #     x_min = torch.clamp(target_coords[0] - 15, 0, self.dims[0])
        #     x_max = torch.clamp(target_coords[0] + 15, 0, self.dims[0])
        #     y_min = torch.clamp(target_coords[1] - 15, 0, self.dims[1])
        #     y_max = torch.clamp(target_coords[1] + 15, 0, self.dims[1])
        #     z_min = torch.clamp(target_coords[2] - 15, 0, self.dims[2])
        #     z_max = torch.clamp(target_coords[2] + 15, 0, self.dims[2])
        #     self.grid[x_min:x_max, y_min:y_max, z_min:z_max, 1:3] = 0.5
        #     self.target_bounds = torch.tensor(
        #         [x_min, y_min, z_min, x_max, y_max, z_max], device=self.device
        #     )
        # self.grid[..., 1:3] = torch.clamp(self.grid[..., 1:3], self.eps, 1.0 - self.eps)

        self.image_size = image_size
        self.num_pts_per_ray = num_pts_per_ray
        self.t_vals = torch.linspace(
            0.0,
            1.0,
            self.num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )

        ray_occ = -1.4 * torch.ones(
            (self.num_pts_per_ray, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.ray_occ = ray_occ.unsqueeze(0).repeat(
            self.image_size[0] * self.image_size[1], 1, 1
        )  # (W x H, num_pts_per_ray, 1)

        ray_sem_conf = -1.4 * torch.ones(
            num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )
        ray_sem_cls = -1 * torch.ones(
            num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )
        ray_sem = torch.stack((ray_sem_conf, ray_sem_cls), dim=-1)
        self.ray_sem = ray_sem.unsqueeze(0).repeat(image_size[0] * image_size[1], 1, 1)
        self.ray_sampler = RaySampler(image_size, intrinsics, self.device)

        # self.kernel_size = 21
        # self.kernel = gaussian_kernel(self.kernel_size, 1.0, device)

        # self.clusterer = Clusterer()
        self.counter = 0

        # self.set_target_roi(target_params)


    def insert_point_cloud(
        self,
        depth_image: np.array,
        semantics: torch.tensor,
        position: torch.tensor,
        orientation: torch.tensor,
        eps: torch.float32 = 1e-7,
    ) -> None:
        """
        Insert a point cloud into the voxel grid
        :param depth_image: depth image from the current viewpoint (W x H)
        :param semantics: semantic confidences and labels from the current viewpoint (2 x W x H)
        :param position: position of the current viewpoint (3,)
        :param orientation: orientation of the current viewpoint (4,)
        :return: None
        """
        # TODO: There should only be one sharp transition from occupied to free along each ray.
        # Check if there is already an occupied voxel along the ray
        transforms = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        # Convert depth image to point cloud
        (
            ray_origins,
            ray_directions,
            points_mask,
        ) = self.ray_sampler.ray_origins_directions(depth_image, transforms)
        ray_points = (
            ray_directions[:, :, None, :] * self.t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)
        # Convert point cloud to voxel grid coordinates
        grid_coords = torch.div(
            ray_points - self.origin, self.voxel_size, rounding_mode="floor"
        )
        valid_indices = self.get_valid_indices(grid_coords, self.dims)
        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)
        # Update the voxel grid
        log_odds = torch.log(
            torch.div(self.grid[gx, gy, gz, 1:3], 1.0 - self.grid[gx, gy, gz, 1:3])
        )
        # Occupancy information
        ray_occ = self.ray_occ.clone()
        ray_occ[:, -1, :] = points_mask.permute(1, 0)
        log_odds[..., 0] += ray_occ.view(-1, 1)[valid_indices, -1]
        # Semantic information
        ray_sem = self.ray_sem.clone()
        ray_sem[..., -1, :] = semantics.view(-1, 2)
        log_odds[..., 1] += ray_sem.view(-1, 2)[valid_indices, 0]
        # Update
        odds = torch.exp(log_odds)
        self.grid[gx, gy, gz, 1:3] = torch.div(odds, 1.0 + odds)
        self.grid[gx, gy, gz, 3] = self.ray_sem.view(-1, 2)[valid_indices, 1]
        self.grid[..., 1:3] = torch.clamp(self.grid[..., 1:3], self.eps, 1.0 - self.eps)

        # # Check the values within the target bounds and count the number of updated voxels
        # if self.target_bounds is not None:
        #     target_voxels = self.grid[
        #         self.target_bounds[0] : self.target_bounds[3],
        #         self.target_bounds[1] : self.target_bounds[4],
        #         self.target_bounds[2] : self.target_bounds[5],
        #         2,
        #     ]
        #     coverage = torch.sum((target_voxels != 0.5)) / target_voxels.numel() * 100
        #     # print("target_voxels:", self.target_bounds)
        #     return coverage
        # else:
        #     print("target_bounds is None")

    def compute_gain(
        self,
        camera_params: torch.tensor,
        target_params: torch.tensor,
    ) -> torch.tensor:
        """
        Compute the gain for a given set of parameters
        :param camera_params: camera parameters
        :param target_params: target parameters
        :param current_params: current parameters
        :return: total gain for the viewpoint defined by the parameters
        """
        quat = look_at_rotation(camera_params, target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_params[None, :]
        )
        self.set_target_roi(target_params)
        # Compute point cloud by ray-tracing along ray origins and directions
        t_vals = self.t_vals.clone().requires_grad_()
        ray_origins, ray_directions, _ = self.ray_sampler.ray_origins_directions(
            transforms=transforms
        )
        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)
        ray_points_nor = self.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        # Sample the occupancy probabilities and semantic confidences along each ray
        grid = self.grid[None, ..., 1:3].permute(4, 0, 1, 2, 3)
        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs = occ_sem_confs.view(2, -1, self.num_pts_per_ray)
        occ_sem_confs = occ_sem_confs.clamp(self.eps, 1.0 - self.eps)
        # Compute the entropy of the semantic confidences along each ray
        opacities = torch.sigmoid(100.0 * (occ_sem_confs[0, ...] - 0.51))
        transmittance = self.shifted_cumprod(1.0 - opacities)
        ray_gains = transmittance * self.entropy(occ_sem_confs[1, ...])

        # Create a gain image for visualization
        gain_image = ray_gains.view(-1, self.num_pts_per_ray).sum(1)
        gain_image = gain_image.view(self.image_size[1], self.image_size[0])
        gain_image = gain_image / gain_image.max()
        # gain_image = gain_image.clamp(0.0, 1.0)
        gain_image = gain_image.detach().cpu().numpy()
        # Plot non-blocking
        # plt.ion()
        # plt.imshow(gain_image, cmap="viridis")
        # self.counter += 1
        # if self.save_path:
        #     plt.savefig(self.save_path + "loss_" + str(self.counter) + ".png")
        # plt.pause(0.05)
        # plt.clf()

        # # find the region-of-interest, i.e. the most uncertain region in the grid
        # sem_grid = self.grid[..., 2].unsqueeze(0).unsqueeze(0)
        # rois = torch.nn.functional.conv3d(sem_grid, self.kernel, padding=0)
        # mean = unravel_index(torch.argmax(rois), sem_grid.shape)[2:]
        # mean = mean * self.voxel_size + self.origin

        # # find the weighted mean of the grid coordinates, weighted by the semantic confidences
        # weighted_coords = self.grid_coords * self.grid[..., 2].view(-1, 1)
        # mean = torch.sum(weighted_coords, dim=0) / torch.sum(self.grid[..., 2])
        # mean = mean * self.voxel_size + self.origin

        # # define the reward function
        # dist_target = torch.norm(mean - target_params)
        # semantic_gain = torch.mean(ray_gains)
        # # loss = 3.0 * (1.0 - semantic_gain) + dist_target
        # loss = -3.0 * semantic_gain + dist_target
        # # loss = -3.0 * semantic_gain

        semantic_gain = torch.mean(ray_gains)
        loss = -semantic_gain

        # print("semantic_gain", semantic_gain)
        # print("loss", loss)
        # return loss, semantic_gain, mean
        return loss, semantic_gain, None

    def compute_gain_volumetric(
        self,
        camera_params: torch.tensor,
        target_params: torch.tensor,
    ) -> torch.tensor:
        """
        Compute the gain for a given set of parameters
        :param camera_params: camera parameters
        :param target_params: target parameters
        :param current_params: current parameters
        :return: total gain for the viewpoint defined by the parameters
        """
        quat = look_at_rotation(camera_params, target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_params[None, :]
        )
        # Compute point cloud by ray-tracing along ray origins and directions
        t_vals = self.t_vals.clone().requires_grad_()
        ray_origins, ray_directions, _ = self.ray_sampler.ray_origins_directions(
            transforms=transforms
        )
        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)
        ray_points_nor = self.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        # Sample the occupancy probabilities and semantic confidences along each ray
        grid = self.grid[None, ..., 1:3].permute(4, 0, 1, 2, 3)
        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs = occ_sem_confs.view(2, -1, self.num_pts_per_ray)
        occ_sem_confs = occ_sem_confs.clamp(self.eps, 1.0 - self.eps)
        # Compute the entropy of the semantic confidences along each ray
        opacities = torch.sigmoid(100.0 * (occ_sem_confs[0, ...] - 0.51))
        transmittance = self.shifted_cumprod(1.0 - opacities)
        ray_gains = transmittance * self.entropy(occ_sem_confs[0, ...])

        volumetric_gain = torch.mean(ray_gains)
        loss = -volumetric_gain

        # print("volumetric_gain", volumetric_gain)
        # print("loss", loss)
        return loss, volumetric_gain, None

    def set_target_roi(self, target_params: torch.tensor, roi_size=0.05) -> None:
        # Define regions of interest around the target
        self.target_bounds = None
        if target_params is None:
            return
        target_coords = torch.div(
            target_params - self.origin, self.voxel_size, rounding_mode="floor"
        ).to(torch.long)
        # convert roi_size to voxels
        roi_size = int(roi_size / self.voxel_size)
        x_min = torch.clamp(target_coords[0] - roi_size, 0, self.dims[0])
        x_max = torch.clamp(target_coords[0] + roi_size, 0, self.dims[0])
        y_min = torch.clamp(target_coords[1] - roi_size, 0, self.dims[1])
        y_max = torch.clamp(target_coords[1] + roi_size, 0, self.dims[1])
        z_min = torch.clamp(target_coords[2] - roi_size, 0, self.dims[2])
        z_max = torch.clamp(target_coords[2] + roi_size, 0, self.dims[2])
        self.grid[..., 0] = 0
        self.grid[x_min:x_max, y_min:y_max, z_min:z_max, 0] = 1
        self.grid[..., 2] = self.eps
        # self.grid[x_min:x_max, y_min:y_max, z_min:z_max, 1:3] = self.eps
        self.grid[x_min:x_max, y_min:y_max, z_min:z_max, 1:3] = 0.5
        self.target_bounds = torch.tensor(
            [x_min, y_min, z_min, x_max, y_max, z_max], device=self.device
        )
        self.grid[..., 1:3] = torch.clamp(self.grid[..., 1:3], self.eps, 1.0 - self.eps)


    def entropy(self, probs: torch.tensor) -> torch.tensor:
        """
        Compute the entropy of a set of probabilities
        :param probs: tensor of probabilities
        :return: tensor of entropies
        """
        probs_inv = 1.0 - probs
        gains = -(probs * torch.log2(probs)) - (probs_inv * torch.log2(probs_inv))
        return gains

    def dist_cost(self, pose_a: np.array, pose_b: np.array) -> float:
        """
        Compute the distance cost between two poses
        :param pose_a: first pose
        :param pose_b: second pose
        :return: distance cost
        """
        # Euclidean distance between the two positions
        euclidean_dist = np.linalg.norm(pose_a[:3] - pose_b[:3])
        # Quaternion distance between the two rotations
        quaternion_dist = py3dr.quaternion_dist(pose_a[3:], pose_b[3:])
        # Sum the Euclidean and Quaternion distances
        dist_cost = 0.5 * (euclidean_dist + quaternion_dist)
        return dist_cost

    def get_valid_indices(
        self, grid_coords: torch.tensor, dims: torch.tensor
    ) -> torch.tensor:
        """
        Get the indices of the grid coordinates that are within the grid bounds
        :param grid_coords: tensor of grid coordinates
        :param dims: tensor of grid dimensions
        :return: tensor of valid indices
        """
        valid_indices = (
            (grid_coords[:, 0] >= 0)
            & (grid_coords[:, 0] < dims[0])
            & (grid_coords[:, 1] >= 0)
            & (grid_coords[:, 1] < dims[1])
            & (grid_coords[:, 2] >= 0)
            & (grid_coords[:, 2] < dims[2])
        )
        return valid_indices

    def normalize_3d_coordinate(self, points):
        """
        Normalize a tensor of 3D points to the range [-1, 1] along each axis.
        :param points: tensor of 3D points of shape (N, 3)
        :return: tensor of normalized 3D points of shape (N, 3)
        """
        # Compute the range of values for each dimension
        x_min, y_min, z_min = self.min_bound
        x_max, y_max, z_max = self.max_bound
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        # Normalize the points to the range [-1, 1]
        n_points = points.clone()
        n_points_out = torch.zeros_like(n_points)
        n_points_out[..., 0] = 2.0 * (n_points[..., 2] - z_min) / z_range - 1.0
        n_points_out[..., 1] = 2.0 * (n_points[..., 1] - y_min) / y_range - 1.0
        n_points_out[..., 2] = 2.0 * (n_points[..., 0] - x_min) / x_range - 1.0
        return n_points_out

    def shifted_cumprod(self, x: torch.tensor, shift: int = 1) -> torch.tensor:
        """
        Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of ones and removes
        `shift` trailing elements to/from the last dimension of the result
        :param x: tensor of shape (N, ..., C)
        :param shift: number of elements to prepend/remove
        :return: tensor of shape (N, ..., C)
        """
        x_cumprod = torch.cumprod(x, dim=-1)
        x_cumprod_shift = torch.cat(
            [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
        )
        return x_cumprod_shift

    def get_occupied_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        grid_coords = torch.nonzero(self.grid[..., 0] > 0.5) ##1 by default
        semantics = self.grid[
            grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 2
        ]
        points = grid_coords * self.voxel_size + self.origin
        return points, semantics

    # def cluster_semantic_points(self):
    #     """
    #     Cluster the semantic points in the grid using affinity propagation
    #     """
    #     grid_coords = torch.nonzero(self.grid[..., 2] > 0.5)
    #     class_ids = self.grid[
    #         grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 3
    #     ]
    #     semantic_points = grid_coords * self.voxel_size + self.origin
    #     print("semantic_points", semantic_points.shape)
    #     print("class_ids", class_ids.shape)
    #     # Cluster the peduncle points using affinity propagation
    #     points = semantic_points.cpu().numpy()
    #     ids = class_ids.cpu().numpy()
    #     self.clusterer.cluster(points, ids)
