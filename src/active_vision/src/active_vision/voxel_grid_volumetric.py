import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pytransform3d import rotations as py3dr
from active_vision.ray_sampler import RaySampler
from utils.torch_utils import (
    transform_from_rotation_translation,
)

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
        image_size: np.array = np.array([960, 540]),
        intrinsics: np.array = np.array([
            [1135.306761618773, 0.0, 480.5],
            [0.0, 1135.306761618773, 270.5],
            [0.0, 0.0, 1.0]]),

        num_pts_per_ray: int = 256,
        num_features: int = 4,
        eps: torch.float32 = 1e-7,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        save_path: str = None,
        set_roi: bool = False,
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
        self.num_features = num_features
        self.reset_grid()

        # grid coordinates
        # self.grid_coords = torch.meshgrid(
        #     [
        #         torch.arange(
        #             0.0, self.dims[0], device=self.device, dtype=torch.float32
        #         ).to(self.device),
        #         torch.arange(
        #             0.0, self.dims[1], device=self.device, dtype=torch.float32
        #         ).to(self.device),
        #         torch.arange(
        #             0.0, self.dims[2], device=self.device, dtype=torch.float32
        #         ).to(self.device),
        #     ]
        # )
        # self.grid_coords = torch.stack(self.grid_coords, dim=-1).view(-1, 3)


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

    def reset_grid(self):
        self.grid = torch.zeros(
            (self.dims[0], self.dims[1], self.dims[2], self.num_features),
            dtype=torch.float32,
            device=self.device,
        )
        self.grid[..., 1] = 0.1 # original 0.1 probability of occupancy
        self.grid[..., 2] = 0.4 # original 0.4 probability of semantic
        self.grid[..., 3] = -1 # original -1 for class
        self.target_bounds = None

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

        #visualize pointcloud using open3d

        # Convert point cloud to voxel grid coordinates
        grid_coords = torch.div(
            ray_points - self.origin, self.voxel_size, rounding_mode="floor"
        )
        valid_indices = self.get_valid_indices(grid_coords, self.dims)
        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)
        # Update the voxel grid
        log_odds = torch.log( # convert odds to log odds
            torch.div(self.grid[gx, gy, gz, 1:3], 1.0 - self.grid[gx, gy, gz, 1:3]) # convert probability to odds
        )
        # Occupancy information
        ray_occ = self.ray_occ.clone()
        ray_occ[:, -1, :] = points_mask.permute(1, 0)
        log_odds[..., 0] += ray_occ.view(-1, 1)[valid_indices, -1]
        # Semantic information
        ray_sem = self.ray_sem.clone()
        ray_sem[..., -1, :] = semantics.view(-1, 2)
        ray_sem = ray_sem.view(-1, 2)
        log_odds[..., 1] += ray_sem[valid_indices, 0]
        # Update
        odds = torch.exp(log_odds)
        self.grid[gx, gy, gz, 1:3] = torch.div(odds, 1.0 + odds) # convert odds to probabilities
        self.grid[gx, gy, gz, 3] = ray_sem[valid_indices, 1]
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
    def occupancy_semantic_map(
                    self,
                    camera_params: torch.tensor,
                    quat: torch.tensor,
                    mode= 'all') -> torch.tensor:
                
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
        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True) # shape (2, 1, image_size[1]* image_size[0]*self.num_pts_per_ray, 1,1)
        occ_sem_confs = occ_sem_confs.view(2, self.image_size[1], self.image_size[0], self.num_pts_per_ray) # shape (2, image_size[1], image_size[0], self.num_pts_per_ray)
        occ_sem_confs = occ_sem_confs.clamp(self.eps, 1.0 - self.eps)
 
        occ_map = (occ_sem_confs[0, ...]>0.5).sum(dim=-1) / self.num_pts_per_ray # shape (image_size[1], image_size[0])
        free_map = (occ_sem_confs[0, ...]<0.5).sum(dim=-1) / self.num_pts_per_ray # shape (image_size[1], image_size[0])
        unknown_map = ((occ_sem_confs[0, ...]>0.4) & (occ_sem_confs[0, ...]<0.6)).sum(dim=-1) / self.num_pts_per_ray
        roi_map = ((occ_sem_confs[0, ...]>0.5) & (occ_sem_confs[1, ...]>0.5)).sum(dim=-1) / self.num_pts_per_ray # shape (image_size[1], image_size[0])

        #resize all maps to the 540//2 x 960//2
        # Add missing dimensions for interpolation (N, C, H, W format)
        occ_map = F.interpolate(occ_map.unsqueeze(0).unsqueeze(0), size=(self.image_size[1]//2, self.image_size[0]//2), mode='bilinear').squeeze(0).squeeze(0)
        free_map = F.interpolate(free_map.unsqueeze(0).unsqueeze(0), size=(self.image_size[1]//2, self.image_size[0]//2), mode='bilinear').squeeze(0).squeeze(0)
        unknown_map = F.interpolate(unknown_map.unsqueeze(0).unsqueeze(0), size=(self.image_size[1]//2, self.image_size[0]//2), mode='bilinear').squeeze(0).squeeze(0)
        roi_map = F.interpolate(roi_map.unsqueeze(0).unsqueeze(0), size=(self.image_size[1]//2, self.image_size[0]//2), mode='bilinear').squeeze(0).squeeze(0)
        
        occ_map = occ_map / occ_map.max()
        free_map = free_map / free_map.max()
        unknown_map = unknown_map / unknown_map.max()
        roi_map = roi_map / roi_map.max()
        # # #visualize the maps on the same plot with name for each map

        # plt.figure(figsize=(12, 8))
        # plt.subplot(2, 2, 1)
        # plt.imshow(occ_map.detach().cpu().numpy(), cmap='cividis')
        # plt.title("Occupancy Map")
        # plt.colorbar()
        # plt.subplot(2, 2, 2)
        # plt.imshow(free_map.detach().cpu().numpy(), cmap='cividis')
        # plt.title("Free Map")
        # plt.colorbar()
        # plt.subplot(2, 2, 3)
        # plt.imshow(unknown_map.detach().cpu().numpy(), cmap='cividis')
        # plt.title("Unknown Map")
        # plt.colorbar()
        # plt.subplot(2, 2, 4)
        # plt.imshow(roi_map.detach().cpu().numpy(), cmap='cividis')
        # plt.title("ROI Map")
        # plt.colorbar()
        # plt.show()

        combined_map = torch.cat((occ_map.unsqueeze(0), free_map.unsqueeze(0), unknown_map.unsqueeze(0), roi_map.unsqueeze(0)), dim=0)
        combined_map = combined_map.unsqueeze(0).to(torch.float32)
        return combined_map
    
    def compute_gain_volumetric(
        self,
        camera_params: torch.tensor,
        quat: torch.tensor,
    ) -> torch.tensor:
        """
        Compute the gain for a given set of parameters
        :param camera_params: camera parameters
        :param target_params: target parameters
        :param current_params: current parameters
        :return: total gain for the viewpoint defined by the parameters
        """
        # quat = #look_at_rotation(camera_params, target_params)
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
        opacities = torch.sigmoid(1e7 * (occ_sem_confs[0, ...] - 0.51))
        # opacities = torch.where(occ_sem_confs > 0.5, 1.0, 0.0)
        transmittance = self.shifted_cumprod(1.0 - opacities) 
        ray_gains = transmittance * self.entropy(occ_sem_confs[0, ...])

        # # Create an image of ray_gains
        ray_gains_image = ray_gains.sum(dim=-1)
        volumetric_gain = torch.sum(ray_gains)
        loss = -volumetric_gain

        return loss, volumetric_gain, ray_gains_image.detach().reshape(self.image_size[1], self.image_size[0])

    def crop_grid(self, roi_bbox: torch.tensor) -> None:
        min_bound = torch.tensor(roi_bbox[0:3]).to(self.device)
        max_bound = torch.tensor(roi_bbox[3:6]).to(self.device)

        min_coords = torch.div(
            min_bound - self.origin, self.voxel_size, rounding_mode="floor"
        ).to(torch.long)
        max_coords = torch.div(
            max_bound - self.origin, self.voxel_size, rounding_mode="floor"
        ).to(torch.long)
    
        x_min = min_coords[0]
        y_min = min_coords[1]
        z_min = min_coords[2]
        x_max = max_coords[0]
        y_max = max_coords[1]
        z_max = max_coords[2]
        
        self.grid[0:x_min, :, :, 1:3] = 0
        self.grid[:, 0:y_min, :, 1:3] = 0
        self.grid[:,:, 0:z_min, 1:3] = 0
        self.grid[x_max:self.dims[0], :, :, 1:3] = 0
        self.grid[:, y_max:self.dims[1], :, 1:3] = 0
        self.grid[:,:, z_max:self.dims[2], 1:3] = 0

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

    def get_occupied_points(self, world_frame: bool = True):
        """
        Returns the coordinates of the occupied points in the grid
        :param world_frame: if True, return the coordinates in the world frame, otherwise return the coordinates in the grid frame
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        grid_coords = torch.nonzero(self.grid[..., 1] > 0.3)
        semantics = self.grid[
            grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 2
        ]
        if world_frame:
            points = grid_coords * self.voxel_size + self.origin
        else:
            points = grid_coords
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
