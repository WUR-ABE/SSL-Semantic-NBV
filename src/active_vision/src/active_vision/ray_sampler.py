import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as scipy_r

from utils.utils import T_from_rot_trans


class RaySampler:
    """
    Generate and manage rays for ray sampling operations
    """

    def __init__(
        self,
        image_size: np.array,
        intrinsics: np.array,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """
        Initialize ray sampler
        :param image_size: size of the image
        :param device: device to use for computation
        """
        self.device = device
        self.image_size = image_size
        # Camera intrinsic matrix
        self.intrinsic = torch.tensor(intrinsics, device=self.device)
        # self.intrinsic = torch.tensor(
        #     [
        #         [824.2428421710242, 0.0, 300.5],
        #         [0.0, 824.2428421710242, 225.5],
        #         [0.0, 0.0, 1.0],
        #     ],
        #     # # Real camera
        #     # [
        #     #     [685.5028076171875, 0.0, 485.35955810546875],
        #     #     [0.0, 685.6409912109375, 270.7330627441406],
        #     #     [0.0, 0.0, 1.0],
        #     # ],
        #     device=self.device,
        # )
        # Generate camera coordinates
        self.generate_camera_coords()
        # Transformation from optical frame to camera frame
        # TODO: make configurable parameter
        r = scipy_r.from_euler("xyz", [-np.pi / 2, 0.0, -np.pi / 2])
        self.T_oc = T_from_rot_trans(r.as_matrix(), np.array([[0.008, 0, 0.045]]))
        self.T_oc = torch.as_tensor(self.T_oc, dtype=torch.float32, device=self.device)

    def generate_camera_coords(self) -> None:
        """
        Generate camera coordinates, which will be used for ray sampling
        :return: None
        """
        # Create a mesh grid of (u, v) coordinates
        u, v = torch.meshgrid(
            [
                torch.arange(
                    0.0, self.image_size[0], device=self.device, dtype=torch.float32
                ),
                torch.arange(
                    0.0, self.image_size[1], device=self.device, dtype=torch.float32
                ),
            ],
            indexing="xy",
        )
        u, v = u + 0.5, v + 0.5
        # Convert the pixel coordinates to homogeneous coordinates
        pixel_coords = torch.stack((u, v, torch.ones_like(u)), dim=-1)
        # Transform the pixel coordinates to camera coordinates
        self.camera_coords = pixel_coords.view(-1, 3) @ torch.inverse(
            self.intrinsic
        ).t().type(torch.float32)

    def ray_origins_directions(
        self,
        depth_image: Optional[torch.tensor] = None,
        transforms: torch.tensor = None,
        near: float = 0.05,
        far: float = 1.5,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the origins and directions for all rays
        :param depth_image: depth image
        :param transforms: transformation matrices
        :param near: near clipping plane
        :param far: far clipping plane
        :return: ray origins and directions
        """
        batch_size = transforms.shape[0]
        min_depths = near * torch.ones(
            (batch_size, self.image_size[0] * self.image_size[1]),
            dtype=torch.float32,
            device=self.device,
        )
        # If depth image is provided, use it to compute the max depth
        # Otherwise, use the far clipping plane
        if depth_image is not None:
            depth_image[torch.isnan(depth_image)] = far
            max_depths = depth_image.view(1, -1)
        else:
            max_depths = far * torch.ones(
                (batch_size, self.image_size[0] * self.image_size[1]),
                dtype=torch.float32,
                device=self.device,
            )
        # Create a mask that is log odds 0.8 if the depth is less than far and log odds of 0.2 otherwise
        points_mask = torch.where(max_depths < far, 1.4, -1.4)
        # Transform the camera coordinates to world coordinates
        camera_coords = self.camera_coords.clone().requires_grad_()
        ray_origins = (camera_coords * min_depths.unsqueeze(-1)).view(batch_size, -1, 3)
        ray_targets = (camera_coords * max_depths.unsqueeze(-1)).view(batch_size, -1, 3)
        ray_origins = self.transform_points(ray_origins, transforms)
        ray_targets = self.transform_points(ray_targets, transforms)
        # Compute the ray directions
        ray_directions = ray_targets - ray_origins
        return ray_origins, ray_directions, points_mask

    def transform_points(
        self,
        points: torch.tensor,
        transforms: torch.tensor,
    ) -> torch.tensor:
        """
        Transform a point cloud from 'camera_frame' to 'world_frame'
        :param points: point cloud
        :param transforms: transformation matrices
        """
        T_oc = self.T_oc.clone().requires_grad_()
        T_cws = transforms.clone().to(torch.float32).requires_grad_()
        T_ows = T_cws @ T_oc
        points_h = nn.functional.pad(points, (0, 1), "constant", 1.0)
        points_w = points_h @ T_ows.permute(0, 2, 1)
        return points_w[:, :, :3]
