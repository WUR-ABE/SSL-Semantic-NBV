#!/usr/bin/env python3
import numpy as np
import random
import torch
import open3d as o3d
from typing import Union
import os
import torch.nn as nn   
from matplotlib import pyplot as plt

class Voxelization(nn.Module):
    def __init__(self, origin: torch.Tensor, dim_xyz: torch.Tensor, voxel_size: torch.Tensor) -> None:
        super(Voxelization, self).__init__()
        self.origin = origin.to(device='cuda')
        self.dim_xyz = dim_xyz.to(device='cuda')
        self.voxel_size = voxel_size.to(device='cuda')
    
    def forward(self, point_cloud: Union[np.ndarray, torch.Tensor], batch_mode: bool = False) -> torch.Tensor:
        grid_shape = (int(self.dim_xyz[0] / self.voxel_size)+1, 
                      int(self.dim_xyz[1] / self.voxel_size)+1, 
                      int(self.dim_xyz[2] / self.voxel_size)+1, 
                      4)
        
        # Convert numpy to tensor if needed
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        else:
            point_cloud = point_cloud.to(dtype=torch.float32)
        
        # Get the device from the point cloud
        device = point_cloud.device
        
        if batch_mode:
            # For batch processing, create a batch of grids
            batch_size = point_cloud.shape[0]
            grid = torch.zeros((batch_size, *grid_shape), device=device)
            
            for b in range(batch_size):
                batch_points = point_cloud[b]
                # Process each batch item
                grid[b] = self._process_single_pointcloud(batch_points, grid_shape)
                
            return grid
        else:
            # Original single point cloud processing
            grid = torch.zeros(grid_shape, device=device)
            return self._process_single_pointcloud(point_cloud, grid_shape)
    
    def _process_single_pointcloud(self, point_cloud: Union[np.ndarray, torch.Tensor], grid_shape) -> torch.Tensor:
        # Get the device from the point cloud
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
            device = torch.device('cpu')
        else:
            device = point_cloud.device
            point_cloud = point_cloud.to(dtype=torch.float32)
        
        grid = torch.zeros(grid_shape, device=device)
        
        point_cloud[..., :3] -= self.origin.to(device)
        point_cloud[..., :3] /= self.voxel_size
        point_cloud[..., :3] = torch.floor(point_cloud[..., :3]) # [N, 6]

        valid_points = torch.all((point_cloud[..., :3] >= 0) & (point_cloud[..., :3] < torch.tensor([grid.shape[0], grid.shape[1], grid.shape[2]], device=device)), dim=-1) # shape: [N]
        point_cloud = point_cloud[valid_points]

        if point_cloud.shape[0] == 0:  # No valid points
            return grid

        x_indices = point_cloud[..., 0].long()
        y_indices = point_cloud[..., 1].long()
        z_indices = point_cloud[..., 2].long()

        target_label = point_cloud[..., -1]==1
        non_target_label = point_cloud[..., -1]==0

        grid[x_indices[target_label], y_indices[target_label], z_indices[target_label], -1] = 1
        grid[x_indices[non_target_label], y_indices[non_target_label], z_indices[non_target_label], -1] = 0.5
        grid[x_indices, y_indices, z_indices, 0:3] = point_cloud[..., 3:6]

        return grid

    def log_odds(self, p): # this functino is used to convert the probability to log odds, log odds represents the confidence of the occupancy
        return torch.log(p / (1 - p))
    
    def inverse_log_odds(self, l):
        return 1 - 1 / (1 + torch.exp(l))

    def visual_grid(self, grid, display_mode='point_cloud', batch_idx=0):
        # If grid is batched, select the specified batch index
        if len(grid.shape) == 5:  # Batched grid
            grid = grid[batch_idx]
        
        grid_cpu = grid.detach().cpu()
        origin_cpu = self.origin.detach().cpu()
        voxel_size_cpu = self.voxel_size.detach().cpu()
        dim_xyz_cpu = self.dim_xyz.detach().cpu()

        valid_indices = torch.nonzero(grid_cpu[..., -1] > 0)
        
        x_indices = valid_indices[:, 0]
        y_indices = valid_indices[:, 1]
        z_indices = valid_indices[:, 2]
        
        valid_points = valid_indices * voxel_size_cpu + origin_cpu
        valid_colors = grid_cpu[x_indices, y_indices, z_indices, 0:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=origin_cpu.numpy(), max_bound=(origin_cpu + dim_xyz_cpu).numpy())
        bbox.color = (1, 0, 0)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        if display_mode == 'point_cloud':
            o3d.visualization.draw_geometries([pcd, axes, bbox])
        elif display_mode == 'voxelgrid':
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size_cpu)
            o3d.visualization.draw_geometries([voxel_grid, axes, bbox])
        elif display_mode == 'octree':
            octree = o3d.geometry.Octree(max_depth=12)
            octree.convert_from_point_cloud(pcd, size_expand=0.01)
            o3d.visualization.draw_geometries([octree, axes, bbox])

    def calculate_entropy(self, grid) -> float:
        # Handle both batched and non-batched grids
        if len(grid.shape) == 5:  # Batched grid
            occupancy = grid[..., -1]
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            entropy = -occupancy*torch.log(occupancy + epsilon) - (1-occupancy)*torch.log(1-occupancy + epsilon)
            return torch.sum(entropy, dim=(1, 2, 3))  # Sum over spatial dimensions, keep batch dimension
        else:
            occupancy = grid[..., -1]
            epsilon = 1e-10
            entropy = -occupancy*torch.log(occupancy + epsilon) - (1-occupancy)*torch.log(1-occupancy + epsilon)
            return torch.sum(entropy)
    
    def slice_grid(self, grid):
        # 从grid中获得切片在x的方向上，每隔10个切片
        print(grid.shape)
        rgb_slice = grid[:,::10, :, :, -1] # shape: [B, 14, H, W]
        return rgb_slice
    
    def show_slice(self, rgb_slice):
        # 将rgb_slice转换为numpy数组
        rgb_slice = rgb_slice.detach().cpu().numpy()
        # 将rgb_slice转换为PIL图像
        rgb_slice_label = rgb_slice.transpose(1, 2, 0)
        rgb_slice_label = rgb_slice_label.reshape(rgb_slice_label.shape[0], rgb_slice_label.shape[1], rgb_slice_label.shape[2])

        plt.imshow(rgb_slice_label)
        plt.show()

    
if __name__ == "__main__":
    number = 10
    dir = '/home/jianchao/dataset/nbv_data/paper4/data_each_view_less_view/plant2/xyzr_0_0_0_0/pcd'
    pcd_list = os.listdir(dir)
    pcds = random.sample(pcd_list, number)
    voxelizater = Voxelization(origin=torch.tensor([-0.4, -0.4, 0]), dim_xyz=torch.tensor([0.8, 0.8, 1.4]), voxel_size= torch.tensor(0.006))
    
    entropy_list = []
    for i in range(1, number+1):
        pcd_to_merge = pcds[:i]
        accumulate_pcd = []
        for pcd in pcd_to_merge:
            pcd_path = os.path.join(dir, pcd)
            pcd = np.load(pcd_path) # shape: [B, N, 3]
            # add one dimension to the last axis, all the elements are 1
            pcd = np.concatenate((pcd, np.ones((pcd.shape[0], pcd.shape[1], 1))), axis=-1)
            accumulate_pcd.append(pcd)
        accumulate_pcd = np.concatenate(accumulate_pcd, axis=1)
        grid = voxelizater(torch.from_numpy(accumulate_pcd).float().cuda())
        voxelizater.visual_grid(grid, 'point_cloud')
        voxelizater.slice_grid(grid)
        # entropy = voxel_ig.calculate_entropy()
        # entropy_list.append(entropy)
    
    # draw the entropy curve
    # import matplotlib.pyplot as plt
    # #convert the entropy to relative values with respect to the previous one
    # entropy_list = np.array(entropy_list)
    # ig = entropy_list[:-1] - entropy_list[1:]

    # # entropy_list = entropy_list * (np.indices(entropy_list.shape)[0]+1)
    # #plot both the entropy and the information gain curve on one figure
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(range(1, len(entropy_list)+1), entropy_list, label='Entropy')
    # ax2.plot(range(1, len(ig)+1), ig, label='Information Gain')
    # plt.show()


    


