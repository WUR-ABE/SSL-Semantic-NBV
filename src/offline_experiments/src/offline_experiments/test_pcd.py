import sys
import open3d as o3d
import os
import numpy as np
from networks.chamfer import chamfer_distance
import torch
from torch_scatter import scatter_mean

def get_chamfer_dists(pcd_x, pcd_y):
    # We compare the two sets of pointclouds by computing the chamfer loss.
    _, _, dist_x, dist_y = chamfer_distance(pcd_x, pcd_y)
    dist_x = dist_x.cpu().detach()
    dist_y = dist_y.cpu().detach()

    return dist_x, dist_y

def cal_coverage2(partial_pc, complete_pc):
    _, dist2 = get_chamfer_dists(partial_pc[..., 0:3], complete_pc)
    covered_points = dist2 < 0.003
    cov_ratio = torch.sum(covered_points).item() / covered_points.shape[1]
    return '%s/%s' % (str(torch.sum(covered_points).item()),
                        str(covered_points.shape[1])), cov_ratio

def point_cloud_down_sampling(points, voxel_size):
    # Convert points to tensors and move them to GPU if available
    points = torch.tensor(points, dtype=torch.float32)
    points = points.cuda()

    num_channels = points.shape[1]  # Get the number of channels

    # Apply voxel grid downsampling
    voxel_indices = torch.floor(points[:, :3] / voxel_size)
    unique_indices, inverse_indices = torch.unique(voxel_indices, return_inverse=True, dim=0)

    downsampled_points = scatter_mean(points, inverse_indices, dim=0)

    # Adjust the number of channels based on the input point cloud
    if num_channels == 3:
        downsampled_points = downsampled_points[:, :3]  # Keep only the XYZ coordinates
    elif num_channels == 6:
        downsampled_points = downsampled_points[:, :6]  # Keep all the XYZ coordinates and additional channels

    return downsampled_points#.cpu().numpy()

def load_complete_pc(complete_plant_dir, plant, plant_idx, plant_rotation, plant_position = [0,0,0], scale = 1):
        print('loading complete point cloud of [tomato%s]' %
              (str(plant_idx) + '_r(%d)' % int(plant_rotation) + '_t(%d_%d_%d)'%(int(plant_position[0]), int(plant_position[1]), int(plant_position[2]))))
        plant_path = os.path.join(complete_plant_dir,
                                  plant + str(plant_idx) + '.pcd')

        pcd = o3d.io.read_point_cloud(plant_path)
        pcd.scale(scale, center=(0.0, 0.0, 0.0))

        T = np.eye(4)
        T[:3,:3] = pcd.get_rotation_matrix_from_xyz((0, 0, plant_rotation)) 
        T[0:3,3] = np.array(plant_position)/100

        pcd = pcd.transform(T)
        points = np.asarray(pcd.points).astype(np.float32)
        points = point_cloud_down_sampling(points, 0.003).reshape(1, -1, 3)
        return points

x_offset = 0
y_offset = 0
z_offset = 0
rotation = 0
plant_idx = [1,2,3,4,5,6,7,8,9]

normal_pcd_path =  '/home/jianchao/dataset/nbv_data/paper4/data_smaller_fov_distance'
more_occlusion_pcd_path = '/home/jianchao/ros_workspace/paper4_semantic_nbv/src/offline_experiments/materials/view_data_collection/Wed Jul  2 16:21:07 2025'

complete_plant_dir = '/home/jianchao/dataset/nbv_data/paper4/gt_pcd'

for plant_idx in plant_idx:
    partial_dir = normal_pcd_path+'/plant'+str(plant_idx)+'/xyzr_'+str(x_offset)+'_'+str(y_offset)+'_'+str(z_offset)+'_'+str(rotation)+'/pcd'
    print(partial_dir)
    complete_pcd = load_complete_pc(complete_plant_dir, 'plant', plant_idx, rotation, plant_position=[x_offset, y_offset, z_offset], scale=1)

    partial_pcds_numpy = []
    for file in os.listdir(partial_dir):
        if file.endswith(".npy"):
            partial_pcds_numpy.append(np.load(os.path.join(partial_dir, file))[0])

    partial_pcds_numpy = np.array(partial_pcds_numpy, dtype=list)
    partial_pcds_numpy = np.concatenate(partial_pcds_numpy, axis=0)#.reshape(1, -1, 3)
    partial_pcds_numpy_downsampled = point_cloud_down_sampling(partial_pcds_numpy, 0.003).unsqueeze(0)

    covered_points, ratio = cal_coverage2(partial_pc=partial_pcds_numpy_downsampled, complete_pc=complete_pcd)
    print(ratio)
    print(covered_points)

# # #visualize complete_pcd and partial_pcd in different colors using open3d
# complete_pcd_o3d = o3d.geometry.PointCloud()
# complete_pcd_o3d.points = o3d.utility.Vector3dVector(complete_pcd[0].cpu().numpy())
# complete_pcd_o3d.paint_uniform_color([0, 0, 1])

# partial_pcd_o3d = o3d.geometry.PointCloud()
# partial_pcd_o3d.points = o3d.utility.Vector3dVector(partial_pcds_numpy_downsampled[0,:, 0:3].cpu().numpy())
# partial_pcd_o3d.paint_uniform_color([1, 0, 0])

# o3d.visualization.draw_geometries([complete_pcd_o3d, partial_pcd_o3d])