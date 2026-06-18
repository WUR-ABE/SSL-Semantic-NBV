#!/usr/bin/env python3
from nbv_compute.nbv_computor import Nbvcomputor
import rospy
import open3d as o3d
import numpy as np
import torch
from view_sampling.view_sampler import ViewSampler


def create_box(color=[0.5, 0.5, 0.5]):
    # Create bounding box
    min_bound = np.array([-0.4, -0.4, 0])
    max_bound = np.array([0.4, 0.4, 1.4])
    
    # Create points for the bounding box corners
    points = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],  # 0
        [max_bound[0], min_bound[1], min_bound[2]],  # 1
        [max_bound[0], max_bound[1], min_bound[2]],  # 2
        [min_bound[0], max_bound[1], min_bound[2]],  # 3
        [min_bound[0], min_bound[1], max_bound[2]],  # 4
        [max_bound[0], min_bound[1], max_bound[2]],  # 5
        [max_bound[0], max_bound[1], max_bound[2]],  # 6
        [min_bound[0], max_bound[1], max_bound[2]]   # 7
    ])

    # Define the lines connecting the points
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ])

    # Create the line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Set grey color for all lines
    grey_color = color  # RGB values for grey
    line_set.colors = o3d.utility.Vector3dVector([grey_color for _ in range(len(lines))])

    return line_set

def resample_pc(pcd, n=1024):
    """
    Downsample point cloud using voxel downsampling.
    
    Args:
        pcd (torch.Tensor): Input point cloud of shape (1, N, 7)
        n (int): Target number of points
    
    Returns:
        torch.Tensor: Resampled point cloud of shape (1, n, 7)
    """
    device = pcd.device
    batch_size, num_points, num_features = pcd.shape
    
    # Remove batch dimension for processing
    pcd_flat = pcd.squeeze(0)  # Shape: (N, 7)
    points = pcd_flat[:, :3]   # Shape: (N, 3) - xyz coordinates
    features = pcd_flat[:, 3:] # Shape: (N, 4) - remaining features
    
    # Compute voxel size based on point cloud bounds
    bbox_min = points.min(dim=0).values  # Shape: (3,)
    bbox_max = points.max(dim=0).values  # Shape: (3,)
    
    # Calculate voxel size to get approximately n voxels
    bbox_volume = (bbox_max - bbox_min).prod()
    voxel_size = (bbox_volume / n) ** (1/3)
    
    # Ensure minimum voxel size to avoid division by zero
    voxel_size = max(voxel_size.item(), 1e-6)
    
    # Quantize points to voxel grid
    voxel_indices = ((points - bbox_min) / voxel_size).long()
    
    # Create unique voxel keys
    # Use large multipliers to avoid hash collisions
    voxel_keys = (voxel_indices[:, 0] * 1000000 + 
                  voxel_indices[:, 1] * 1000 + 
                  voxel_indices[:, 2])
    
    # Find unique voxels and get one representative point per voxel
    unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
    
    # Select one point from each voxel (take the first occurrence)
    sampled_indices = []
    for i, unique_key in enumerate(unique_keys):
        # Find all points in this voxel
        voxel_mask = (voxel_keys == unique_key)
        voxel_point_indices = voxel_mask.nonzero(as_tuple=True)[0]
        
        # Take the first point in this voxel
        sampled_indices.append(voxel_point_indices[0])
    
    sampled_indices = torch.stack(sampled_indices)
    
    # Handle the case where we have more or fewer voxels than target n
    if len(sampled_indices) >= n:
        # Randomly select n points from sampled voxels
        perm = torch.randperm(len(sampled_indices), device=device)[:n]
        final_indices = sampled_indices[perm]
    else:
        # If we have fewer voxels than target, pad with random points
        remaining = n - len(sampled_indices)
        additional_indices = torch.randint(
            0, num_points, (remaining,), device=device
        )
        final_indices = torch.cat([sampled_indices, additional_indices])
    
    # Gather the selected points and features
    resampled_pcd = pcd_flat[final_indices]  # Shape: (n, 7)
    
    # Add batch dimension back
    resampled_pcd = resampled_pcd.unsqueeze(0)  # Shape: (1, n, 7)
    
    return resampled_pcd

# Create visualization window
def visualize_pcd_and_bbox(acc_pcd_o3d, bbox=None, view_pose=None, pcd_type='part'):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=1280)

    # Add geometries
    vis.add_geometry(acc_pcd_o3d)
    if bbox is not None:
        vis.add_geometry(bbox)
    
    # add the coordinate axis
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    # IMPORTANT: Get view control AFTER adding geometries
    view_control = vis.get_view_control()
    
    if pcd_type == 'part':
        eye = view_pose
        center = np.array([0.0, 0.0, view_pose[2]-0.1])
        zoom = 0.3

    elif pcd_type == 'acc':
        # Set fixed camera parameters (using simple method)
        eye = np.array([-3.0, 1.0, 1.5])        # Fixed camera position
        center = np.array([0.0, 0.0, 0.7])     # Fixed look-at point
        zoom = 0.6

    up = np.array([0.0, 0.0, 1.0])         # Fixed up vector
    
    # # Apply the fixed view parameters
    # view_control.set_lookat(center)
    # view_control.set_up(up) 
    # view_control.set_front(eye - center)
    # view_control.set_zoom(0.8)
    
    # # CRITICAL: Force the view to stay fixed
    # # This prevents automatic view adjustment
    # view_control.set_constant_z_near(0.1)
    # view_control.set_constant_z_far(3.0)
    
    # Set fixed rendering options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.line_width = 1.0
    opt.background_color = np.array([212/255, 225/255, 227/255])
    
    # FORCE the view to be exactly what we set
    # This overrides any automatic adjustments
    view_control.set_lookat(center)
    view_control.set_up(up)
    view_control.set_front(eye - center)
    view_control.set_zoom(zoom)

    # Run visualization
    vis.run()
    vis.destroy_window()

part_pcd_dir = '/home/jianchao/dataset/nbv_data/paper4/data_smaller_fov_distance'
complete_pcd_dir = '/home/jianchao/dataset/nbv_data/paper4/data_complete'

plant_index = 5
x = 0
y = 0
h = 0
rotation = 0

row_range = [1,2,3,4,5,6,7,8]
col_range = [4]

if __name__ == '__main__':
    rospy.init_node('test_part_pcd')
    nbv_compute = Nbvcomputor(octomap_resulotion = 0.003, recon_target='FRUIT')
     # Assuming you have the ViewSampler and required imports
    view_config = {
                'center_x': -0.6, 'center_y': 0,
                'num_col': 9,
                'num_row': 9,
                'interval_h': 0.15,
                'interval_w': 0.15,
            }
    
    view_sampler = ViewSampler(distribution_type="plane")  # plane, cylinder
    viewpoints = view_sampler.sample_views(**view_config)

    nbv_compute.load_complete_pc(
            'plant',
            plant_index,
            plant_rotation=rotation,
            plant_position=[x, y, h],
            scale = 1,
            visual_com = False,
            visual_roi = False
        )
    print('complete pc shape:', nbv_compute.complete_pc.shape)

    plant_path = part_pcd_dir + '/plant%s' % str(plant_index)+ '/xyzr_%s_%s_%s_%s' % (str(x), str(y), str(h), str(rotation))

    init_pcd = True
    for row in row_range:
        for col in col_range:
            print('row:', row, 'col:', col)

            part_pcd = torch.tensor(np.load(plant_path + '/pcd' + '/pcd_%s_%s.npy' % (str(row), str(col)))).cuda()
            part_pcd = nbv_compute.classify_points_use_bbox(part_pcd)
            rgb = np.load(plant_path + '/rgb/rgb_%s_%s.npy' % (str(row), str(col)))
            depth = np.load(plant_path + '/depth/depth_%s_%s.npy' % (str(row), str(col)))
            view_pose = np.load(plant_path + '/view/view_%s_%s.npy' % (str(row), str(col)))
            view_xyz = view_pose[0:3]

            if init_pcd:
                acc_pcd = part_pcd
                init_pcd = False

            else:
                acc_pcd, _, _, _ = nbv_compute.combine_pc(acc_pcd, part_pcd, visual_pc=False)

            # visualize the part_pcd
            part_pcd_ = part_pcd.cpu().numpy(
            )
            part_pcd_xyz = part_pcd_[..., 0:3].reshape(-1, 3)
            part_pcd_class = part_pcd_[..., -1].reshape(-1, 1)
            part_pcd_rgb = np.zeros((part_pcd_xyz.shape[0], 3))
            part_pcd_rgb[(part_pcd_class == 0).squeeze(), ...] = [0, 0.7, 0]
            part_pcd_rgb[(part_pcd_class == 1).squeeze(), ...] = [0.7, 0, 0]
            part_pcd_o3d = o3d.geometry.PointCloud()
            part_pcd_o3d.points = o3d.utility.Vector3dVector(part_pcd_xyz)
            part_pcd_o3d.colors = o3d.utility.Vector3dVector(part_pcd_rgb)

            # set light blue color
            line_set = create_box(color = [212/255, 225/255, 227/255])

            visualize_pcd_and_bbox(part_pcd_o3d, line_set, view_xyz, pcd_type='part')

            print('acc_pcd shape:', acc_pcd.shape)

            _, cr = nbv_compute.cal_coverage(acc_pcd)
            print('coverage:', cr)

            # # # #visualize the complete point cloud and acc_pcd using open3d and give different colors
            # complete_pc_ = nbv_compute.complete_pc.cpu().numpy()
            # complete_pcd_o3d= o3d.geometry.PointCloud()
            # complete_xyz = complete_pc_[:, 0:3].reshape(-1, 3)
            # complete_class = complete_pc_[:, -1].reshape(-1, 1)
            # #get the rgb for each point based on the class, if the classs is 0, set to green, if the class is 1, set to red, if the class is 2, set to blue
            # complete_rgb = np.zeros((complete_xyz.shape[0], 3))
            # complete_rgb[complete_class == 0] = [0, 1, 0]
            # complete_rgb[complete_class == 1] = [1, 0, 0]

            # complete_pcd_o3d.points = o3d.utility.Vector3dVector(complete_xyz)
            # complete_pcd_o3d.colors = o3d.utility.Vector3dVector(complete_rgb)
            # # assign uniform color to the complete point cloud
            # # complete_pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])

            acc_pcd_ = acc_pcd.cpu().numpy()
            acc_pcd_xyz = acc_pcd_[..., 0:3].reshape(-1, 3)
            acc_pcd_class = acc_pcd_[..., -1].reshape(-1, 1)

            acc_pcd_rgb = np.zeros((acc_pcd_xyz.shape[0], 3))
            acc_pcd_rgb[(acc_pcd_class == 0).squeeze(), ...] = [0, 0.7, 0]
            acc_pcd_rgb[(acc_pcd_class == 1).squeeze(), ...] = [0.7, 0, 0]

            acc_pcd_o3d = o3d.geometry.PointCloud()
            acc_pcd_o3d.points = o3d.utility.Vector3dVector(acc_pcd_xyz)
            acc_pcd_o3d.colors = o3d.utility.Vector3dVector(acc_pcd_rgb)

            line_set = create_box(color = [212/255, 225/255, 227/255])
            # Visualize both point cloud and bounding box
            visualize_pcd_and_bbox(acc_pcd_o3d, line_set, pcd_type='acc')

            print('acc_pcd shape:', acc_pcd.shape)

            acc_pcd_downsampled = resample_pc(acc_pcd, n=1024)
            print('acc_pcd_downsampled shape:', acc_pcd_downsampled.shape)

            acc_pcd_downsampled_ = acc_pcd_downsampled.cpu().numpy()
            acc_pcd_downsampled_xyz = acc_pcd_downsampled_[..., 0:3].reshape(-1, 3)
            acc_pcd_downsampled_class = acc_pcd_downsampled_[..., -1].reshape(-1, 1)
            acc_pcd_downsampled_rgb = np.zeros((acc_pcd_downsampled_xyz.shape[0], 3))
            acc_pcd_downsampled_rgb[(acc_pcd_downsampled_class == 0).squeeze(), ...] = [0, 0.7, 0]
            acc_pcd_downsampled_rgb[(acc_pcd_downsampled_class == 1).squeeze(), ...] = [0.7, 0, 0]
            acc_pcd_downsampled_o3d = o3d.geometry.PointCloud()
            acc_pcd_downsampled_o3d.points = o3d.utility.Vector3dVector(acc_pcd_downsampled_xyz)
            acc_pcd_downsampled_o3d.colors = o3d.utility.Vector3dVector(acc_pcd_downsampled_rgb)

            line_set = create_box(color = [212/255, 225/255, 227/255])
            # Visualize both point cloud and bounding box
            visualize_pcd_and_bbox(acc_pcd_downsampled_o3d, line_set, pcd_type='acc')



    
            
    

    


    




