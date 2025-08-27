import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pandas as pd
from view_sampling.view_sampler import ViewSampler
import os
from torch_scatter import scatter_mean
import torch
import open3d.visualization.rendering as rendering

def create_coordinate_frame(position, orientation, size=0.01):
    # Create a coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    r = R.from_quat(orientation)
    coord_frame.rotate(r.as_matrix(), center=(0, 0, 0))
    coord_frame.translate(position)
    
    return coord_frame

def transform_pcd(pcd, x, y, z, rotation, scale=1, downsample_voxel_size=0.003):
    pcd.scale(scale, center=(0.0, 0.0, 0.0))

    T = np.eye(4)
    T[:3,:3] = pcd.get_rotation_matrix_from_xyz((0, 0, rotation)) 
    T[0:3,3] = np.array([x, y, z])/100

    pcd = pcd.transform(T)
    points = np.asarray(pcd.points).astype(np.float32)
    points = point_cloud_down_sampling(points, downsample_voxel_size)
    return points

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

    return downsampled_points.cpu().numpy()

def classify_points_use_bbox(point_cloud, recon_target, fruit_poses, fruit_sizes, node_poses, node_sizes):
        """
        Optimized version using vectorized operations with NumPy
        """
        if recon_target is None:
            return point_cloud
        elif recon_target == 'PLANT':
            # Create a mask of ones with shape (point_cloud.shape[0], 1)
            mask = np.ones((point_cloud.shape[0], 1))
            return np.concatenate([point_cloud, mask], axis=1)
            
        # Vectorized bbox calculations
        center_poses = fruit_poses if recon_target == 'FRUIT' else node_poses
        sizes = fruit_sizes if recon_target == 'FRUIT' else node_sizes
        
        points = point_cloud[:, :3]
        
        # Initialize mask array
        in_any_bbox = np.zeros(points.shape[0], dtype=bool)
        
        # Check each point against all bounding boxes
        for center, size in zip(center_poses, sizes):
            # Calculate distance from points to this center
            distances = np.abs(points - center)
            # Check if point is within bounds of this bbox
            in_this_bbox = np.all(distances <= size, axis=1)
            # Update the mask
            in_any_bbox = np.logical_or(in_any_bbox, in_this_bbox)
        
        # Reshape mask to match expected dimensions
        mask = in_any_bbox.reshape(-1, 1)
        
        # Combine results
        return np.concatenate([point_cloud, mask], axis=1)

def interpolate_color(start_color, end_color, factor):
    """
    Create a color from the jet colormap (blue->cyan->green->yellow->red)
    factor should be between 0 (blue) and 1 (red)
    """
    # Ensure factor is within [0, 1]
    factor = max(0, min(1, factor))
    
    # Jet colormap implementation
    if factor < 0.125:
        # Blue to cyan region (0.0-0.125)
        r = 0
        g = 4 * factor / 0.125
        b = 1
    elif factor < 0.375:
        # Cyan to green region (0.125-0.375)
        r = 0
        g = 1
        b = 1 - 4 * (factor - 0.125) / 0.25
    elif factor < 0.625:
        # Green to yellow region (0.375-0.625)
        r = 4 * (factor - 0.375) / 0.25
        g = 1
        b = 0
    elif factor < 0.875:
        # Yellow to red region (0.625-0.875)
        r = 1
        g = 1 - 4 * (factor - 0.625) / 0.25
        b = 0
    else:
        # Red region (0.875-1.0)
        r = 1
        g = 0
        b = 0
    
    # Ensure all color values are within [0, 1]
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    
    return np.array([r, g, b])

def get_rios(plant_idx, rotation, position):
    roi_path = '/home/jianchao/dataset/nbv_data/paper4/gt_nodes'

    fruit_poses = []
    sizes_fruits = []
    node_poses = []
    sizes_nodes = []

    rois_fruits = pd.read_csv(roi_path + '/plant%s_export_fruit.csv'%str(plant_idx))
    rois_nodes = pd.read_csv(roi_path + '/plant%s_export_internode.csv'%str(plant_idx))

    for roi in range(len(rois_fruits)):
        #read the fruits one by one and convert the pose using rotation and position and add to fruits list
        fruit = np.array([rois_fruits['loc_x'][roi], rois_fruits['loc_y'][roi], rois_fruits['loc_z'][roi]])
        #rotate the fruit
        fruit = np.dot(np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), fruit)
        #translate the fruit
        fruit_pose = fruit + np.array([position[0], position[1], position[2]])/100
        fruit_poses.append(fruit_pose)
        sizes_fruits.append(rois_fruits['radius'][roi])
    
    for roi in range(2, len(rois_nodes)-2):
        node = np.array([rois_nodes['loc_x'][roi], rois_nodes['loc_y'][roi], rois_nodes['loc_z'][roi]])
        #rotate the fruit
        node = np.dot(np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), node)
        #translate the fruit
        node_pose = node + np.array([position[0], position[1], position[2]])/100
        node_poses.append(node_pose)
        sizes_nodes.append(0.015)

    return np.array(fruit_poses), np.array(sizes_fruits), np.array(node_poses), np.array(sizes_nodes)

def create_arrow(start_point, end_point, color, shaft_radius=0.001*2, head_radius=0.007*2, head_length=0.03):
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=shaft_radius,
                                                   cone_radius=head_radius,
                                                   cylinder_height=np.linalg.norm(end_point - start_point) - head_length,
                                                   cone_height=head_length)
    # Set arrow color
    arrow.paint_uniform_color(color)
    
    # Calculate direction vector
    direction = (end_point - start_point) / np.linalg.norm(end_point - start_point)
    
    # Handle the case when direction is parallel or anti-parallel to [0, 0, 1]
    if np.isclose(np.abs(np.dot(direction, [0, 0, 1])), 1.0, atol=1e-6):
        # If direction is [0, 0, 1] or [0, 0, -1], use a different approach
        if direction[2] > 0:  # Direction is [0, 0, 1]
            rotation_matrix = np.eye(3)  # No rotation needed
        else:  # Direction is [0, 0, -1]
            # Rotate 180 degrees around X-axis
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
    else:
        # Normal case - use align_vectors
        rotation_matrix = R.align_vectors([direction], [[0, 0, 1]])[0].as_matrix()
    
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    arrow.translate(start_point)
    return arrow

def create_box(position, color, size=0.03):
    """
    Create a colored box at the specified position
    """
    box = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    box.translate(position - np.array([size/2, size/2, size/2]))  # Center the box at the position
    box.paint_uniform_color(color)
    return box

def jet_colormap(factor):
    """
    Returns a color from the jet colormap (blue->cyan->green->yellow->red)
    factor should be between 0 (blue) and 1 (red)
    """
    if factor < 0.125:
        # Dark blue to blue
        return np.array([0, 0, 0.5 + 4*factor])
    elif factor < 0.375:
        # Blue to cyan
        return np.array([0, 4*(factor-0.125), 1])
    elif factor < 0.625:
        # Cyan to green to yellow
        return np.array([4*(factor-0.375), 1, 1-4*(factor-0.375)])
    elif factor < 0.875:
        # Yellow to red
        return np.array([1, 1-4*(factor-0.625), 0])
    else:
        # Red to dark red
        return np.array([1-4*(factor-0.875), 0, 0])
    
def vis_trajectory_with_frames(pcd, viewpoints, all_viewpoints, save_path=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    roi_index = pcd[:,-1]
    colors = np.zeros((pcd.shape[0], 3))
     # set to light red
    colors[roi_index==1, :] = [1, 0.6, 0.6]
    colors[roi_index==0, :] = [0.8, 0.8, 0.8]

    com_pcd = o3d.geometry.PointCloud()
    com_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    com_pcd.colors = o3d.utility.Vector3dVector(colors)
    #add frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis.add_geometry(frame)
    vis.add_geometry(com_pcd)
    
    # Set start and end colors for the orange to purple gradient
    start_color = np.array([0, 0, 0.5])  # dark blue
    end_color = np.array([0.5, 0.0, 0.0])    # red

    num_arrows = max(1, len(viewpoints) - 1)  # Avoid division by zero
    
    # Add blue box at the starting point
    if len(viewpoints) > 0:
        start_box = create_box(viewpoints[0][:3], [0, 0, 1])  # Blue box
        vis.add_geometry(start_box)
    
    # Add red box at the ending point
    if len(viewpoints) > 1:
        end_box = create_box(viewpoints[-1][:3], [1, 0, 0])  # Red box
        vis.add_geometry(end_box)
    
    for i, vp in enumerate(viewpoints):
        position = vp[:3]
        orientation = vp[3:]
        frame = create_coordinate_frame(position, orientation)
        vis.add_geometry(frame)
        
        if i > 0:
            color = jet_colormap(i / num_arrows)
            arrow = create_arrow(viewpoints[i-1][:3], position, color)
            vis.add_geometry(arrow)
    
    for view in all_viewpoints.reshape(-1, 7):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(view[:3])
        sphere.paint_uniform_color([88/255, 165/255, 193/255])  # Ensure values are in [0,1]
        vis.add_geometry(sphere)
    
    # for view in all_viewpoints.reshape(-1, 7)[0:1]:
    #     attention_map, material = create_attention_map(view)
    #     vis.get_render_option().mesh_show_back_face = True  # Optional: improve visibility
    #     vis.add_geometry(attention_map)
    #     rendering_scene = vis.get_scene()
    #     if rendering_scene is not None:
    #         rendering_scene.assign_material(attention_map, material)

    # o3d.visualization.draw([
    #     {'name': 'pcd', 'geometry': com_pcd},
    #     {'name': 'arrow', 'geometry': arrow},
    #     {'name': 'frame', 'geometry': frame},
    #     {'name': 'attention_map', 'geometry': attention_map, 'material': material}
    # ])
    # Center the view
    # positions = np.array([vp[:3] for vp in viewpoints])
    # center = np.mean(positions, axis=0)
    center =  np.array([-0.5, 0, 0.7])
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_up([0, 0, 1])
    # Adjust front vector to look 30 degrees downward along the X-axis
    # Using a combination of X and Z components to achieve the downward angle
    ctr.set_front([-np.cos(np.radians(-10)), 0, -np.sin(np.radians(-10))])
    
    # Save the visualization if a save path is provided
    if save_path is not None:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Render and save the image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        print(f"Visualization saved to {save_path}")
    vis.run()

def create_search_map(viewpoint):
    print(viewpoint)
    print('1111')
    position = viewpoint[:3]
    orientation = viewpoint[3:]
    orientation = orientation / np.linalg.norm(orientation)

    h_fov = 0.778  # radians
    v_fov = 0.469  # radians
    distance = 0.8

    width = 2 * distance * np.tan(h_fov / 2)
    height = 2 * distance * np.tan(v_fov / 2)


    plane = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=0.001)
    plane.compute_vertex_normals()

    # Rotate to face forward
    plane.rotate(R.from_euler('y', 90, degrees=True).as_matrix(), center=plane.get_center())
    plane.rotate(R.from_euler('x', 90, degrees=True).as_matrix(), center=plane.get_center())

    # Center and translate
    plane.translate(-plane.get_center())
    plane.translate([position[0],
                     position[1],
                     position[2]])

    # Set blue color
    plane.paint_uniform_color([0, 0, 1])

    # Create transparent material
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [0, 0, 1, 0.3]  # blue, 30% transparent
    material.base_metallic = 0.0
    material.base_roughness = 1.0
    material.transmission = 1.0

    # plot the plane
    o3d.visualization.draw([
        {'name': 'plane', 'geometry': plane, 'material': material}
    ])

    #return plane, material


if __name__ == '__main__':
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

    
    # Create output directory for saved visualizations
    object_name = 'NODE'
    planner = 'ssl_semantic_nbv'

    complete_pcd_dir = '/home/jianchao/dataset/nbv_data/paper4/gt_pcd'
    results = pd.read_csv(f'/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/prilinary_test/draw_trajectory_attentionmap/{object_name}/results.csv')
    
    output_dir = f'/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/prilinary_test/draw_trajectory_attentionmap/short_cut/{object_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    for trajectory_index in range(20):
        views_poses = ['curr_hw_0','curr_hw_1','curr_hw_2', 'curr_hw_3', 'curr_hw_4', 'curr_hw_5', 'curr_hw_6', 'curr_hw_7', 'curr_hw_8', 'curr_hw_9', 'curr_hw_10', 'curr_hw_11', 'curr_hw_12', 'curr_hw_13', 'curr_hw_14']
        
        planner_df = results[results['planner']==planner]
        trajectory = planner_df.iloc[trajectory_index]
        views = trajectory[views_poses]
        plant_x_y_z_rotation = trajectory['plant_xyzr']
        plant, x, y, z, rotation = plant_x_y_z_rotation.split('_')
        x, y, z, rotation = float(x), float(y), float(z), float(rotation)

        complete_pcd_path = os.path.join(complete_pcd_dir, f'plant{plant}.pcd')
        pcd = o3d.io.read_point_cloud(complete_pcd_path)
        fruit_poses, fruit_sizes, node_poses, node_sizes = get_rios(plant, rotation, [x, y, z])

        pcd = transform_pcd(pcd, x, y, z, rotation)
        pcd = classify_points_use_bbox(pcd, object_name, fruit_poses, fruit_sizes, node_poses, node_sizes)

        view_coordinates = []
        for view in views:
            view = view.strip('()')
            view = view.split(',')
            x, y = int(view[0]), int(view[1])
            view_coordinates.append((x, y))

        view_to_draw = []
        for view in view_coordinates:
            view_to_draw.append(viewpoints[view[0], view[1], :])
        
        # Define save path for this trajectory
        save_path = os.path.join(output_dir, f'{planner}_trajectory_{trajectory_index}.png')
            
        # Pass the save path to the visualization function
        vis_trajectory_with_frames(pcd, view_to_draw, viewpoints, save_path)
