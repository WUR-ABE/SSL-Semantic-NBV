from utils.paths import MATERIALS_DIR, gt_nodes_file, gt_pcd_file, real_plant_dir
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pandas as pd
import os
from torch_scatter import scatter_mean
import torch
import open3d.visualization.rendering as rendering
import cv2


def create_coordinate_frame(position, orientation, size=0.01):
    # Create a coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    r = R.from_quat(orientation)
    coord_frame.rotate(r.as_matrix(), center=(0, 0, 0))
    coord_frame.translate(position)

    return coord_frame


def transform_pcd(pcd, x, y, z, rotation, scale=1, downsample_voxel_size=0.01):
    pcd.scale(scale, center=(0.0, 0.0, 0.0))

    T = np.eye(4)
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, 0, rotation))
    T[0:3, 3] = np.array([x, y, z]) / 100

    pcd = pcd.transform(T)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32)

    pcd = np.concatenate([points, colors], axis=1)
    pcd = point_cloud_down_sampling(pcd, downsample_voxel_size)
    return pcd


def point_cloud_down_sampling(points, voxel_size):
    # Convert points to tensors and move them to GPU if available
    points = torch.tensor(points, dtype=torch.float32)

    num_channels = points.shape[1]  # Get the number of channels

    # Apply voxel grid downsampling
    voxel_indices = torch.floor(points[:, :3] / voxel_size)
    unique_indices, inverse_indices = torch.unique(voxel_indices, return_inverse=True, dim=0)

    downsampled_points = scatter_mean(points, inverse_indices, dim=0)

    # Adjust the number of channels based on the input point cloud
    if num_channels == 3:
        downsampled_points = downsampled_points[:, :3]  # Keep only the XYZ coordinates
    elif num_channels == 6:
        downsampled_points = downsampled_points[
            :, :6
        ]  # Keep all the XYZ coordinates and additional channels

    return downsampled_points.cpu().numpy()


def classify_points_use_bbox(
    point_cloud, recon_target, fruit_poses, fruit_sizes, node_poses, node_sizes
):
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

    fruit_poses = []
    sizes_fruits = []
    node_poses = []
    sizes_nodes = []

    rois_fruits = pd.read_csv(gt_nodes_file(plant_idx, 'fruit'))
    rois_nodes = pd.read_csv(gt_nodes_file(plant_idx, 'internode'))

    if len(rois_fruits) > 0:
        for roi in range(len(rois_fruits)):
            # read the fruits one by one and convert the pose using rotation and position and add to fruits list
            fruit = np.array(
                [rois_fruits['loc_x'][roi], rois_fruits['loc_y'][roi], rois_fruits['loc_z'][roi]]
            )
            # rotate the fruit
            fruit = np.dot(
                np.array(
                    [
                        [np.cos(rotation), -np.sin(rotation), 0],
                        [np.sin(rotation), np.cos(rotation), 0],
                        [0, 0, 1],
                    ]
                ),
                fruit,
            )
            # translate the fruit
            fruit_pose = fruit + np.array([position[0], position[1], position[2]])
            fruit_poses.append(fruit_pose)
            sizes_fruits.append(rois_fruits['radius'][roi])
    else:
        fruit_poses = [np.array([0, 0, 0])]
        sizes_fruits = [0.001]

    for roi in range(len(rois_nodes)):
        node = np.array(
            [rois_nodes['loc_x'][roi], rois_nodes['loc_y'][roi], rois_nodes['loc_z'][roi]]
        )
        # rotate the fruit
        node = np.dot(
            np.array(
                [
                    [np.cos(rotation), -np.sin(rotation), 0],
                    [np.sin(rotation), np.cos(rotation), 0],
                    [0, 0, 1],
                ]
            ),
            node,
        )
        # translate the fruit
        node_pose = node + np.array([position[0], position[1], position[2]])
        node_poses.append(node_pose)
        sizes_nodes.append(0.03)

    return (
        np.array(fruit_poses),
        np.array(sizes_fruits),
        np.array(node_poses),
        np.array(sizes_nodes),
    )


def create_arrow(
    start_point, end_point, color, shaft_radius=0.001, head_radius=0.007, head_length=0.03
):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_radius,
        cone_radius=head_radius,
        cylinder_height=np.linalg.norm(end_point - start_point) - head_length,
        cone_height=head_length,
    )
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
            rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
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
    box.translate(
        position - np.array([size / 2, size / 2, size / 2])
    )  # Center the box at the position
    box.paint_uniform_color(color)
    return box


def jet_colormap(factor):
    """
    Returns a color from the jet colormap (blue->cyan->green->yellow->red)
    factor should be between 0 (blue) and 1 (red)
    """
    if factor < 0.125:
        # Dark blue to blue
        return np.array([0, 0, 0.5 + 4 * factor])
    elif factor < 0.375:
        # Blue to cyan
        return np.array([0, 4 * (factor - 0.125), 1])
    elif factor < 0.625:
        # Cyan to green to yellow
        return np.array([4 * (factor - 0.375), 1, 1 - 4 * (factor - 0.375)])
    elif factor < 0.875:
        # Yellow to red
        return np.array([1, 1 - 4 * (factor - 0.625), 0])
    else:
        # Red to dark red
        return np.array([1 - 4 * (factor - 0.875), 0, 0])


def vis_trajectory_with_frames(pcd, viewpoints, all_viewpoints, save_path=None, point_size=2):
    vis = o3d.visualization.Visualizer()

    vis.create_window(visible=False, width=1920, height=1080)
    roi_index = pcd[:, -1]
    colors = pcd[:, 3:6]
    # 0 = Green (Recon Non-Target)
    colors[roi_index == 0] = [0.0, 1.0, 0.0]
    # 1 = Red (Recon Target)
    colors[roi_index == 1] = [1.0, 0.0, 0.0]
    # 2 = Grey (Ghost/Complete)
    colors[roi_index == 2] = [0.7, 0.7, 0.7]

    com_pcd = o3d.geometry.PointCloud()
    com_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    com_pcd.colors = o3d.utility.Vector3dVector(colors)
    # add frame
    vis.add_geometry(com_pcd)
    opt = vis.get_render_option()
    opt.point_size = point_size  # adjust as needed

    # Set start and end colors for the orange to purple gradient
    start_color = np.array([0, 0, 0.5])  # dark blue
    end_color = np.array([0.5, 0.0, 0.0])  # red

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
            arrow = create_arrow(viewpoints[i - 1][:3], position, color)
            vis.add_geometry(arrow)

    # Center the view
    center = np.array([0.5, -1, 0.85])
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_up([0, 0, 1])
    # Adjust front vector to look forward along the -y -axis
    # Using a combination of y and Z components to achieve the downward angle
    ctr.set_front([0, np.cos(np.radians(-5)), -np.sin(np.radians(2))])

    # Save the visualization if a save path is provided
    if save_path is not None:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Render and save the image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        print(f"Visualization saved to {save_path}")
        vis.destroy_window()


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
    plane.translate([position[0], position[1], position[2]])

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
    o3d.visualization.draw([{'name': 'plane', 'geometry': plane, 'material': material}])


def get_pcd(plant, rotation, position: list, hw: list):
    pcd_path = os.path.join(
        real_plant_dir(plant), "pcd_sam3", "pcd_%s_%s.npy" % (str(hw[0]), str(hw[1]))
    )
    if not os.path.exists(pcd_path):
        print(f"Warning: Partial PCD not found at {pcd_path}")
        return np.empty((0, 3))
    pcd = np.load(pcd_path)
    return pcd  # Returns (N, 3) or similar


if __name__ == '__main__':
    # Parameters
    downsample_voxel_size = 0.006
    point_size = 2

    # Create output directory for saved visualizations
    object_name = 'FRUIT'  # Change this to 'PLANT' or 'NODE' as needed
    planner = 'ssl_semantic_nbv_NOCOLOR_SAM3'

    results_path = MATERIALS_DIR + '/NO_COLOR_RESULTS/2_FRUIT_sam3/results.csv'
    MATERIALS_DIR + '/NO_COLOR_RESULTS/2_FRUIT/results.csv'
    # MATERIALS_DIR + '/NO_COLOR_RESULTS/2_FRUIT_sam3/results.csv'
    results = pd.read_csv(results_path)

    output_dir = MATERIALS_DIR + '/NO_COLOR_RESULTS/2_FRUIT_sam3/trajectory_vis'
    os.makedirs(output_dir, exist_ok=True)

    # Dynamically generate view columns
    views_poses = [f'curr_hw_{i}' for i in range(20)]

    planner_df = results[results['planner'] == planner]

    for idx, row in planner_df.iterrows():
        trajectory_index = row['test_num']
        print(f"Processing trajectory {trajectory_index}")

        views = row[views_poses]
        auc_val = row['auc']
        plant_x_y_z_rotation = row['plant_xyzr']
        plant, x_str, y_str, z_str, rotation_str = plant_x_y_z_rotation.split('_')

        # Floats for transformation
        x_f, y_f, z_f, rotation_f = float(x_str), float(y_str), float(z_str), float(rotation_str)

        # Strings for path (to match directory structure like xyzr_0_0_0_0)
        position_list_str = [x_str, y_str, z_str]

        # Assuming you have the ViewSampler and required imports
        viewpoints_path = os.path.join(real_plant_dir(plant), 'viewpoints.npy')
        if not os.path.exists(viewpoints_path):
            print(f"Warning: Viewpoints file not found: {viewpoints_path}")
            continue

        viewpoints = np.load(viewpoints_path, allow_pickle=True)
        viewpoints = np.array(viewpoints, dtype=np.float32)
        viewpoints = viewpoints[..., [0, 1, 2, 6, 3, 4, 5]]
        viewpoints[..., 1] -= 0.2  # Shift all candidate views by -1 on y-axis

        # 1. Load Complete PCD (Ghost)
        complete_pcd_path = gt_pcd_file(plant)
        if not os.path.exists(complete_pcd_path):
            print(f"Warning: PCD file not found: {complete_pcd_path}")
            continue

        pcd_ghost = o3d.io.read_point_cloud(complete_pcd_path)

        # Transform ghost to world frame
        pcd_ghost = transform_pcd(
            pcd_ghost, x_f, y_f, z_f, rotation_f, downsample_voxel_size=downsample_voxel_size
        )

        # Get ROI used for classification
        fruit_poses, fruit_sizes, node_poses, node_sizes = get_rios(
            int(plant), rotation_f, [x_f, y_f, z_f]
        )

        # Classify Ghost (adds mask column at index 7) - though we might not strictly need it for ghost if it's all grey
        # NOTE: Classify adds a column. If we run it, 0=non-target, 1=target.
        # But we want ghost to be a third color.
        # So we can classify it, then set the entire mask column to 2.
        pcd_ghost = classify_points_use_bbox(
            pcd_ghost, object_name, fruit_poses, fruit_sizes, node_poses, node_sizes
        )

        # Set mask to 2 for Ghost
        pcd_ghost[:, -1] = 2.0

        # 2. Reconstruct from Views
        view_coordinates = []
        for view in views:
            if pd.isna(view):
                break
            view = str(view).strip('()')
            if ',' not in view:
                continue
            view_parts = view.split(',')
            x_coord, y_coord = int(view_parts[0]), int(view_parts[1])
            view_coordinates.append((x_coord, y_coord))

        if not view_coordinates:
            print("No valid views found for this trajectory.")
            continue

        view_to_draw = []
        partial_pcds = []

        for view_coord in view_coordinates:
            view_to_draw.append(viewpoints[view_coord[0], view_coord[1], :])
            # Load partial PCD using STRINGS for path
            partial = get_pcd(
                plant, rotation_str, position_list_str, [view_coord[0], view_coord[1]]
            )

            if partial.size > 0:
                # Flatten (1, N, C) -> (N, C)
                if partial.ndim == 3 and partial.shape[0] == 1:
                    partial = partial[0]

                # Normalize to (N, 3) XYZ
                if partial.ndim == 2:
                    # Heuristic: if dim 0 is small channel count, transpose
                    if partial.shape[0] in [3, 4, 6] and partial.shape[1] > 10:
                        partial = partial.T

                    # Now we expect (N, C). Slice to (N, 3) if present
                    if partial.shape[1] >= 3:
                        partial = partial
                    else:
                        print(f"Skipping malformed PCD with shape {partial.shape}")
                        continue

                partial_pcds.append(partial)

        view_to_draw = np.array(view_to_draw)

        if partial_pcds:
            # Merge partials
            reconstructed_points = np.concatenate(partial_pcds, axis=0)  # (N, 3) or (N, C)
            print("Reconstructed points shape:", reconstructed_points.shape)

            # If partials don't have color/mask, we need to handle shape
            if reconstructed_points.shape[1] == 3:
                # Add placeholder colors [0,0,0]
                colors = np.zeros_like(reconstructed_points)
                reconstructed_pcd = np.concatenate([reconstructed_points, colors], axis=1)
            else:
                reconstructed_pcd = reconstructed_points

            # Downsample if needed (optional but good for speed/viz)
            if reconstructed_pcd.shape[0] > 0:
                reconstructed_pcd = point_cloud_down_sampling(
                    reconstructed_pcd, downsample_voxel_size
                )

                # 3. Merge Ghost and Reconstruction
                # We simply stack them. The visualizer will render all points.
                final_viz_pcd = np.concatenate([pcd_ghost, reconstructed_pcd], axis=0)
            else:
                final_viz_pcd = pcd_ghost
        else:
            final_viz_pcd = pcd_ghost

        # Define save path for this trajectory
        save_path = os.path.join(
            output_dir, f'{planner}_trajectory_{trajectory_index}_sam3_auc_{auc_val:.3f}.png'
        )

        # Pass the save path to the visualization function
        vis_trajectory_with_frames(
            final_viz_pcd, view_to_draw, viewpoints, save_path, point_size=point_size
        )
