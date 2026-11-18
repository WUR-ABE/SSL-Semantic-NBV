#!/usr/bin/env python3
"""Step 1 of the pipeline: sample ground-truth point clouds from the plant meshes.

    $NBV_DATA_ROOT/simulation/meshes/plant<i>.ply  ->  $NBV_DATA_ROOT/simulation/gt_pcd/plant<i>.pcd

Usage:  python3 get_gt_pcd_from_mesh.py [plant ids ...]   (default: 1-20)
"""

import os
import sys

import open3d as o3d

from utils.paths import GT_PCD_DIR, MESH_DIR


def sample_points_from_mesh(file_path, number_of_points=200000):
    try:
        # Read the mesh file
        mesh = o3d.io.read_triangle_mesh(file_path)

        # Check if the mesh is empty
        if mesh.is_empty():
            print(f"Error: The mesh file {file_path} is empty or corrupted.")
            return None

        # Ensure the mesh has vertex normals for point sampling
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Check if the mesh has triangles
        if len(mesh.triangles) == 0:
            print(f"Error: The mesh file {file_path} has no triangles.")
            return None

        # Sample points uniformly from the mesh
        sampled_points = mesh.sample_points_uniformly(number_of_points)

        # Return the sampled points as a point cloud
        return sampled_points
    except Exception as e:
        print(f"An error occurred while processing the mesh file {file_path}: {e}")
        return None


if __name__ == "__main__":
    models = [int(m) for m in sys.argv[1:]] or list(range(1, 21))
    os.makedirs(GT_PCD_DIR, exist_ok=True)
    for model in models:
        mesh_path = os.path.join(MESH_DIR, f'plant{model}.ply')
        out_path = os.path.join(GT_PCD_DIR, f'plant{model}.pcd')
        print(mesh_path, '->', out_path)
        pcd = sample_points_from_mesh(mesh_path)
        if pcd is not None:
            o3d.io.write_point_cloud(out_path, pcd)
