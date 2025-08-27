import open3d as o3d
import os

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
    dir_mesh = '/home/jianchao/dataset/nbv_data/paper4/mesh_ply'
    dir_pcd_save = '/home/jianchao/dataset/nbv_data/paper4/gt_pcd'
    
    models = [10, 11,12,13,14,15,16,17,18,19,20]#[21, 22, 24, 25, 27, 28]#[1,2,4,7,8,10 ]
    for model in models:
        mesh_path = os.path.join(dir_mesh, f'plant{model}.ply')
        print(mesh_path)
        pcd = sample_points_from_mesh(mesh_path)
        if pcd is not None:
            o3d.io.write_point_cloud(os.path.join(dir_pcd_save, f'plant{model}.pcd'), pcd)
            # o3d.visualization.draw_geometries([pcd])
