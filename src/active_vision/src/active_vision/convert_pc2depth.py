import numpy as np
from scipy.spatial.transform import Rotation as R


def get_tranform_matrix(translation, orientation):
    trans_world2base = np.array([translation[0], translation[1], translation[2]])  # [x, y, z]
    rotate_world2base = np.array([-orientation[0],-orientation[1],-orientation[2],orientation[3]])  # Quaternion [x, y, z, w]
    rotation_matrix_world2base = R.from_quat(rotate_world2base).as_matrix()
    transformation_world2base = np.identity(4)
    transformation_world2base[:3, :3] = rotation_matrix_world2base
    transformation_world2base[:3, 3] = -np.dot(rotation_matrix_world2base, trans_world2base)

    trans_base2len = np.array([0.008, 0.000, 0.045])  # [x, y, z]
    rotate_base2len = np.array([0.5, -0.5, 0.5, 0.5])  # Quaternion [x, y, z, w]
    rotation_matrix_base2lan = R.from_quat(rotate_base2len).as_matrix()
    transformation_base2len = np.identity(4)
    transformation_base2len[:3, :3] = rotation_matrix_base2lan
    transformation_base2len[:3, 3] = -np.dot(rotation_matrix_base2lan, trans_base2len)

    transformation_world2len = np.dot(transformation_base2len, transformation_world2base)
    return transformation_world2len, transformation_base2len

def transform_global2len(point_cloud, viewpoint):
    translation = viewpoint[0:3]
    orientation = viewpoint[3:]

    homogeneous_points = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=1)
    matrix_world2len, matrix_base2len = get_tranform_matrix(translation, orientation)
    transformed_xyz = np.dot(matrix_world2len, homogeneous_points.T).T[:, :3] 
    return transformed_xyz, matrix_world2len, matrix_base2len

def convert_pc2depth(point_cloud):
    image_width = 960
    image_height = 540

    K = np.array([[1135.306761618773, 0.0, 480.5],
                [0.0, 1135.306761618773, 270.5],
                [0.0, 0.0, 1.0]])
    
    x, y, z = point_cloud.T
    projected_points = np.dot(K, np.array([x, y, z]))
    pixel_x = (projected_points[0] / projected_points[2]).astype(int)
    pixel_y = (projected_points[1] / projected_points[2]).astype(int)

    valid_mask = (pixel_x >= 0) & (pixel_x < image_width) & (pixel_y >= 0) & (pixel_y < image_height)
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)
    depth_image[pixel_y[valid_mask], pixel_x[valid_mask]] = z[valid_mask]
    return depth_image
    

if __name__ == '__main__':
    viewpoints = np.load('/home/jianchao/ros_workspace/deep_learning_nbv/src/nbv_data_collection/materials/poses.pkl', allow_pickle=True)
    viewpoint = viewpoints.poses[0]
    viewpoint = np.array([viewpoint.position.x, 
                        viewpoint.position.y, 
                        viewpoint.position.z, 
                        viewpoint.orientation.x, 
                        viewpoint.orientation.y, 
                        viewpoint.orientation.z, 
                        viewpoint.orientation.w])

    global_point_cloud = np.load('/home/jianchao/dataset/nbv_data/pc_on_different_viewpoints/viewpoints/plant1/xyr_0_0_0/0_pc.npy')
    global_point_cloud = global_point_cloud.reshape(-1, 6)[:, 0:3]

    point_cloud, matrix_world2len, matrix_base2len = transform_global2len(global_point_cloud, viewpoint)
    depth_image = convert_pc2depth(point_cloud)
    print(depth_image)



