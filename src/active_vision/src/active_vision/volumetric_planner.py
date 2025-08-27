import torch
import numpy as np

from active_vision.voxel_grid_volumetric import VoxelGrid
from utils.rviz_visualizer import RvizVisualizer
from utils.utils import numpy_to_pose, numpy_to_pose_array
from scipy.spatial.transform import Rotation as R


class ActivePlanner:
    """
    Class to plan active viewpoints
    """
        #self.image_size = np.array([960, 540])
        #self.intrinsics = np.array([
         #   [1135.306761618773, 0.0, 480.5],
          #  [0.0, 1135.306761618773, 270.5],
           # [0.0, 0.0, 1.0]])

    def __init__(self) -> None:
        self.image_size = np.array([960, 540])
        self.intrinsics = np.array([
            [1135.306761618773, 0.0, 480.5],
            [0.0, 1135.306761618773, 270.5],
            [0.0, 0.0, 1.0]])
        
        self.num_pts_per_ray = 128
        self.num_features = 4
        self.save_path = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rviz_visualizer = RvizVisualizer()
    
    def init_voxel_grid(
                        self,
                        grid_size: np.array = np.array([0.6, 0.6, 0.6]),
                        voxel_size: np.array = np.array([0.003]),
                        grid_center: np.array = np.array([0.0, 0.0, 0.3])):
        
        self.grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        self.grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.voxel_grid = VoxelGrid(
            grid_size=self.grid_size,
            voxel_size=voxel_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            num_pts_per_ray=self.num_pts_per_ray,
            num_features=self.num_features,
            device=self.device,
            save_path=self.save_path,
        )

    def get_tranform_matrix(self, translation, orientation):
        trans_world2base = np.array([translation[0], translation[1], translation[2]])  # [x, y, z]
        rotate_world2base = np.array([-orientation[1],-orientation[2],-orientation[3],orientation[0]])  # Quaternion [x, y, z, w]
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

    def convert_pc2depth(self, point_cloud_global: np.array, viewpoint: np.array, mode='xyz'):
        translation = viewpoint[0:3]
        orientation = viewpoint[3:]

        if mode == 'xyz':
            xyz = point_cloud_global.detach().cpu().numpy().reshape(-1, 3)
        elif mode == 'xyzrgbc':
            point_cloud_global = point_cloud_global.detach().cpu().numpy().reshape(-1, 7)  # xyzrgbc
            xyz = point_cloud_global[:, :3]
            classes = point_cloud_global[:, -1]  # Extract the class (c)

        homogeneous_points = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        matrix_world2len, matrix_base2len = self.get_tranform_matrix(translation, orientation)
        transformed_xyz = np.dot(matrix_world2len, homogeneous_points.T).T[:, :3] 

        x, y, z = transformed_xyz.T
        projected_points = np.dot(self.intrinsics, np.array([x, y, z]))
        pixel_x = (projected_points[0] / projected_points[2]).astype(int)
        pixel_y = (projected_points[1] / projected_points[2]).astype(int)

        valid_mask = (pixel_x >= 0) & (pixel_x < self.image_size[0] ) & (pixel_y >= 0) & (pixel_y < self.image_size[1])
        depth_image = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
        if mode == 'xyzrgbc':
            # semantics_image = np.zeros((self.image_size[1], self.image_size[0], 2), dtype=np.float32)
            # occupied_odds = self.logodds(0.9)
            # free_odds = self.logodds(0.4)

            score_mask = np.zeros((self.image_size[1], self.image_size[0], 1))
            label_mask = np.zeros((self.image_size[1], self.image_size[0], 1))

            classes[classes == -1] = 0
            label_mask[pixel_y[valid_mask], pixel_x[valid_mask], 0] = classes[valid_mask]
            score_mask[label_mask == 1] = 0.7

            # score_mask[label_mask == 0] = occupied_odds
            semantics_image = np.concatenate((score_mask, label_mask), axis=2)
        
        depth_image[pixel_y[valid_mask], pixel_x[valid_mask]] = z[valid_mask]
        depth_image = np.where(depth_image <= 0, np.nan, depth_image)

        if mode == 'xyz':
            return depth_image
        
        elif mode == 'xyzrgbc':
            # semantics_image = np.where(semantics_image <= 0, np.nan, semantics_image)
            return depth_image, semantics_image
    
    def logodds(self, p):
        return np.log(p/(1-p))
    
    def depth_to_voxels(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth image and insert it into a voxel grid
        :param depth_image: depth image (H, W)
        :param semantics: confidence scores and class ids (H, W, 2)
        :param viewpoint: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(
            viewpoint[3:], dtype=torch.float32, device=self.device
        )
        self.voxel_grid.insert_point_cloud(
            depth_image, semantics, position, orientation
        )

    def active_view(self, view_samples) -> np.array:
        """
        Sample a viewpoint that maximizes the information gain
        """

        self.rviz_visualizer.visualize_view_samples(
            numpy_to_pose_array(view_samples[:, :7])
        )
        camera_poses = torch.tensor(
            view_samples, dtype=torch.float32, device=self.device
        )
        best_gain = -np.inf
        nbv = None 
        nbv_idx = None
        for i, view in enumerate(view_samples):
            with torch.no_grad():
                _, gain, _ = self.voxel_grid.compute_gain_volumetric(
                    camera_poses[i, :3], camera_poses[i, 3:]
                )
                gain = gain.detach().cpu().numpy()
                if gain > best_gain:
                    best_gain = gain
                    nbv = view
                    nbv_idx = i
        self.viewpoint = nbv[:7]
        self.target_position = nbv[7:]
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(self.viewpoint))
        return self.viewpoint, nbv_idx, 1.0 - best_gain, len(view_samples)

    def get_voxel_point(self):
        voxel_points, semantic_points = self.voxel_grid.get_occupied_points()
        return voxel_points.unsqueeze(0)

    def visualize(self):
        """
        Visualize the voxel grid as a point cloud in rviz
        """
        voxel_points, semantic_points = self.voxel_grid.get_occupied_points()
        voxel_points = voxel_points.cpu().numpy()
        semantic_points = semantic_points.cpu().numpy()
        self.rviz_visualizer.visualize_voxels(voxel_points, semantic_points)
        self.rviz_visualizer.visualize_bbox(center = self.grid_center, grid_size = self.grid_size)
