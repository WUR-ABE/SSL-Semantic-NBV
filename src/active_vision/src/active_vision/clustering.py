import open3d as o3d
import numpy as np


class Clusterer:
    """
    Cluster point cloud based on the semantic class
    """

    def __init__(self) -> None:
        """
        Initialize the clusterer
        """
        self.eps = 0.02
        self.min_points = 10

    def cluster(self, points: np.array, semantics: np.array) -> np.array:
        """
        Cluster point cloud based on the semantic class
        :param points: point cloud (N, 3)
        :param semantics: semantic class (N)
        :return: cluster labels (N)
        """
        # Separate the point cloud based on the semantic class
        clouds = []
        for class_id in np.unique(semantics):
            # Create an Open3D point cloud
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points[semantics == class_id])
            clouds.append(cloud)
        # Cluster each point cloud separately using Open3D's DBSCAN
        labels = np.zeros_like(semantics).astype(int)
        for i, cloud in enumerate(clouds):
            # Cluster the point cloud
            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                cloud_labels = np.array(
                    cloud.cluster_dbscan(
                        eps=self.eps, min_points=self.min_points, print_progress=True
                    )
                )
            # Update the labels
            labels[semantics == i] = cloud_labels.max() + 1
        # Visualize the clusters in Open3D
        if labels.shape[0] > 0:
            colors = np.random.uniform(0, 1, size=(int(labels.max()) + 1, 3))
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors[labels])
            o3d.visualization.draw_geometries([cloud])
        return labels
