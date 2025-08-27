import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import Point, Quaternion, Pose, PoseArray, PoseStamped
from scipy.spatial.transform import Rotation as scipy_r
from pytransform3d.transformations import transform_from_pq, pq_from_transform


def lookup_transform(frame1, frame2):
    # TF2
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    try:
        trans = tfBuffer.lookup_transform(
            frame1, frame2, rospy.Time(), rospy.Duration(0.5)
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        rospy.logwarn("[PyUtils] Transform lookup failed.")
        return False
    else:
        return trans


def T_from_rot_trans(rot, trans):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3:] = trans.T
    return T


def point_to_numpy(point):
    return np.array([point.x, point.y, point.z])


def pose_to_numpy(view):
    return np.array(
        [
            view.position.x,
            view.position.y,
            view.position.z,
            view.orientation.w,
            view.orientation.x,
            view.orientation.y,
            view.orientation.z,
        ]
    )


def numpy_to_pose(view):
    return Pose(
        Point(view[0], view[1], view[2]),
        Quaternion(view[4], view[5], view[6], view[3]),
    )


def pose_array_to_numpy(views):
    views_np = []
    for view in views.poses:
        views_np.append(pose_to_numpy(view))
    return np.array(views_np)


def numpy_to_pose_array(views):
    view_poses = PoseArray()
    for view in views:
        view_poses.poses.append(numpy_to_pose(view))
    return view_poses


def look_at_rotation(
    eye: np.array, target: np.array, ref: np.array = [1.0, 0.0, 0.0]
) -> np.array:
    dir = target - eye
    dir = dir / np.linalg.norm(dir)

    rot_axis = np.cross(ref, dir)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(ref, dir))

    if np.isnan(rot_axis).any():
        return np.array([1.0, 0.0, 0.0, 0.0])

    quat = axangle2quat(rot_axis, rot_angle, True)
    return quat


def axangle2quat(vector: np.array, theta: float, is_normalized=False):
    if not is_normalized:
        vector = vector / np.linalg.norm(vector)
    t2 = theta / 2.0
    st2 = np.sin(t2)
    return np.concatenate(([np.cos(t2)], vector * st2))


def optical_to_world(
    pose_o: np.array, rot_euler: np.array = [np.pi / 2, 0.0, np.pi / 2]
):
    # Convert pose from optical frame to world frame
    r = scipy_r.from_euler("zyx", rot_euler, degrees=False)
    T_co = T_from_rot_trans(r.as_matrix(), np.zeros((1, 3)))
    T_ow = transform_from_pq(pose_o)
    T_cw = T_ow @ T_co
    pose_w = pq_from_transform(T_cw)
    return pose_w


def transform_pose(frame, pose, transform):
    pose_stamped = PoseStamped()
    pose_stamped.pose = pose
    pose_stamped.header.frame_id = frame
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped_t = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
    return pose_stamped_t.pose
