#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Pose, Point, Quaternion


class CameraControl:
    def __init__(self):
        # Initialize.
        rospy.loginfo("[CameraControl] Initializing CameraControl.")
        # ROS service to get and set model state.
        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")
        # Parameters.
        self.camera_name = "l515"
        self.ref_frame = "world"

    def get_camera_pose(self):
        get_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )  ###GetModelState is a data type that saves the state of the model.
        model_state = get_state(self.camera_name, self.ref_frame)
        if model_state:
            return model_state
        else:
            print("[CameraControl] Failed to get model state")

    def move_camera_to_pose(self, pose):
        if not isinstance(pose, Pose):
            pose_ = Pose()
            pose_.position = Point(pose[0], pose[1], pose[2])
            pose_.orientation = Quaternion(pose[3], pose[4], pose[5], pose[6])
            pose = pose_
        state_msg = ModelState()
        state_msg.model_name = self.camera_name
        state_msg.pose = pose
        state_msg.reference_frame = self.ref_frame
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state_msg)
        except rospy.ServiceException as e:
            print("[CameraControl] Service call to set_model_state failed: ", e)


if __name__ == "__main__":

    rospy.init_node("camera_control")

    camera_control = CameraControl()
    camera_control.get_camera_pose()

    pose = Pose()
    pose.position.x = 1.0
    pose.position.y = 1.0
    pose.position.z = 1.0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    camera_control.move_camera_to_pose(pose)
