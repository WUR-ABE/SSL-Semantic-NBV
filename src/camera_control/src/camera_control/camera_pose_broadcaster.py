#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion
from camera_control.camera_control import CameraControl

class CameraPoseBroadcaster:
    def __init__(self):
        rospy.loginfo("[CameraPoseBroadcaster] Initializing CameraPoseBroadcaster.")
        self.br = tf2_ros.TransformBroadcaster()
        self.camera_frame = "base_link" 
        self.ref_frame = "world"

    def broadcast(self, camera_pose):
        if not isinstance(camera_pose, Pose):
            camera_pose_ = Pose()
            camera_pose_.position = Point(camera_pose[0], camera_pose[1], camera_pose[2])
            camera_pose_.orientation = Quaternion(camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6])
            camera_pose = camera_pose_

        t = TransformStamped()
        t.header.stamp = rospy.Time()
        t.header.frame_id = self.ref_frame
        t.child_frame_id = self.camera_frame
        t.transform.translation = camera_pose.position
        t.transform.rotation = camera_pose.orientation
        self.br.sendTransform(t)

if __name__ == "__main__":

    rospy.init_node("camera_pose_broadcaster")

    camera_pose_broadcaster = CameraPoseBroadcaster()

    while not rospy.is_shutdown():
        camera_pose_broadcaster.broadcast()










    

