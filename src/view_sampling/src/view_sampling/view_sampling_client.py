#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from flexcraft_msgs.srv import SemiCylinder
import pickle
from visualization_msgs.msg import Marker, MarkerArray

class ViewSamplerClient:
    def __init__(self, save_pose = False):
        self.save_pose = save_pose
        # Initialize the subscribers/publishers
        rospy.loginfo("[ViewSamplerClient] Initializing view sampler client python")
        # Check / wait for communication with view sampling module to become operational
        rospy.loginfo(
            "[ViewSamplerClient] Waiting for the sercives to become available"
        )
        rospy.wait_for_service("view_sampler_service")
        rospy.loginfo("[ViewSamplerClient] View sampler client node has started")
        # Initialize the services
        self.ref_frame = "world"
        self.view_sampler_service = rospy.ServiceProxy(
            "view_sampler_service", SemiCylinder
        )
        self.marker_pub = rospy.Publisher(
            "visual_candidates", MarkerArray, queue_size=50
        )
        self.get_candidate_poses()

    def get_candidate_poses(self):
        load_candidates_from_local = rospy.get_param("/view_sampling/load_candidates_from_local")
        local_candidates_path = rospy.get_param("/view_sampling/local_candidates_path")

        if load_candidates_from_local:
            with open(local_candidates_path, 'rb') as f:
                self.candidate_views = pickle.load(f)
        else:
            sampler_type = rospy.get_param("/view_sampling/sampling_type")
            num_samples = rospy.get_param("/view_sampling/num_samples")
            center = Point(
                rospy.get_param("/view_sampling/center/x"),
                rospy.get_param("/view_sampling/center/y"),
                rospy.get_param("/view_sampling/center/z"),
            )

            length = rospy.get_param("/view_sampling/length")
            radius = rospy.get_param("/view_sampling/radius")
            min_theta = rospy.get_param("/view_sampling/min_theta")
            max_theta = rospy.get_param("/view_sampling/max_theta")

            self.candidate_views = self.sample_views(
            sampler_type,
            num_samples,
            center,
            radius,
            min_theta,
            max_theta,
            length
            )

            if self.save_pose:
                with open(local_candidates_path, 'wb') as f:
                    pickle.dump(self.candidate_views, f)

    def sample_views(
        self,
        sampler_type,
        num_samples=33,
        center=Point(0.0, 0.0, 0.3),
        radius=0.6,
        min_theta=0.0,
        max_theta=360.0,
        length = 0.85
    ):
        try:
            response = self.view_sampler_service(
                sampler_type,
                num_samples,
                center,
                radius,
                min_theta,
                max_theta,
                length,
            )
            rospy.loginfo("[ViewSamplerClient] Succeeded to sample views")
            return response.view_samples
        except rospy.ServiceException as e:
            rospy.loginfo("[ViewSamplerClient] Failed to sample views")
            print("Service call failed: %s" % e)
            return False

    def pose2marker(self, poses):
        
        markerArray = MarkerArray() 

        for id, pose in enumerate(poses):
            marker = Marker()
            marker.header.frame_id = self.ref_frame
            marker.id = id
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.scale.x = 0.03
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose = pose
            markerArray.markers.append(marker)
        return markerArray

    def visual_candidates(self, nbv_index: int = None):
        markerArray = self.pose2marker(self.candidate_views.poses)
        if nbv_index is not None:
            markerArray.markers[nbv_index].color.r = 0.0
            markerArray.markers[nbv_index].color.g = 1.0
            markerArray.markers[nbv_index].color.b = 0.0
        self.marker_pub.publish(markerArray)
