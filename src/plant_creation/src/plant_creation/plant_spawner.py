#!/usr/bin/env python3
import rospy
import rospkg
from gazebo_msgs.srv import *   ###internal package of ros. SpawnModel is kind of srv
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray ###internal package of ros
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.msg import ModelState
import pandas as pd
import numpy as np

class PlantSpawner:
    def __init__(self):
        # Initialize.
        rospy.loginfo("[PlantSpawner] Initializing PlantSpawner.")

        self.model_name = "tomato"
        self.ref_frame = "world"
        
        rospack = rospkg.RosPack()
        self.model_path = rospack.get_path("greenhouse_gazebo") + "/urdf/"  ####model _path saved all the urdf file paths, of tomatoes. 
        
        self.roi_path = '/home/jianchao/dataset/nbv_data/paper4/gt_nodes'
        self.petiole_pub = rospy.Publisher("petiole_marker", MarkerArray, queue_size=50)
        self.peduncle_pub = rospy.Publisher("peduncle_marker", MarkerArray, queue_size=50)
        self.fruits_pub = rospy.Publisher("fruits_marker", MarkerArray, queue_size=10)

    def get_linear_pose(self, plant_num:int):
        plant_poses = PoseArray()
        for i in range(plant_num):
            pose = Pose(position=Point(0, i*0.7, 0), orientation=Quaternion(0, 0, 0, 1))
            plant_poses.poses.append(pose)
        return plant_poses
    
    def set_model_name(self, model_name:str):
        self.model_name = model_name

    def spawn_list_plants(self, plants_list:list, plant_poses:PoseArray, print_log = True):
        rospy.wait_for_service('gazebo/spawn_sdf_model')

        for (plant, pose) in zip(plants_list, plant_poses.poses):
            try:
                spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
                spawner(
                model_name=self.model_name + str(plant),
                model_xml=open(

                    self.model_path + self.model_name + str(plant) + ".sdf",
                    "r",
                ).read(),
                robot_namespace="/map",
                initial_pose=pose,
                reference_frame="world",
                )
                if print_log:
                    print('spawning %s[%s] successfully'% (self.model_name, str(plant)))
            except rospy.ServiceException as e:
                print("Service call failed: ", e)

    def reset_pose(self, plant_idx, pose, print_log = True):
        state_msg = ModelState()
        state_msg.model_name = self.model_name + str(plant_idx)
        state_msg.pose = pose
        state_msg.reference_frame = self.ref_frame
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state_msg)
            if print_log:
                print('reset tomato[%s] successfully'% str(plant_idx))
        except rospy.ServiceException as e:
            print("[PlantSpawner] Service call to set_model_state failed: ", e)

    def set_pose(self, plant_idx, rotation, position, print_log = True):
        state_msg = ModelState()
        state_msg.model_name = self.model_name + str(plant_idx)
        state_msg.pose = Pose(position=Point(position[0]/100, position[1]/100, position[2]/100), orientation = Quaternion(*quaternion_from_euler(0, 0, rotation)))
        state_msg.reference_frame = self.ref_frame

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state_msg)
            if print_log:
                print('move tomato[%s] to pose[x:%s;y:%s;z:%s;r:%r]successfully'% (str(plant_idx), str(position[0]), str(position[1]), str(position[2]), str(rotation)))
        except rospy.ServiceException as e:
            print("[PlantSpawner] Service call to set_model_state failed: ", e)

    def spawn_plant(self, plant_idx, plant_rotation, plant_position = [0, 0, 0], print_log = True):
        # Set poses for plants.  use -0.024 if acc_pc is above the ground plan
        # plant_pose = Pose(position=Point(1, 0, 1.15), orientation = Quaternion(*quaternion_from_euler(0, 0, plant_rotation))) #plant_poses saved the poses of all plant 0-10 that you want to create
        
        self.plant_idx = plant_idx
        self.plant_rotation = plant_rotation
        plant_pose = Pose(position=Point(plant_position[0]/100, plant_position[1]/100, plant_position[2]/100), orientation = Quaternion(*quaternion_from_euler(0, 0, plant_rotation))) #plant_poses saved the poses of all plant 0-10 that you want to create
        # self.visual_roi(plant_idx, plant_rotation)
        plant_position = [str(i) for i in plant_position]
        plant_position = ''.join(plant_position)
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)  ###many methods to call a service. (1) spawner(para1, para2, para3) (2) spawner.call(para_class)
            spawner(
                model_name=self.model_name + str(plant_idx),
                model_xml=open(
                    self.model_path + self.model_name + str(plant_idx) + ".sdf",
                    "r",
                ).read(),
                robot_namespace="/map",
                initial_pose=plant_pose,
                reference_frame="world",
            )
            if print_log:
                print('spawning plants [%s] successfully'% str(self.model_name + str(plant_idx)))
        except rospy.ServiceException as e:
            print("Service call failed: ", e)
    
    def spawn_block(self, block_position = [-0.7, 0, 0.3]):
        block_pose = Pose(position=Point(block_position[0], block_position[1], block_position[2]), orientation = Quaternion(*quaternion_from_euler(0, 0, 0))) #plant_poses saved the poses of all plant 0-10 that you want to create
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)  ###many methods to call a service. (1) spawner(para1, para2, para3) (2) spawner.call(para_class)
            spawner(
                model_name='block',
                model_xml=open(
                    self.model_path + "/block.sdf",
                    "r",
                ).read(),
                robot_namespace="/map",
                initial_pose=block_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def delete_plants(self, plant_idx, print_log = False):
        rospy.wait_for_service('gazebo/delete_model')
        try:
            delete_model_service = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            # delete_model_service(model_name=self.model_name + str(plant_idx) + '_%s'% str(plant_rotation) + '_%s'%plant_position)
            delete_model_service(model_name=self.model_name + str(plant_idx))
            if print_log:
                print('delete tomato[%s] successfully'% (str(plant_idx)))
        except Exception as e:
            print("Service call failed: ", e)
    
    def visual_roi(self, plant_idx, rotation, position): ## moved to nbv_computor.py
        fruits = []
        radius = []
        rois = pd.read_csv(self.roi_path + '/plant%s_export_fruit.csv'%str(plant_idx))
        
        for roi in range(len(rois)):
            #read the fruits one by one and convert the pose using rotation and position and add to fruits list
            fruit = np.array([rois[' loc_x'][roi], rois[' loc_y'][roi], rois[' loc_z'][roi]])
            #rotate the fruit
            fruit = np.dot(np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), fruit)
            #translate the fruit
            fruit = fruit + np.array([position[0], position[1], position[2]])
            fruits.append(fruit)
            radius.append(rois[' radius'][roi])

        fruitArray = MarkerArray() 
        for id, (f, r) in enumerate(zip(fruits, radius)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker.type = Marker.SPHERE
            marker.id = id
            # Set the scale of the marker
            marker.scale.x = r*2
            marker.scale.y = r*2
            marker.scale.z = r*2
            # Set the color
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.2

            # Set the pose of the marker
            marker.pose.position.x = f[0]
            marker.pose.position.y = f[1]
            marker.pose.position.z = f[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            fruitArray.markers.append(marker)
        rospy.sleep(30.0)
        self.fruits_pub.publish(fruitArray)
    
    # def spawn_plant_rviz(self, plant_idx: int):
    #     from std_msgs.msg import String
    #     import subprocess
    #     import os
    #     # Load URDF from file
    #     urdf_path = os.path.join(self.model_path, self.model_name + str(plant_idx) + ".urdf")
    #     if not os.path.isfile(urdf_path):
    #         rospy.logerr(f"URDF file not found: {urdf_path}")
    #         return

    #     with open(urdf_path, "r") as file:
    #         urdf_xml = file.read()

    #     # Publish to /robot_description
    #     pub = rospy.Publisher("/robot_description", String, queue_size=1, latch=True)
    #     rospy.sleep(1.0)  # Give time for publisher to connect
    #     pub.publish(urdf_xml)
    #     rospy.loginfo(f"[RViz] Published URDF for plant[{plant_idx}]")

    #     # Start robot_state_publisher if not already running
    #     try:
    #         subprocess.Popen(["rosrun", "robot_state_publisher", "robot_state_publisher"])
    #         rospy.loginfo("[RViz] robot_state_publisher launched")
    #     except Exception as e:
    #         rospy.logwarn(f"[RViz] Failed to start robot_state_publisher: {e}")
            
if __name__ == "__main__":
    rospy.init_node('plant_spawner')
    # import random
    # from nbv_compute.nbv_computor import Nbvcomputor
    # nbv_compute = Nbvcomputor(octomap_resulotion=0.003)
    
    ps = PlantSpawner()
    # ps.spawn_block()
    # rospy.sleep(10.0)
    ps.set_model_name('plant')
    # plant_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # poses =  ps.get_linear_pose(len(plant_list))

    # for i, plant in enumerate(plant_list):
    #     pose = poses.poses[i]

    #     rotation = pose.orientation.z
    #     position = np.array([pose.position.x, pose.position.y, pose.position.z]) #position in meter
 
    # ps.spawn_plant(4, plant_rotation= 0, plant_position = [0,0,0])
    ps.spawn_plant_rviz(4)
        # rospy.sleep(3.0)


        # ps.delete_plants(i)

        # ps.visual_roi(i, rotation, position)
        # rospy.sleep(3.0)

    #     nbv_compute.load_complete_pc(
    #     'plant',
    #     plant_list[i],
    #     plant_rotation=rotation,
    #     plant_position=position*100,
    #     scale = 1,
    #     visual_com=True,
    # )
    #     rospy.sleep(100.0)


        # ps.delete_plants(i)
    # ps.spawn_plant(plant_list[0], plant_rotation= 0, plant_position = [0, 0, 0])
    # pose =  ps.get_linear_pose(len(plant_list))
    # ps.spawn_list_plants(plant_list, pose)
    # for i in plant_list:
    #     ps.delete_plants(i)
    #     ps.spawn_plant(20, plant_rotation= 0, plant_position = [0, 0, 0])

    #     ps.delete_plants(i)

    # plant_list = [5]
    # position = [0, 0, 0]
    # rotation = 0
    # # ps.spawn_plant(plant_list[0], plant_rotation= rotation, plant_position = position)

    # ps.visual_roi(plant_list[0], rotation, position)

    
   


    



