/*
 Author: Akshay Kumar Burusa
 Email: akshaykumar.burusa@wur.nl
 */

#ifndef ARM_CONTROL_HPP
#define ARM_CONTROL_HPP

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <flexcraft_msgs/ArmGoal.h>

namespace arm_control
{
class ArmControl
{
protected:
  // MoveIt.
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

public:
  /* Constructor.
   * Inputs: Name of MoveIt planning group, Usecase */
  ArmControl(std::string, std::string);
  // Destructor.
  ~ArmControl();
  /* Move end-effector to goal pose.
   * Output: Status of pose update */
  bool MoveToGoal(geometry_msgs::Pose);
  /* Service to move end-effector to goal pose.
   * Input: ArmGoal request, ArmGoal response
   * Output: Status of pose update */
  bool MoveToGoalSrv(flexcraft_msgs::ArmGoal::Request&, flexcraft_msgs::ArmGoal::Response&);
  /* Set collision constraints.
   * Input: Array of sizes of collision object, Array of poses of collision object */
  void AddCollisionObjects(std::vector<geometry_msgs::Vector3>, geometry_msgs::PoseArray);
};
}  // namespace arm_control

#endif  // ARM_CONTROL_HPP
