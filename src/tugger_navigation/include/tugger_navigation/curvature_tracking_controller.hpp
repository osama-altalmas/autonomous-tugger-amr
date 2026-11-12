// Route follower: Nav2 Regulated Pure Pursuit for speed, direction, collision
// checking and the approach slow-down, with the turning command replaced by
// path-curvature tracking and the curve speed planned from the path (see
// path_tracking_law.hpp).

#ifndef TUGGER_NAVIGATION__CURVATURE_TRACKING_CONTROLLER_HPP_
#define TUGGER_NAVIGATION__CURVATURE_TRACKING_CONTROLLER_HPP_

#include <memory>
#include <string>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "tugger_navigation/path_tracking_law.hpp"

namespace tugger_navigation
{

class CurvatureTrackingController
  : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  void configure(
    const nav2::LifecycleNode::WeakPtr & parent,
    std::string name, nav2::TransformBuffer::SharedPtr tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void activate() override;
  void deactivate() override;

  void newPathReceived(const nav_msgs::msg::Path & raw_global_path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker,
    const nav_msgs::msg::Path & transformed_global_plan,
    const geometry_msgs::msg::PoseStamped & global_goal) override;

private:
  path_tracking::TrackingGains gains_;
  double max_heading_error_ = 1.05;
  // Schedule signal of the last cycle and where the robot was then.
  double last_signal_ = 0.0;
  bool have_last_position_ = false;
  double last_x_ = 0.0;
  double last_y_ = 0.0;
  std::string tracking_name_;
  rclcpp::Logger tracking_logger_ {rclcpp::get_logger("CurvatureTrackingController")};
  rclcpp::Clock::SharedPtr clock_;
  // [lateral error m, heading error rad, path curvature, commanded curvature,
  //  pure pursuit curvature, equivalent look-ahead m, planned speed limit m/s
  //  (-1 none), commanded speed m/s, path ahead m]
  nav2::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tracking_pub_;
};

}  // namespace tugger_navigation

#endif  // TUGGER_NAVIGATION__CURVATURE_TRACKING_CONTROLLER_HPP_
