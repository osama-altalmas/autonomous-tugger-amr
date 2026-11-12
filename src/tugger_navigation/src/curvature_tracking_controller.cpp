#include "tugger_navigation/curvature_tracking_controller.hpp"

#include <cmath>
#include <vector>

#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.hpp"

namespace tugger_navigation
{

void CurvatureTrackingController::configure(
  const nav2::LifecycleNode::WeakPtr & parent,
  std::string name, nav2::TransformBuffer::SharedPtr tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  RegulatedPurePursuitController::configure(parent, name, tf, costmap_ros);
  auto node = parent.lock();
  if (!node) {
    throw std::runtime_error("CurvatureTrackingController: missing lifecycle node");
  }
  tracking_name_ = name;
  tracking_logger_ = node->get_logger();
  clock_ = node->get_clock();
  auto declare_double = [&](const std::string & key, double & value) {
      const std::string parameter = name + "." + key;
      if (!node->has_parameter(parameter)) {
        node->declare_parameter(parameter, value);
      }
      node->get_parameter(parameter, value);
    };
  declare_double("tracking_straight_lookahead", gains_.straight_lookahead);
  declare_double("tracking_curve_lookahead", gains_.curve_lookahead);
  declare_double("tracking_schedule_low_curvature", gains_.schedule_low_curvature);
  declare_double("tracking_schedule_high_curvature", gains_.schedule_high_curvature);
  declare_double("tracking_schedule_ahead", gains_.schedule_ahead);
  declare_double("tracking_schedule_behind", gains_.schedule_behind);
  declare_double("tracking_preview_time", gains_.preview_time);
  declare_double("tracking_curvature_window", gains_.curvature_window);
  declare_double("tracking_max_curvature", gains_.max_curvature);
  declare_double("tracking_max_heading_error", max_heading_error_);
  declare_double("curve_speed", gains_.curve_speed);
  declare_double("curve_speed_curvature", gains_.curve_speed_curvature);
  declare_double("curve_speed_deceleration", gains_.speed_deceleration);
  declare_double("curve_speed_margin", gains_.speed_margin);
  declare_double("curve_speed_scan", gains_.speed_scan);
  if (gains_.straight_lookahead <= 0.0 || gains_.curve_lookahead <= 0.0 ||
    gains_.schedule_high_curvature < gains_.schedule_low_curvature ||
    gains_.schedule_ahead <= 0.0 || gains_.schedule_behind < 0.0 ||
    gains_.preview_time < 0.0 || gains_.curvature_window <= 0.0 ||
    gains_.max_curvature <= 0.0 || max_heading_error_ <= 0.0 ||
    gains_.curve_speed <= 0.0 || gains_.curve_speed_curvature <= 0.0 ||
    gains_.speed_deceleration <= 0.0 || gains_.speed_margin < 0.0 ||
    gains_.speed_scan < 0.0)
  {
    throw std::runtime_error("CurvatureTrackingController: invalid tracking parameters");
  }
  tracking_pub_ = node->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/tugger/path_tracking", nav2::qos::StandardTopicQoS());
  RCLCPP_INFO(
    tracking_logger_,
    "%s: curvature tracking, look-ahead %.2f m straight / %.2f m curves, "
    "preview %.2f s, curve speed %.2f m/s", name.c_str(), gains_.straight_lookahead,
    gains_.curve_lookahead, gains_.preview_time, gains_.curve_speed);
}

void CurvatureTrackingController::activate()
{
  RegulatedPurePursuitController::activate();
  tracking_pub_->on_activate();
}

void CurvatureTrackingController::deactivate()
{
  tracking_pub_->on_deactivate();
  RegulatedPurePursuitController::deactivate();
}

void CurvatureTrackingController::newPathReceived(const nav_msgs::msg::Path & raw_global_path)
{
  RegulatedPurePursuitController::newPathReceived(raw_global_path);
  last_signal_ = 0.0;
  have_last_position_ = false;
}

geometry_msgs::msg::TwistStamped CurvatureTrackingController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker,
  const nav_msgs::msg::Path & transformed_global_plan,
  const geometry_msgs::msg::PoseStamped & global_goal)
{
  // Pure pursuit decides speed, direction, collision stops and the approach
  // slow-down; it throws on a collision ahead, as before.
  auto command = RegulatedPurePursuitController::computeVelocityCommands(
    pose, velocity, goal_checker, transformed_global_plan, global_goal);
  const double v = command.twist.linear.x;
  if (std::abs(v) < 1e-6) {
    return command;
  }
  if (transformed_global_plan.header.frame_id != pose.header.frame_id) {
    RCLCPP_WARN_THROTTLE(
      tracking_logger_, *clock_, 5000,
      "%s: plan frame '%s' differs from pose frame '%s'; using pure pursuit steering",
      tracking_name_.c_str(), transformed_global_plan.header.frame_id.c_str(),
      pose.header.frame_id.c_str());
    return command;
  }

  std::vector<path_tracking::Point2D> path;
  path.reserve(transformed_global_plan.poses.size());
  for (const auto & p : transformed_global_plan.poses) {
    path.push_back({p.pose.position.x, p.pose.position.y});
  }
  const double yaw = tf2::getYaw(pose.pose.orientation);
  const double travel_yaw = v < 0.0 ? yaw + M_PI : yaw;
  const double x = pose.pose.position.x;
  const double y = pose.pose.position.y;
  const double travelled = have_last_position_ ? std::hypot(x - last_x_, y - last_y_) : 0.0;
  const double remembered = path_tracking::remembered_signal(last_signal_, travelled, gains_);
  const auto tracking = path_tracking::track_path(path, x, y, travel_yaw, v, gains_, remembered);
  const double pure_pursuit_curvature = command.twist.angular.z / v;
  if (!tracking.valid || std::abs(tracking.heading_error) > max_heading_error_) {
    // No usable section, or the travel direction disagrees with the path:
    // keep the pure pursuit command.
    RCLCPP_WARN_THROTTLE(
      tracking_logger_, *clock_, 5000,
      "%s: tracking unavailable (valid %d, heading error %.2f rad); using pure pursuit",
      tracking_name_.c_str(), tracking.valid, tracking.heading_error);
    return command;
  }
  last_signal_ = tracking.schedule_signal;
  last_x_ = x;
  last_y_ = y;
  have_last_position_ = true;

  // Planned curve speed: never faster than pure pursuit allows.
  const double speed = std::min(std::abs(v), tracking.speed_limit);
  command.twist.linear.x = std::copysign(speed, v);
  // Positive curvature turns left relative to the travel direction, which is
  // a positive yaw rate whether the robot drives front or rear first.
  command.twist.angular.z = speed * tracking.curvature;

  if (tracking_pub_ && tracking_pub_->is_activated()) {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {tracking.lateral_error, tracking.heading_error, tracking.path_curvature,
      tracking.curvature, pure_pursuit_curvature, tracking.lookahead,
      std::isfinite(tracking.speed_limit) ? tracking.speed_limit : -1.0,
      command.twist.linear.x, tracking.path_ahead};
    tracking_pub_->publish(msg);
  }
  return command;
}

}  // namespace tugger_navigation

PLUGINLIB_EXPORT_CLASS(tugger_navigation::CurvatureTrackingController, nav2_core::Controller)
