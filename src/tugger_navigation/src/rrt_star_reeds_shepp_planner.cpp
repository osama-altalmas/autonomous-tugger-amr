#include <cstdio>
#include "tugger_navigation/rrt_star_reeds_shepp_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace tugger_navigation
{

void RRTStarReedsSheppPlanner::configure(
  const nav2::LifecycleNode::WeakPtr & parent,
  std::string name,
  nav2::TransformBuffer::SharedPtr tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  (void)tf;

  parent_ = parent;
  name_ = std::move(name);
  costmap_ros_ = std::move(costmap_ros);

  auto node = parent_.lock();

  if (!node || !costmap_ros_) {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: missing lifecycle node or costmap");
  }

  auto declare_double = [&](const std::string & key, double & value) {
    const std::string parameter = name_ + "." + key;

    if (!node->has_parameter(parameter)) {
      node->declare_parameter(parameter, value);
    }

    node->get_parameter(parameter, value);
  };

  auto declare_int = [&](const std::string & key, int & value) {
    const std::string parameter = name_ + "." + key;

    if (!node->has_parameter(parameter)) {
      node->declare_parameter(parameter, value);
    }

    node->get_parameter(parameter, value);
  };

  declare_double("turning_radius", turning_radius_);
  declare_double("sample_resolution", sample_resolution_);
  declare_double("max_extension_length", max_extension_length_);
  declare_double("neighbor_radius", neighbor_radius_);
  declare_double("goal_bias", goal_bias_);
  declare_double("planning_time_seconds", planning_time_seconds_);
  declare_int("max_iterations", max_iterations_);

  if (turning_radius_ <= 0.0 ||
      sample_resolution_ <= 0.0 ||
      max_extension_length_ <= 0.0 ||
      neighbor_radius_ <= 0.0 ||
      goal_bias_ < 0.0 || goal_bias_ > 1.0 ||
      planning_time_seconds_ <= 0.0 ||
      max_iterations_ <= 0)
  {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: invalid configuration");
  }

  // Keep the latest tree available for RViz subscribers,
  // including subscribers that connect after planning ends.
  tree_marker_pub_ =
    node->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/rrt_star/tree",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

  RCLCPP_INFO(
    node->get_logger(),
    "RRT* Reeds-Shepp planner configured: radius=%.3f, "
    "resolution=%.3f, iterations=%d, budget=%.2fs",
    turning_radius_, sample_resolution_,
    max_iterations_, planning_time_seconds_);
}

void RRTStarReedsSheppPlanner::cleanup()
{
  tree_marker_pub_.reset();
  costmap_ros_.reset();
  parent_.reset();
}

void RRTStarReedsSheppPlanner::activate()
{
}

void RRTStarReedsSheppPlanner::deactivate()
{
}

nav_msgs::msg::Path RRTStarReedsSheppPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  const std::vector<geometry_msgs::msg::PoseStamped> & viapoints,
  std::function<bool()> cancel_checker)
{
  (void)viapoints;

  auto node = parent_.lock();

  if (!node || !costmap_ros_) {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: not configured");
  }

  const std::string frame = costmap_ros_->getGlobalFrameID();

  if (start.header.frame_id != frame ||
      goal.header.frame_id != frame)
  {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: start/goal frame mismatch");
  }

  const auto * costmap = costmap_ros_->getCostmap();

  if (!costmap) {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: null costmap");
  }

  const auto yaw_of = [](const geometry_msgs::msg::Quaternion & q) {
    return std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  };

  const Pose2D start_pose{
    start.pose.position.x,
    start.pose.position.y,
    yaw_of(start.pose.orientation)
  };

  const Pose2D goal_pose{
    goal.pose.position.x,
    goal.pose.position.y,
    yaw_of(goal.pose.orientation)
  };

  const double min_x = costmap->getOriginX();
  const double min_y = costmap->getOriginY();

  const double max_x =
    min_x + costmap->getSizeInMetersX();

  const double max_y =
    min_y + costmap->getSizeInMetersY();

  const SamplingBounds bounds{
    min_x, max_x, min_y, max_y
  };

  using FootprintChecker =
    nav2_costmap_2d::FootprintCollisionChecker<
      nav2_costmap_2d::Costmap2D *>;

  FootprintChecker footprint_checker(
    costmap_ros_->getCostmap());

  const auto footprint = costmap_ros_->getRobotFootprint();

  if (footprint.size() < 3) {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: invalid robot footprint");
  }

  const auto pose_is_free =
    [&footprint_checker, &footprint, costmap](
      const Pose2D & pose)
    {
      unsigned int mx = 0;
      unsigned int my = 0;

      if (!costmap->worldToMap(pose.x, pose.y, mx, my)) {
        return false;
      }

      const double footprint_cost =
        footprint_checker.footprintCostAtPose(
          pose.x,
          pose.y,
          pose.yaw,
          footprint);

      const bool free =
        footprint_cost >= 0.0 &&
        footprint_cost <
        nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;

      static int rejected_samples_logged = 0;

      if (!free && rejected_samples_logged < 10) {
        ++rejected_samples_logged;

        std::fprintf(
          stderr,
          "[RRT* FOOTPRINT REJECT] "
          "x=%.3f y=%.3f yaw=%.3f "
          "center_cost=%u footprint_cost=%.1f\\n",
          pose.x, pose.y, pose.yaw,
          static_cast<unsigned int>(costmap->getCost(mx, my)),
          footprint_cost);
      }

      return free;
    };

  const auto log_pose_cost = [&](const char * label, const Pose2D & pose) {
    unsigned int mx = 0;
    unsigned int my = 0;

    const bool in_map = costmap->worldToMap(pose.x, pose.y, mx, my);

    const double footprint_cost = in_map ?
      footprint_checker.footprintCostAtPose(
        pose.x, pose.y, pose.yaw, footprint) : -1.0;

    const int center_cost = in_map ?
      static_cast<int>(costmap->getCost(mx, my)) : -1;

    RCLCPP_WARN(
      node->get_logger(),
      "RRT* %s: x=%.2f y=%.2f yaw=%.2f "
      "in_map=%d center_cost=%d footprint_cost=%.1f free=%d",
      label, pose.x, pose.y, pose.yaw,
      static_cast<int>(in_map), center_cost,
      footprint_cost, static_cast<int>(pose_is_free(pose)));
  };

  log_pose_cost("START", start_pose);
  log_pose_cost("GOAL", goal_pose);

  ReedsSheppSteering steering(
    turning_radius_, sample_resolution_);

  RRTStarCost cost_model;

  RRTStarSampler sampler(bounds, goal_bias_);

  RRTStarExpansion expansion(
    steering,
    cost_model,
    sampler,
    max_extension_length_,
    neighbor_radius_);

  RRTStarPlannerCore planner(
    steering,
    cost_model,
    sampler,
    expansion);

  const auto budget = std::chrono::duration_cast<
    std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(planning_time_seconds_));

  // Publish the actual RRT* tree, including curved Reeds-Shepp edges.
  // A complete snapshot replaces the previous visualization.
  const auto visualize_tree =
    [this, node, frame, start_pose, goal_pose](
      const RRTStarTree & tree, std::size_t iterations)
    {
      if (!tree_marker_pub_) {
        return;
      }

      using Marker = visualization_msgs::msg::Marker;
      visualization_msgs::msg::MarkerArray markers;

      Marker clear;
      clear.header.frame_id = frame;
      clear.header.stamp = node->now();
      clear.action = Marker::DELETEALL;
      markers.markers.push_back(clear);

      auto make_marker = [&](int id, int type) {
        Marker marker;
        marker.header.frame_id = frame;
        marker.header.stamp = node->now();
        marker.ns = "rrt_star";
        marker.id = id;
        marker.type = type;
        marker.action = Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.color.a = 1.0;
        return marker;
      };

      Marker edges = make_marker(1, Marker::LINE_LIST);
      edges.scale.x = 0.015;
      edges.color.r = 0.12f;
      edges.color.g = 0.55f;
      edges.color.b = 1.0f;

      Marker nodes = make_marker(2, Marker::POINTS);
      nodes.scale.x = 0.045;
      nodes.scale.y = 0.045;
      nodes.color.r = 0.0f;
      nodes.color.g = 0.9f;
      nodes.color.b = 1.0f;

      auto point_of = [](const Pose2D & pose) {
        geometry_msgs::msg::Point point;
        point.x = pose.x;
        point.y = pose.y;
        point.z = 0.08;
        return point;
      };

      for (std::size_t i = 0; i < tree.size(); ++i) {
        const auto & tree_node = tree.at(static_cast<int>(i));
        nodes.points.push_back(point_of(tree_node.pose));

        if (tree_node.parent < 0) {
          continue;
        }

        const auto & samples = tree_node.incoming_connection.samples;

        for (std::size_t j = 1; j < samples.size(); ++j) {
          edges.points.push_back(point_of(samples[j - 1].pose));
          edges.points.push_back(point_of(samples[j].pose));
        }
      }

      markers.markers.push_back(std::move(edges));
      markers.markers.push_back(std::move(nodes));

      auto add_endpoint = [&](int id, const Pose2D & pose,
                              float red, float green, float blue) {
        Marker marker = make_marker(id, Marker::SPHERE);
        marker.pose.position = point_of(pose);
        marker.scale.x = 0.13;
        marker.scale.y = 0.13;
        marker.scale.z = 0.13;
        marker.color.r = red;
        marker.color.g = green;
        marker.color.b = blue;
        markers.markers.push_back(std::move(marker));
      };

      add_endpoint(3, start_pose, 0.0f, 1.0f, 0.2f);
      add_endpoint(4, goal_pose, 1.0f, 0.65f, 0.0f);

      tree_marker_pub_->publish(markers);

      RCLCPP_INFO(
        node->get_logger(),
        "RRT* visualization: iterations=%zu, nodes=%zu, edges=%zu",
        iterations, tree.size(),
        markers.markers[1].points.size() / 2);
    };

  const auto result = planner.plan(
    start_pose,
    goal_pose,
    static_cast<std::size_t>(max_iterations_),
    pose_is_free,
    {},
    cancel_checker,
    budget,
    visualize_tree);

  if (result.cancelled) {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: planning cancelled");
  }

  if (!result.success || result.trajectory.empty()) {
    throw std::runtime_error(
      "RRTStarReedsSheppPlanner: no feasible path found");
  }

  nav_msgs::msg::Path path;
  path.header.frame_id = frame;
  path.header.stamp = node->now();

  path.poses.reserve(result.trajectory.size());

  for (const auto & sample : result.trajectory) {
    geometry_msgs::msg::PoseStamped pose;

    pose.header = path.header;

    pose.pose.position.x = sample.pose.x;
    pose.pose.position.y = sample.pose.y;
    pose.pose.position.z = 0.0;

    const double half_yaw = sample.pose.yaw * 0.5;

    pose.pose.orientation.z = std::sin(half_yaw);
    pose.pose.orientation.w = std::cos(half_yaw);

    path.poses.push_back(std::move(pose));
  }

  RCLCPP_INFO(
    node->get_logger(),
    "RRT* path generated: %zu poses, %zu iterations, "
    "tree=%zu, cost=%.3f, timed_out=%s",
    path.poses.size(),
    result.iterations,
    result.tree_size,
    result.best_cost,
    result.timed_out ? "true" : "false");

  return path;
}

}  // namespace tugger_navigation

PLUGINLIB_EXPORT_CLASS(
  tugger_navigation::RRTStarReedsSheppPlanner,
  nav2_core::GlobalPlanner)
