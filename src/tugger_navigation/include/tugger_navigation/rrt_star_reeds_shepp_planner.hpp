#ifndef TUGGER_NAVIGATION__RRT_STAR_REEDS_SHEPP_PLANNER_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_REEDS_SHEPP_PLANNER_HPP_

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "tugger_navigation/rrt_star_planner_core.hpp"

namespace tugger_navigation
{

class RRTStarReedsSheppPlanner : public nav2_core::GlobalPlanner
{
public:
  RRTStarReedsSheppPlanner() = default;
  ~RRTStarReedsSheppPlanner() override = default;

  void configure(
    const nav2::LifecycleNode::WeakPtr & parent,
    std::string name,
    nav2::TransformBuffer::SharedPtr tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    const std::vector<geometry_msgs::msg::PoseStamped> & viapoints,
    std::function<bool()> cancel_checker) override;

private:
  nav2::LifecycleNode::WeakPtr parent_;
  std::string name_;

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;

  // RViz visualization of the RRT* search tree.
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    tree_marker_pub_;

  double turning_radius_{0.29};
  double sample_resolution_{0.05};
  double max_extension_length_{1.0};
  double neighbor_radius_{2.0};
  double goal_bias_{0.10};
  double planning_time_seconds_{20.0};

  int max_iterations_{3000};
};

}  // namespace tugger_navigation

#endif
