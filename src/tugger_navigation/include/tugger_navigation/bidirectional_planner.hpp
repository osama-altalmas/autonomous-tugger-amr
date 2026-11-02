#ifndef TUGGER_NAVIGATION__BIDIRECTIONAL_PLANNER_HPP_
#define TUGGER_NAVIGATION__BIDIRECTIONAL_PLANNER_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "nav2_core/global_planner.hpp"
#include "nav2_smac_planner/smac_planner_hybrid.hpp"

namespace tugger_navigation
{

class BidirectionalPlanner : public nav2_core::GlobalPlanner
{
public:
  BidirectionalPlanner() = default;
  ~BidirectionalPlanner() override = default;

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
  enum class Direction
  {
    NONE,
    FORWARD,
    REVERSE
  };

  geometry_msgs::msg::PoseStamped rotateHeadingPi(
    const geometry_msgs::msg::PoseStamped & pose) const;

  nav_msgs::msg::Path restoreReverseHeadings(
    const nav_msgs::msg::Path & path) const;

  double pathLength(const nav_msgs::msg::Path & path) const;
  double pathScore(const nav_msgs::msg::Path & path) const;

  bool sameGoal(
    const geometry_msgs::msg::PoseStamped & a,
    const geometry_msgs::msg::PoseStamped & b) const;

  nav_msgs::msg::Path createForwardPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    const std::vector<geometry_msgs::msg::PoseStamped> & viapoints,
    std::function<bool()> cancel_checker);

  nav_msgs::msg::Path createReversePlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    const std::vector<geometry_msgs::msg::PoseStamped> & viapoints,
    std::function<bool()> cancel_checker);

  nav2::LifecycleNode::WeakPtr parent_;
  std::string name_;

  std::unique_ptr<nav2_smac_planner::SmacPlannerHybrid> forward_planner_;
  std::unique_ptr<nav2_smac_planner::SmacPlannerHybrid> reverse_planner_;

  Direction locked_direction_{Direction::NONE};

  geometry_msgs::msg::PoseStamped locked_goal_;
  bool have_locked_goal_{false};

};

}  // namespace tugger_navigation

#endif
