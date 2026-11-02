#ifndef TUGGER_NAVIGATION__RRT_STAR_PLANNER_CORE_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_PLANNER_CORE_HPP_

#include <chrono>
#include <cstddef>
#include <functional>
#include <vector>

#include "tugger_navigation/rrt_star_expansion.hpp"

namespace tugger_navigation
{

struct RRTStarPlanResult
{
  bool success{false};
  bool cancelled{false};
  bool timed_out{false};

  std::size_t iterations{0};
  std::size_t tree_size{0};

  double best_cost{0.0};

  std::vector<Pose2D> path;
  std::vector<MotionSample> trajectory;
};

class RRTStarPlannerCore
{
public:
  using CancelChecker = std::function<bool()>;

  // Read-only access to the current search tree for visualization.
  // The callback must not modify the tree.
  using TreeCallback =
    std::function<void(const RRTStarTree &, std::size_t)>;

  RRTStarPlannerCore(
    const ReedsSheppSteering & steering,
    const RRTStarCost & cost_model,
    RRTStarSampler & sampler,
    const RRTStarExpansion & expansion);

  RRTStarPlanResult plan(
    const Pose2D & start,
    const Pose2D & goal,
    std::size_t max_iterations,
    const RRTStarExpansion::PoseIsFree & pose_is_free,
    const RRTStarCost::ObstaclePenalty & obstacle_penalty = {},
    const CancelChecker & cancel_checker = {},
    std::chrono::steady_clock::duration time_budget =
      std::chrono::steady_clock::duration::max(),
    const TreeCallback & tree_callback = {}) const;

private:
  const ReedsSheppSteering & steering_;
  const RRTStarCost & cost_model_;
  RRTStarSampler & sampler_;
  const RRTStarExpansion & expansion_;
};

}  // namespace tugger_navigation

#endif
