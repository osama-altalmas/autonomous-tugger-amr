#ifndef TUGGER_NAVIGATION__RRT_STAR_EXPANSION_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_EXPANSION_HPP_

#include <functional>

#include "tugger_navigation/rrt_star_sampling.hpp"

namespace tugger_navigation
{

struct ExpansionResult
{
  bool inserted{false};
  int node_index{-1};
  int parent_index{-1};
  double edge_cost{0.0};
  int rewired_nodes{0};
  bool interrupted{false};
};

class RRTStarExpansion
{
public:
  using PoseIsFree = std::function<bool(const Pose2D &)>;

  RRTStarExpansion(
    const ReedsSheppSteering & steering,
    const RRTStarCost & cost_model,
    const RRTStarSampler & sampler,
    double max_extension_length = 1.0,
    double neighbor_radius = 2.0);

  ExpansionResult expand(
    RRTStarTree & tree,
    const Pose2D & target,
    const PoseIsFree & pose_is_free,
    const RRTStarCost::ObstaclePenalty & obstacle_penalty = {},
    const std::function<bool()> & should_stop = {}) const;

private:
  const ReedsSheppSteering & steering_;
  const RRTStarCost & cost_model_;
  const RRTStarSampler & sampler_;
  double max_extension_length_;
  double neighbor_radius_;
};

}  // namespace tugger_navigation

#endif
