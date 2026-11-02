#ifndef TUGGER_NAVIGATION__RRT_STAR_TREE_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_TREE_HPP_

#include <cstddef>
#include <functional>
#include <vector>

#include "tugger_navigation/reeds_shepp_steering.hpp"
#include "tugger_navigation/rrt_star_cost.hpp"

namespace tugger_navigation
{

struct RRTStarNode
{
  Pose2D pose;

  int parent{-1};
  std::vector<int> children;

  double cost_from_start{0.0};
  double incoming_edge_cost{0.0};

  // Motion direction at the end of the incoming connection.
  int arrival_direction{0};

  SteeringResult incoming_connection;
};

class RRTStarTree
{
public:
  explicit RRTStarTree(const Pose2D & start);

  int addNode(
    const Pose2D & pose,
    int parent,
    const SteeringResult & connection,
    double edge_cost);

  bool rewire(
    int node,
    int new_parent,
    const SteeringResult & connection,
    double edge_cost);

  bool rewireWithCostModel(
    int node,
    int new_parent,
    const SteeringResult & connection,
    const RRTStarCost & cost_model,
    const RRTStarCost::ObstaclePenalty & obstacle_penalty = {},
    const std::function<bool()> & should_stop = {});

  bool isAncestor(int ancestor, int node) const;

  std::vector<int> pathToRoot(int node) const;

  const RRTStarNode & at(int index) const;

  std::size_t size() const;

private:
  std::vector<RRTStarNode> nodes_;

  void propagateCosts(int node);
};

}  // namespace tugger_navigation

#endif
