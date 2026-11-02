#include "tugger_navigation/rrt_star_planner_core.hpp"
#include "tugger_navigation/rrt_star_trajectory.hpp"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tugger_navigation
{

RRTStarPlannerCore::RRTStarPlannerCore(
  const ReedsSheppSteering & steering,
  const RRTStarCost & cost_model,
  RRTStarSampler & sampler,
  const RRTStarExpansion & expansion)
: steering_(steering),
  cost_model_(cost_model),
  sampler_(sampler),
  expansion_(expansion)
{
}

RRTStarPlanResult RRTStarPlannerCore::plan(
  const Pose2D & start,
  const Pose2D & goal,
  std::size_t max_iterations,
  const RRTStarExpansion::PoseIsFree & pose_is_free,
  const RRTStarCost::ObstaclePenalty & obstacle_penalty,
  const CancelChecker & cancel_checker,
  std::chrono::steady_clock::duration time_budget,
  const TreeCallback & tree_callback) const
{
  RRTStarPlanResult result;

  const auto planning_start =
    std::chrono::steady_clock::now();

  const bool deadline_enabled =
    time_budget != std::chrono::steady_clock::duration::max();

  auto deadline_reached = [&]() {
    return deadline_enabled &&
      std::chrono::steady_clock::now() - planning_start >=
      time_budget;
  };

  auto should_stop = [&]() {
    if (cancel_checker && cancel_checker()) {
      result.cancelled = true;
      return true;
    }

    if (deadline_reached()) {
      result.timed_out = true;
      return true;
    }

    return false;
  };

  if (!pose_is_free ||
      !pose_is_free(start) ||
      !pose_is_free(goal)) {
    return result;
  }

  // A stationary goal is already reached. OMPL represents
  // this connection with no motion segments, so handle it
  // before searching for a nonzero-length trajectory.
  const double position_error =
    std::hypot(goal.x - start.x, goal.y - start.y);

  const double heading_error =
    std::abs(std::remainder(goal.yaw - start.yaw, 2.0 * std::acos(-1.0)));

  if (position_error < 1e-6 && heading_error < 1e-6) {
    if (should_stop()) {
      return result;
    }

    result.success = true;
    result.best_cost = 0.0;
    result.tree_size = 1;
    result.path.push_back(start);
    result.trajectory.push_back({start, 0, 0.0, 0.0, false});
    return result;
  }

  RRTStarTree tree(start);

  // Visualization observes the tree without changing its structure.
  auto publish_tree = [&]() {
    if (tree_callback) {
      tree_callback(tree, result.iterations);
    }
  };

  publish_tree();

  // Cache each node-to-goal connection once.
  // Geometry and collision feasibility remain unchanged
  // during this standalone planning request.
  struct GoalCandidate
  {
    int node_index{-1};
    SteeringResult connection;
  };

  std::vector<GoalCandidate> goal_candidates;

  auto cache_goal_connection = [&](int node_index) {
    if (should_stop()) {
      return;
    }

    const auto & node = tree.at(node_index);

    SteeringResult connection =
      steering_.connect(node.pose, goal);

    if (should_stop()) {
      return;
    }

    if (!connection.valid ||
        connection.segments.empty() ||
        !steering_.collisionFree(
          connection, pose_is_free, should_stop)) {
      return;
    }

    if (should_stop()) {
      return;
    }

    goal_candidates.push_back({
      node_index,
      std::move(connection)
    });
  };

  // The root may already have a feasible direct connection.
  cache_goal_connection(0);

  for (std::size_t iteration = 0;
       iteration < max_iterations;
       ++iteration)
  {
    if (should_stop()) {
      break;
    }

    const Pose2D target = sampler_.sample(goal);

    const ExpansionResult expansion_result =
      expansion_.expand(
        tree,
        target,
        pose_is_free,
        obstacle_penalty,
        should_stop);

    ++result.iterations;

    if (result.iterations % 50 == 0) {
      publish_tree();
    }

    if (expansion_result.inserted && !result.cancelled &&
        !result.timed_out) {
      cache_goal_connection(expansion_result.node_index);
    }

    if (expansion_result.interrupted || should_stop()) {
      if (expansion_result.interrupted &&
          !result.cancelled && !result.timed_out) {
        should_stop();
      }
      break;
    }
  }

  // Rewiring can change both the cost-to-come and the
  // arrival direction of existing nodes. Recalculate
  // goal costs using the final tree state.
  int best_goal_parent = -1;
  SteeringResult best_goal_connection;

  double lowest_cost =
    std::numeric_limits<double>::infinity();

  for (const auto & candidate : goal_candidates) {
    const auto & node = tree.at(candidate.node_index);

    const double edge_cost =
      cost_model_.evaluate(
        candidate.connection,
        node.arrival_direction,
        obstacle_penalty).total();

    const double total_cost =
      node.cost_from_start + edge_cost;

    if (std::isfinite(total_cost) &&
        total_cost < lowest_cost) {
      lowest_cost = total_cost;
      best_goal_parent = candidate.node_index;
      best_goal_connection = candidate.connection;
    }
  }

  if (best_goal_parent >= 0) {
    result.success = true;
    result.best_cost = lowest_cost;
  }

  result.tree_size = tree.size();

  // Preserve the final tree for inspection, including failed searches.
  publish_tree();

  if (!result.success) {
    std::fprintf(
      stderr,
      "[RRT* DIAGNOSTIC] iterations=%zu tree_size=%zu "
      "goal_candidates=%zu timed_out=%d cancelled=%d\n",
      result.iterations,
      result.tree_size,
      goal_candidates.size(),
      result.timed_out,
      result.cancelled);
    return result;
  }

  const auto indices =
    tree.pathToRoot(best_goal_parent);

  for (const int index : indices) {
    result.path.push_back(tree.at(index).pose);
  }

  result.path.push_back(goal);

  std::vector<SteeringResult> connections;

  for (std::size_t i = 1; i < indices.size(); ++i) {
    connections.push_back(
      tree.at(indices[i]).incoming_connection);
  }

  connections.push_back(best_goal_connection);

  result.trajectory = assembleTrajectory(connections);

  return result;
}

}  // namespace tugger_navigation
