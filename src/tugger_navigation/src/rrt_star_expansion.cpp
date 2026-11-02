#include "tugger_navigation/rrt_star_expansion.hpp"
#include "tugger_navigation/planning_interrupted.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace tugger_navigation
{

RRTStarExpansion::RRTStarExpansion(
  const ReedsSheppSteering & steering,
  const RRTStarCost & cost_model,
  const RRTStarSampler & sampler,
  double max_extension_length,
  double neighbor_radius)
: steering_(steering),
  cost_model_(cost_model),
  sampler_(sampler),
  max_extension_length_(max_extension_length),
  neighbor_radius_(neighbor_radius)
{
  if (!std::isfinite(neighbor_radius_) ||
      neighbor_radius_ <= 0.0) {
    throw std::invalid_argument(
      "Neighbor radius must be positive and finite");
  }

  if (!std::isfinite(max_extension_length_) ||
      max_extension_length_ <= 0.0)
  {
    throw std::invalid_argument(
      "Maximum extension length must be positive and finite");
  }
}

ExpansionResult RRTStarExpansion::expand(
  RRTStarTree & tree,
  const Pose2D & target,
  const PoseIsFree & pose_is_free,
  const RRTStarCost::ObstaclePenalty & obstacle_penalty,
  const std::function<bool()> & should_stop) const
{
  ExpansionResult result;

  static std::size_t attempts = 0;
  static std::size_t steering_failed = 0;
  static std::size_t collision_failed = 0;
  static std::size_t cost_failed = 0;
  static std::size_t accepted = 0;

  ++attempts;

  auto report = [&]() {
    if (attempts % 500 == 0) {
      std::fprintf(stderr,
        "[RRT* EXPANSION STATS] attempts=%zu steering=%zu "
        "collision=%zu cost=%zu accepted=%zu\\n",
        attempts, steering_failed, collision_failed,
        cost_failed, accepted);
    }
  };

  if (!pose_is_free) {
    return result;
  }

  int nearest_index = -1;

  try {
    nearest_index = sampler_.nearest(
      tree, target, steering_, should_stop);
  } catch (const PlanningInterrupted &) {
    result.interrupted = true;
    return result;
  }

  const auto & parent = tree.at(nearest_index);

  const SteeringResult connection = steering_.connectLimited(
    parent.pose,
    target,
    max_extension_length_);

  if (!connection.valid ||
      connection.segments.empty() ||
      connection.total_length <= 1e-9)
  {
    ++steering_failed;
    report();
    return result;
  }

  if (!steering_.collisionFree(
      connection, pose_is_free, should_stop)) {
    ++collision_failed;

    if (collision_failed <= 5) {
      for (const auto & sample : connection.samples) {
        if (!pose_is_free(sample.pose)) {
          std::fprintf(stderr,
            "[RRT* COLLISION] sample=(%.3f, %.3f, %.3f) "
            "target=(%.3f, %.3f, %.3f)\\n",
            sample.pose.x, sample.pose.y, sample.pose.yaw,
            target.x, target.y, target.yaw);
          break;
        }
      }
    }

    result.interrupted =
      should_stop && should_stop();
    report();
    return result;
  }

  const Pose2D new_pose =
    connection.samples.back().pose;

  // The nearest node is always our initial feasible parent.
  int best_parent = nearest_index;
  SteeringResult best_connection = connection;

  double best_edge_cost = cost_model_.evaluate(
    connection,
    parent.arrival_direction,
    obstacle_penalty).total();

  double best_total_cost =
    parent.cost_from_start + best_edge_cost;

  if (!std::isfinite(best_total_cost) ||
      best_edge_cost < 0.0) {
    ++cost_failed;
    report();
    return result;
  }

  // Evaluate alternative parents in the nearby neighborhood.
  std::vector<int> neighbors;

  try {
    neighbors = sampler_.near(
      tree, new_pose, steering_,
      neighbor_radius_, should_stop);
  } catch (const PlanningInterrupted &) {
    result.interrupted = true;
    return result;
  }

  for (const int candidate_index : neighbors) {
    if (should_stop && should_stop()) {
      result.interrupted = true;
      return result;
    }

    if (candidate_index == nearest_index) {
      continue;
    }

    const auto & candidate = tree.at(candidate_index);

    const SteeringResult candidate_connection =
      steering_.connect(candidate.pose, new_pose);

    if (!candidate_connection.valid ||
        candidate_connection.segments.empty() ||
        candidate_connection.total_length <= 1e-9 ||
        candidate_connection.total_length >
          max_extension_length_ + 1e-9) {
      continue;
    }

    if (!steering_.collisionFree(
        candidate_connection, pose_is_free, should_stop)) {
      if (should_stop && should_stop()) {
        result.interrupted = true;
        return result;
      }
      continue;
    }

    const double candidate_edge_cost =
      cost_model_.evaluate(
        candidate_connection,
        candidate.arrival_direction,
        obstacle_penalty).total();

    const double candidate_total_cost =
      candidate.cost_from_start + candidate_edge_cost;

    if (!std::isfinite(candidate_total_cost) ||
        candidate_edge_cost < 0.0) {
      continue;
    }

    if (candidate_total_cost + 1e-9 < best_total_cost) {
      best_parent = candidate_index;
      best_connection = candidate_connection;
      best_edge_cost = candidate_edge_cost;
      best_total_cost = candidate_total_cost;
    }
  }

  if (should_stop && should_stop()) {
    result.interrupted = true;
    return result;
  }

  const int new_index = tree.addNode(
    new_pose,
    best_parent,
    best_connection,
    best_edge_cost);

  ++accepted;
  report();

  result.inserted = true;
  result.node_index = new_index;
  result.parent_index = best_parent;
  result.edge_cost = best_edge_cost;

  // Rewire existing neighbors through the newly inserted node.
  std::vector<int> rewire_neighbors;

  try {
    rewire_neighbors = sampler_.near(
      tree, new_pose, steering_,
      neighbor_radius_, should_stop);
  } catch (const PlanningInterrupted &) {
    result.interrupted = true;
    return result;
  }

  for (const int neighbor_index : rewire_neighbors) {
    if (should_stop && should_stop()) {
      result.interrupted = true;
      return result;
    }

    if (neighbor_index == new_index ||
        neighbor_index == best_parent ||
        neighbor_index == 0) {
      continue;
    }

    // Never create a cycle by connecting an ancestor
    // through one of its descendants.
    if (tree.isAncestor(neighbor_index, new_index)) {
      continue;
    }

    const auto & neighbor = tree.at(neighbor_index);

    const SteeringResult rewire_connection =
      steering_.connect(new_pose, neighbor.pose);

    if (!rewire_connection.valid ||
        rewire_connection.segments.empty() ||
        rewire_connection.total_length <= 1e-9 ||
        rewire_connection.total_length >
          max_extension_length_ + 1e-9) {
      continue;
    }

    if (!steering_.collisionFree(
        rewire_connection, pose_is_free, should_stop)) {
      if (should_stop && should_stop()) {
        result.interrupted = true;
        return result;
      }
      continue;
    }

    if (should_stop && should_stop()) {
      result.interrupted = true;
      return result;
    }

    const bool rewired = tree.rewireWithCostModel(
      neighbor_index,
      new_index,
      rewire_connection,
      cost_model_,
      obstacle_penalty,
      should_stop);

    if (should_stop && should_stop()) {
      result.interrupted = true;
      return result;
    }

    if (rewired) {
      ++result.rewired_nodes;
    }
  }

  return result;
}

}  // namespace tugger_navigation
