#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/rrt_star_planner_core.hpp"

using namespace tugger_navigation;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);
  RRTStarCost cost_model;

  RRTStarSampler sampler(
    SamplingBounds{-1.0, 4.0, -2.0, 2.0},
    0.15, 42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0, 2.0);

  RRTStarPlannerCore planner(
    steering, cost_model, sampler, expansion);

  // Rectangular obstacle, including a small safety margin.
  const auto free_space = [](const Pose2D & pose) {
    return !(
      pose.x >= 1.0 && pose.x <= 2.0 &&
      pose.y >= -0.5 && pose.y <= 0.5);
  };

  const Pose2D start{0.0, 0.0, 0.0};
  const Pose2D goal{3.0, 0.0, 0.0};

  const auto result = planner.plan(
    start, goal, 300, free_space);

  std::cout << "Success: "
            << (result.success ? "yes" : "no") << '\n';

  std::cout << "Iterations: "
            << result.iterations << '\n';

  std::cout << "Tree nodes: "
            << result.tree_size << '\n';

  std::cout << "Path waypoints: "
            << result.path.size() << '\n';

  std::cout << "Trajectory samples: "
            << result.trajectory.size() << '\n';

  if (!result.success) {
    std::cout << "No solution found in this run.\n";
    return 1;
  }

  int distance_resets = 0;
  int discontinuities = 0;
  int collision_samples = 0;

  double maximum_spacing = 0.0;

  for (std::size_t i = 0;
       i < result.trajectory.size();
       ++i)
  {
    const auto & current = result.trajectory[i];

    if (!free_space(current.pose)) {
      ++collision_samples;
    }

    if (i == 0) {
      continue;
    }

    const auto & previous =
      result.trajectory[i - 1];

    const double spacing = std::hypot(
      current.pose.x - previous.pose.x,
      current.pose.y - previous.pose.y);

    maximum_spacing =
      std::max(maximum_spacing, spacing);

    if (current.distance + 1e-8 <
        previous.distance) {
      ++distance_resets;
    }

    if (spacing > 0.051) {
      ++discontinuities;
    }
  }

  std::cout << "Maximum spatial spacing: "
            << maximum_spacing << " m\n";

  std::cout << "Distance resets: "
            << distance_resets << '\n';

  std::cout << "Spatial discontinuities: "
            << discontinuities << '\n';

  std::cout << "Collision samples: "
            << collision_samples << '\n';

  std::cout << "Final recorded distance: "
            << result.trajectory.back().distance
            << " m\n";

  std::cout << "Best path cost: "
            << result.best_cost << '\n';

  require(
    result.path.size() > 2,
    "Expected a multi-edge route");

  require(
    collision_samples == 0,
    "Extracted trajectory intersects obstacle");

  require(
    discontinuities == 0,
    "Extracted trajectory has spatial gaps");

  require(
    distance_resets == 0,
    "Extracted trajectory has cumulative distance resets");

  require(
    std::abs(
      result.trajectory.back().distance -
      result.trajectory.front().distance) > 1.0,
    "Extracted trajectory distance is unexpectedly short");

  std::cout << "\nOBSTACLE TRAJECTORY DIAGNOSTIC: PASS\n";

  return 0;
}
