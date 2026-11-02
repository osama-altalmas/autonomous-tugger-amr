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
    SamplingBounds{-5.0, 5.0, -5.0, 5.0},
    0.2, 42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0, 2.0);

  RRTStarPlannerCore planner(
    steering, cost_model, sampler, expansion);

  const auto free_space = [](const Pose2D &) {
    return true;
  };

  const auto result = planner.plan(
    {0.0, 0.0, 0.0},
    {3.0, 0.0, 0.0},
    100,
    free_space);

  require(result.success, "Planner failed in free space");
  require(!result.cancelled, "Unexpected cancellation");
  require(result.iterations == 100, "Iteration count incorrect");
  require(result.tree_size > 1, "Tree did not expand");
  require(!result.path.empty(), "No path extracted");
  require(!result.trajectory.empty(), "No trajectory extracted");

  require(
    std::abs(result.path.front().x) < 1e-8,
    "Incorrect path start");

  require(
    std::abs(result.path.back().x - 3.0) < 1e-8,
    "Incorrect path goal");

  std::cout << "Free-space planning: PASS\n";
  std::cout << "Iterations: " << result.iterations << '\n';
  std::cout << "Tree nodes: " << result.tree_size << '\n';
  std::cout << "Path waypoints: " << result.path.size() << '\n';
  std::cout << "Trajectory samples: "
            << result.trajectory.size() << '\n';
  std::cout << "Best cost: " << result.best_cost << '\n';

  const auto cancelled = planner.plan(
    {0.0, 0.0, 0.0},
    {3.0, 0.0, 0.0},
    100,
    free_space,
    {},
    []() {return true;});

  require(cancelled.cancelled, "Cancellation not detected");
  require(cancelled.iterations == 0, "Cancelled plan executed iterations");

  std::cout << "Cancellation handling: PASS\n";

  std::cout << "\nRRT* PLANNING LOOP: PASS\n";

  return 0;
}
