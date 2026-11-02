#include <chrono>
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
    0.15,
    42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0, 2.0);

  RRTStarPlannerCore planner(
    steering, cost_model, sampler, expansion);

  const Pose2D start{0.0, 0.0, 0.0};
  const Pose2D goal{3.0, 0.0, 0.0};

  // An infinite vertical barrier divides the plane.
  // No continuous collision-free route can cross it.
  const auto free_pose = [](const Pose2D & pose) {
    return pose.x < 1.0 || pose.x > 2.0;
  };

  const auto result = planner.plan(
    start,
    goal,
    100000,
    free_pose,
    {},
    {},
    std::chrono::milliseconds(50));

  std::cout << "Success: "
            << (result.success ? "yes" : "no") << '\n';

  std::cout << "Timed out: "
            << (result.timed_out ? "yes" : "no") << '\n';

  std::cout << "Cancelled: "
            << (result.cancelled ? "yes" : "no") << '\n';

  std::cout << "Iterations: "
            << result.iterations << '\n';

  std::cout << "Tree nodes: "
            << result.tree_size << '\n';

  std::cout << "Trajectory samples: "
            << result.trajectory.size() << '\n';

  require(result.timed_out, "Expected planning timeout");
  require(!result.success, "Unreachable goal incorrectly accepted");
  require(!result.cancelled, "Timeout incorrectly marked cancellation");
  require(result.path.empty(), "Unexpected path for unreachable goal");
  require(result.trajectory.empty(), "Unexpected trajectory for unreachable goal");

  std::cout << "\nUNREACHABLE DEADLINE: PASS\n";

  return 0;
}
