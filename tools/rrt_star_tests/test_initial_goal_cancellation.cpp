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
    SamplingBounds{-1.0, 6.0, -2.0, 2.0},
    0.1,
    42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0, 2.0);

  RRTStarPlannerCore planner(
    steering, cost_model, sampler, expansion);

  const Pose2D start{0.0, 0.0, 0.0};
  const Pose2D goal{5.0, 0.0, 0.0};

  int collision_checks = 0;

  const auto free_pose = [&](const Pose2D &) {
    ++collision_checks;
    return true;
  };

  // Allow the two endpoint checks and the first ten
  // collision samples, then request cancellation.
  const auto cancel_checker = [&]() {
    return collision_checks >= 12;
  };

  const auto result = planner.plan(
    start,
    goal,
    1000,
    free_pose,
    {},
    cancel_checker);

  require(result.cancelled, "Cancellation was not detected");
  require(!result.timed_out, "Cancellation incorrectly marked timeout");
  require(!result.success, "Cancelled initial path was accepted");
  require(result.iterations == 0, "Planning iterations started after cancellation");
  require(result.tree_size == 1, "Unexpected tree expansion");
  require(result.trajectory.empty(), "Unexpected trajectory after cancellation");

  std::cout << "Cancellation detected: PASS\n";
  std::cout << "No planning iterations: PASS\n";
  std::cout << "No invalid trajectory: PASS\n";
  std::cout << "Collision checks performed: "
            << collision_checks << '\n';
  std::cout << "\nINITIAL GOAL CANCELLATION: PASS\n";

  return 0;
}
