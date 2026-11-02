#include <cassert>
#include <cmath>
#include <iostream>

#include "tugger_navigation/reeds_shepp_steering.hpp"

using tugger_navigation::Pose2D;
using tugger_navigation::ReedsSheppSteering;

constexpr double PI = 3.14159265358979323846;

void runTest(
  const char * name,
  const ReedsSheppSteering & steering,
  const Pose2D & start,
  const Pose2D & goal)
{
  const auto result = steering.connect(start, goal);

  assert(result.valid);
  assert(!result.samples.empty());

  const auto & final_pose = result.samples.back().pose;

  const double position_error = std::hypot(
    final_pose.x - goal.x,
    final_pose.y - goal.y);

  const double heading_error = std::abs(std::atan2(
    std::sin(final_pose.yaw - goal.yaw),
    std::cos(final_pose.yaw - goal.yaw)));

  assert(position_error < 1e-5);
  assert(heading_error < 1e-5);

  for (std::size_t i = 1; i < result.samples.size(); ++i) {
    const double step =
      result.samples[i].distance -
      result.samples[i - 1].distance;

    assert(step >= -1e-9);
    assert(step <= steering.sampleResolution() + 1e-8);
  }

  assert(steering.collisionFree(
    result,
    [](const Pose2D &) {return true;}));

  std::cout << name
            << " | length=" << result.total_length
            << " | reverse=" << result.reverse_length
            << " | gear_changes=" << result.gear_changes
            << " | samples=" << result.samples.size()
            << " | PASS\n";
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  runTest(
    "Straight forward",
    steering,
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0});

  runTest(
    "Straight reverse",
    steering,
    {0.0, 0.0, 0.0},
    {-2.0, 0.0, 0.0});

  runTest(
    "90-degree turn",
    steering,
    {0.0, 0.0, 0.0},
    {2.0, 1.0, PI / 2.0});

  runTest(
    "Close goal with opposite heading",
    steering,
    {0.0, 0.0, 0.0},
    {0.0, 0.0, PI});

  std::cout << "\nALL STEERING MODULE TESTS PASSED\n";
  return 0;
}
