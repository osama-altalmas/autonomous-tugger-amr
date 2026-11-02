#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/reeds_shepp_steering.hpp"

using namespace tugger_navigation;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double a, double b, double tolerance = 1e-6)
{
  return std::abs(a - b) < tolerance;
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  const Pose2D start{0.0, 0.0, 0.0};

  // Test 1: Forward straight-line truncation.
  const auto forward = steering.connectLimited(
    start, {5.0, 0.0, 0.0}, 1.0);

  require(forward.valid, "Forward extension invalid");

  require(
    near(forward.total_length, 1.0),
    "Forward extension exceeds limit");

  require(
    near(forward.samples.back().pose.x, 1.0),
    "Incorrect forward endpoint");

  require(
    near(forward.samples.back().pose.y, 0.0),
    "Incorrect forward lateral position");

  std::cout << "Forward bounded extension: PASS\n";

  // Test 2: Reverse straight-line truncation.
  const auto reverse = steering.connectLimited(
    start, {-5.0, 0.0, 0.0}, 1.0);

  require(reverse.valid, "Reverse extension invalid");

  require(
    near(reverse.total_length, 1.0),
    "Reverse extension exceeds limit");

  require(
    near(reverse.reverse_length, 1.0),
    "Incorrect reverse distance");

  require(
    near(reverse.samples.back().pose.x, -1.0),
    "Incorrect reverse endpoint");

  std::cout << "Reverse bounded extension: PASS\n";

  // Test 3: A nearby target must remain unchanged.
  const auto full = steering.connect(
    start, {0.5, 0.0, 0.0});

  const auto unchanged = steering.connectLimited(
    start, {0.5, 0.0, 0.0}, 1.0);

  require(
    near(full.total_length, unchanged.total_length),
    "Short connection was incorrectly truncated");

  std::cout << "Short connection preservation: PASS\n";

  // Test 4: Truncated turning motion.
  const Pose2D turning_goal{2.0, 2.0, 1.57};

  const auto turning = steering.connectLimited(
    start, turning_goal, 0.5);

  require(turning.valid, "Turning extension invalid");

  require(
    near(turning.total_length, 0.5),
    "Turning extension has incorrect length");

  require(
    !turning.segments.empty(),
    "Turning extension has no segments");

  require(
    steering.collisionFree(
      turning,
      [](const Pose2D &) {return true;}),
    "Turning extension failed free-space collision check");

  std::cout << "Curvature-constrained extension: PASS\n";

  // Test 5: Every sample must remain within the limit.
  double previous_distance = -1.0;

  for (const auto & sample : turning.samples) {
    require(
      sample.distance <= 0.5 + 1e-8,
      "Sample exceeds extension limit");

    require(
      sample.distance >= previous_distance - 1e-8,
      "Sample distances are not monotonic");

    previous_distance = sample.distance;
  }

  std::cout << "Sample distance validation: PASS\n";

  std::cout << "\nALL LIMITED STEERING TESTS PASSED\n";

  return 0;
}
