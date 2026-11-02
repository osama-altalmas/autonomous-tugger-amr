#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/reeds_shepp_steering.hpp"

using namespace tugger_navigation;

constexpr double PI = 3.14159265358979323846;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  const auto result = steering.connect(
    {0.0, 0.0, 0.0},
    {0.0, 0.0, PI});

  require(result.valid, "Invalid steering result");
  require(result.gear_changes == 2, "Expected two gear changes");

  double length_sum = 0.0;

  for (const auto & segment : result.segments) {
    length_sum += segment.length;

    require(
      std::abs(segment.curvature) <= 1.0 / 0.29 + 1e-9,
      "Curvature exceeds limit");
  }

  require(
    std::abs(length_sum - result.total_length) < 1e-8,
    "Segment lengths do not match total length");

  int verified_cusps = 0;

  for (std::size_t i = 1; i < result.samples.size(); ++i) {
    const auto & previous = result.samples[i - 1];
    const auto & current = result.samples[i];

    if (previous.cusp && current.cusp &&
        previous.direction != current.direction)
    {
      require(
        std::hypot(
          previous.pose.x - current.pose.x,
          previous.pose.y - current.pose.y) < 1e-8,
        "Cusp positions differ");

      require(
        std::abs(previous.distance - current.distance) < 1e-8,
        "Cusp distances differ");

      ++verified_cusps;
    }
  }

  require(
    verified_cusps == result.gear_changes,
    "Not all gear changes are represented");

  require(
    steering.collisionFree(
      result,
      [](const Pose2D &) {return true;}),
    "Free path rejected");

  require(
    !steering.collisionFree(
      result,
      [](const Pose2D &) {return false;}),
    "Blocked path accepted");

  std::cout << "Verified gear-change cusps: "
            << verified_cusps << '\n';

  std::cout << "Curvature limit: PASS\n";
  std::cout << "Segment lengths: PASS\n";
  std::cout << "Collision callback: PASS\n";
  std::cout << "Cusp representation: PASS\n";
  std::cout << "\nALL STEERING VALIDATION TESTS PASSED\n";

  return 0;
}
