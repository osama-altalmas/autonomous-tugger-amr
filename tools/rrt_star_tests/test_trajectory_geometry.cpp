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

double positionDistance(const Pose2D & a, const Pose2D & b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
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

  const Pose2D start{0.0, 0.0, 0.0};
  const Pose2D goal{3.0, 0.0, 0.0};

  const auto result = planner.plan(
    start, goal, 100, free_space);

  require(result.success, "Planning failed");

  const auto & trajectory = result.trajectory;

  require(!trajectory.empty(), "Trajectory is empty");

  require(
    positionDistance(trajectory.front().pose, start) < 1e-6,
    "Trajectory does not start at robot pose");

  require(
    positionDistance(trajectory.back().pose, goal) < 1e-6,
    "Trajectory does not end at goal pose");

  double maximum_spacing = 0.0;
  int distance_resets = 0;
  int discontinuities = 0;

  for (std::size_t i = 1; i < trajectory.size(); ++i) {
    const auto & previous = trajectory[i - 1];
    const auto & current = trajectory[i];

    const double spacing =
      positionDistance(previous.pose, current.pose);

    maximum_spacing = std::max(maximum_spacing, spacing);

    if (current.distance + 1e-8 < previous.distance) {
      ++distance_resets;
    }

    if (spacing > 0.051) {
      ++discontinuities;
    }
  }

  std::cout << "Trajectory samples: "
            << trajectory.size() << '\n';

  std::cout << "Maximum spatial spacing: "
            << maximum_spacing << " m\n";

  std::cout << "Distance resets: "
            << distance_resets << '\n';

  std::cout << "Spatial discontinuities: "
            << discontinuities << '\n';

  std::cout << "Final recorded distance: "
            << trajectory.back().distance << " m\n";

  require(
    discontinuities == 0,
    "Trajectory contains spatial discontinuities");

  require(
    distance_resets == 0,
    "Trajectory cumulative distance is not monotonic");

  std::cout << "\nTRAJECTORY GEOMETRY: PASS\n";

  return 0;
}
