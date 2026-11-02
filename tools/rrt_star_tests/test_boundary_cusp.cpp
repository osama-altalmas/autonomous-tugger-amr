#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tugger_navigation/rrt_star_trajectory.hpp"

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

  const Pose2D start{0.0, 0.0, 0.0};
  const Pose2D boundary{1.0, 0.0, 0.0};
  const Pose2D goal{0.5, 0.0, 0.0};

  const auto forward = steering.connect(start, boundary);
  const auto reverse = steering.connect(boundary, goal);

  require(forward.valid, "Forward connection invalid");
  require(reverse.valid, "Reverse connection invalid");

  require(
    forward.segments.front().direction == 1,
    "First connection must travel forward");

  require(
    reverse.segments.front().direction == -1,
    "Second connection must travel backward");

  const auto trajectory =
    assembleTrajectory({forward, reverse});

  bool found_boundary_cusp = false;
  int direction_changes = 0;
  int unmarked_direction_changes = 0;
  int distance_resets = 0;

  for (std::size_t i = 1; i < trajectory.size(); ++i) {
    const auto & previous = trajectory[i - 1];
    const auto & current = trajectory[i];

    if (current.distance + 1e-8 < previous.distance) {
      ++distance_resets;
    }

    if (previous.direction != current.direction) {
      ++direction_changes;

      const bool paired_cusp =
        previous.cusp &&
        current.cusp &&
        std::abs(previous.distance - current.distance) < 1e-8 &&
        std::hypot(
          previous.pose.x - current.pose.x,
          previous.pose.y - current.pose.y) < 1e-8;

      if (!paired_cusp) {
        ++unmarked_direction_changes;
      }

      if (paired_cusp &&
          std::abs(previous.pose.x - 1.0) < 1e-8) {
        found_boundary_cusp = true;
      }
    }
  }

  std::cout << "Trajectory samples: "
            << trajectory.size() << '\n';

  std::cout << "Direction changes: "
            << direction_changes << '\n';

  std::cout << "Unmarked direction changes: "
            << unmarked_direction_changes << '\n';

  std::cout << "Distance resets: "
            << distance_resets << '\n';

  std::cout << "Boundary cusp detected: "
            << (found_boundary_cusp ? "yes" : "no") << '\n';

  std::cout << "Final cumulative distance: "
            << trajectory.back().distance << " m\n";

  require(direction_changes == 1, "Expected exactly one gear change");
  require(found_boundary_cusp, "Boundary cusp not preserved");
  require(unmarked_direction_changes == 0, "Unmarked direction change");
  require(distance_resets == 0, "Cumulative distance reset");

  require(
    std::abs(trajectory.back().distance - 1.5) < 1e-8,
    "Incorrect cumulative distance");

  std::cout << "\nBOUNDARY CUSP: PASS\n";

  return 0;
}
