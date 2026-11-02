#include "tugger_navigation/rrt_star_trajectory.hpp"

#include <cmath>

namespace tugger_navigation
{

std::vector<MotionSample> assembleTrajectory(
  const std::vector<SteeringResult> & connections)
{
  std::vector<MotionSample> trajectory;
  double cumulative_distance = 0.0;

  for (const auto & connection : connections) {
    if (connection.samples.empty()) {
      continue;
    }

    const double offset = cumulative_distance;

    for (std::size_t i = 0; i < connection.samples.size(); ++i) {
      MotionSample sample = connection.samples[i];
      sample.distance += offset;

      if (!trajectory.empty() && i == 0) {
        const auto & previous = trajectory.back();

        const double separation = std::hypot(
          sample.pose.x - previous.pose.x,
          sample.pose.y - previous.pose.y);

        if (separation < 1e-8) {
          if (previous.direction == sample.direction) {
            continue;
          }

          trajectory.back().cusp = true;
          sample.cusp = true;
        }
      }

      trajectory.push_back(sample);
    }

    cumulative_distance += connection.total_length;
  }

  return trajectory;
}

}  // namespace tugger_navigation
