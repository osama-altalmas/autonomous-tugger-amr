#include "tugger_navigation/rrt_star_sampling.hpp"
#include "tugger_navigation/planning_interrupted.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tugger_navigation
{

namespace
{
constexpr double PI = 3.14159265358979323846;
}

RRTStarSampler::RRTStarSampler(
  const SamplingBounds & bounds,
  double goal_bias,
  std::uint32_t seed)
: bounds_(bounds),
  goal_bias_(goal_bias),
  rng_(seed),
  probability_(0.0, 1.0),
  x_distribution_(bounds.min_x, bounds.max_x),
  y_distribution_(bounds.min_y, bounds.max_y),
  yaw_distribution_(-PI, PI)
{
  if (!std::isfinite(bounds.min_x) ||
      !std::isfinite(bounds.max_x) ||
      !std::isfinite(bounds.min_y) ||
      !std::isfinite(bounds.max_y) ||
      bounds.min_x >= bounds.max_x ||
      bounds.min_y >= bounds.max_y ||
      !std::isfinite(goal_bias) ||
      goal_bias < 0.0 || goal_bias > 1.0)
  {
    throw std::invalid_argument("Invalid RRT* sampling parameters");
  }
}

Pose2D RRTStarSampler::sample(const Pose2D & goal)
{
  if (probability_(rng_) < goal_bias_) {
    return goal;
  }

  return {
    x_distribution_(rng_),
    y_distribution_(rng_),
    yaw_distribution_(rng_)
  };
}

int RRTStarSampler::nearest(
  const RRTStarTree & tree,
  const Pose2D & target,
  const ReedsSheppSteering & steering,
  const std::function<bool()> & should_stop) const
{
  if (tree.size() == 0) {
    throw std::invalid_argument("Cannot search an empty RRT* tree");
  }

  int best_index = 0;
  double best_distance = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < tree.size(); ++i) {
    if (should_stop && should_stop()) {
      throw PlanningInterrupted();
    }

    const double distance = steering.distance(
      tree.at(static_cast<int>(i)).pose,
      target);

    if (distance < best_distance) {
      best_distance = distance;
      best_index = static_cast<int>(i);
    }
  }

  return best_index;
}


std::vector<int> RRTStarSampler::near(
  const RRTStarTree & tree,
  const Pose2D & target,
  const ReedsSheppSteering & steering,
  double radius,
  const std::function<bool()> & should_stop) const
{
  if (!std::isfinite(radius) || radius <= 0.0) {
    throw std::invalid_argument(
      "Nearby-node radius must be positive and finite");
  }

  std::vector<int> neighbors;

  for (std::size_t i = 0; i < tree.size(); ++i) {
    if (should_stop && should_stop()) {
      throw PlanningInterrupted();
    }

    const double distance = steering.distance(
      tree.at(static_cast<int>(i)).pose,
      target);

    if (std::isfinite(distance) && distance <= radius) {
      neighbors.push_back(static_cast<int>(i));
    }
  }

  return neighbors;
}

}  // namespace tugger_navigation
