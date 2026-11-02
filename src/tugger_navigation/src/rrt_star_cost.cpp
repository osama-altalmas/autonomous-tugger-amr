#include "tugger_navigation/rrt_star_cost.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tugger_navigation
{

namespace
{

constexpr double PI = 3.14159265358979323846;

void validateWeight(double value)
{
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(
      "RRT* cost weights must be finite and nonnegative");
  }
}

}  // namespace

double RRTStarCostBreakdown::total() const
{
  return distance + reverse + gear_change +
         steering + cardinal + obstacle;
}

RRTStarCost::RRTStarCost(
  const RRTStarCostWeights & weights)
: weights_(weights)
{
  validateWeight(weights_.distance);
  validateWeight(weights_.reverse);
  validateWeight(weights_.gear_change);
  validateWeight(weights_.steering);
  validateWeight(weights_.cardinal);
  validateWeight(weights_.obstacle);
}

double RRTStarCost::cardinalDeviation(
  double travel_heading)
{
  // Zero at cardinal directions; one at diagonal directions.
  // Periodic every 90 degrees.
  const double value = std::sin(2.0 * travel_heading);
  return value * value;
}

RRTStarCostBreakdown RRTStarCost::evaluate(
  const SteeringResult & connection,
  int parent_arrival_direction,
  const ObstaclePenalty & obstacle_penalty) const
{
  if (!connection.valid || connection.segments.empty()) {
    throw std::invalid_argument(
      "Cannot evaluate an invalid or empty connection");
  }

  if (parent_arrival_direction < -1 ||
      parent_arrival_direction > 1) {
    throw std::invalid_argument(
      "Invalid parent arrival direction");
  }

  RRTStarCostBreakdown result;

  result.distance =
    weights_.distance * connection.total_length;

  result.reverse =
    weights_.reverse * connection.reverse_length;

  int gear_changes = connection.gear_changes;

  if (parent_arrival_direction != 0 &&
      parent_arrival_direction !=
      connection.segments.front().direction) {
    ++gear_changes;
  }

  result.gear_change =
    weights_.gear_change * gear_changes;

  result.steering =
    weights_.steering * connection.total_heading_change;

  // Integrate heading preference along the sampled trajectory.
  // Each sample represents the motion interval ending there.
  for (std::size_t i = 1; i < connection.samples.size(); ++i) {
    const auto & previous = connection.samples[i - 1];
    const auto & current = connection.samples[i];

    const double ds =
      current.distance - previous.distance;

    if (ds <= 0.0) {
      continue;
    }

    const double travel_heading =
      current.pose.yaw +
      (current.direction < 0 ? PI : 0.0);

    result.cardinal +=
      weights_.cardinal *
      cardinalDeviation(travel_heading) * ds;

    if (weights_.obstacle > 0.0 && obstacle_penalty) {
      const double penalty = obstacle_penalty(current.pose);

      if (!std::isfinite(penalty) || penalty < 0.0) {
        throw std::runtime_error(
          "Obstacle penalty must be finite and nonnegative");
      }

      result.obstacle +=
        weights_.obstacle * penalty * ds;
    }
  }

  return result;
}

}  // namespace tugger_navigation
