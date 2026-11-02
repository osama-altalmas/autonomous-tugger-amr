#include "tugger_navigation/reeds_shepp_steering.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ompl/base/spaces/ReedsSheppStateSpace.h>

namespace tugger_navigation
{

namespace
{

namespace ob = ompl::base;

using RS = ob::ReedsSheppStateSpace;

void setPose(ob::State * state, const Pose2D & pose)
{
  auto * s = state->as<ob::SE2StateSpace::StateType>();
  s->setX(pose.x);
  s->setY(pose.y);
  s->setYaw(pose.yaw);
}

Pose2D getPose(const ob::State * state)
{
  const auto * s = state->as<ob::SE2StateSpace::StateType>();
  return {s->getX(), s->getY(), s->getYaw()};
}

MotionSegment::Type segmentType(
  RS::ReedsSheppPathSegmentType type)
{
  switch (type) {
    case RS::RS_LEFT:
      return MotionSegment::Type::LEFT;
    case RS::RS_RIGHT:
      return MotionSegment::Type::RIGHT;
    default:
      return MotionSegment::Type::STRAIGHT;
  }
}

double segmentCurvature(
  MotionSegment::Type type,
  double radius)
{
  if (type == MotionSegment::Type::LEFT) {
    return 1.0 / radius;
  }

  if (type == MotionSegment::Type::RIGHT) {
    return -1.0 / radius;
  }

  return 0.0;
}

}  // namespace

ReedsSheppSteering::ReedsSheppSteering(
  double turning_radius,
  double sample_resolution)
: turning_radius_(turning_radius),
  sample_resolution_(sample_resolution)
{
  if (!std::isfinite(turning_radius_) ||
      !std::isfinite(sample_resolution_) ||
      turning_radius_ <= 0.0 ||
      sample_resolution_ <= 0.0)
  {
    throw std::invalid_argument(
      "Turning radius and sample resolution must be positive and finite");
  }
}

SteeringResult ReedsSheppSteering::connect(
  const Pose2D & start,
  const Pose2D & goal) const
{
  SteeringResult result;

  if (!std::isfinite(start.x) ||
      !std::isfinite(start.y) ||
      !std::isfinite(start.yaw) ||
      !std::isfinite(goal.x) ||
      !std::isfinite(goal.y) ||
      !std::isfinite(goal.yaw))
  {
    return result;
  }

  RS space(turning_radius_);

  auto * from = space.allocState();
  auto * to = space.allocState();
  auto * sampled = space.allocState();

  setPose(from, start);
  setPose(to, goal);

  const auto path = space.getPath(from, to);

  int previous_direction = 0;
  bool first_segment = true;

  for (int i = 0; i < 5; ++i) {
    const double signed_length = path.length_[i];

    if (path.type_[i] == RS::RS_NOP ||
        std::abs(signed_length) < 1e-10) {
      continue;
    }

    MotionSegment segment;
    segment.type = segmentType(path.type_[i]);
    segment.direction = signed_length > 0.0 ? 1 : -1;
    segment.length = std::abs(signed_length) * turning_radius_;
    segment.curvature = segmentCurvature(
      segment.type, turning_radius_);

    if (!first_segment) {
      ++result.steering_transitions;
    }

    if (previous_direction != 0 &&
        previous_direction != segment.direction) {
      ++result.gear_changes;
    }

    previous_direction = segment.direction;
    first_segment = false;

    result.total_length += segment.length;
    result.total_heading_change +=
      std::abs(segment.curvature) * segment.length;

    if (segment.direction < 0) {
      result.reverse_length += segment.length;
    }

    result.segments.push_back(segment);
  }

  const int intervals = std::max(
    1,
    static_cast<int>(
      std::ceil(result.total_length / sample_resolution_)));

  // Segment boundaries are sampled explicitly so that
  // direction changes and steering transitions are preserved.
  double accumulated = 0.0;
  int last_direction = 0;

  space.interpolate(from, to, 0.0, sampled);

  result.samples.push_back({
    getPose(sampled),
    result.segments.empty() ? 0 : result.segments.front().direction,
    result.segments.empty() ? 0.0 : result.segments.front().curvature,
    0.0,
    false
  });

  for (const auto & segment : result.segments) {
    const double start_distance = accumulated;
    accumulated += segment.length;

    const bool cusp =
      last_direction != 0 &&
      last_direction != segment.direction;

    if (cusp) {
      const double t = start_distance / result.total_length;

      space.interpolate(from, to, t, sampled);

      // Preserve both sides of the gear transition at the
      // same geometric pose. The vehicle must stop here.
      const Pose2D cusp_pose = getPose(sampled);

      result.samples.push_back({
        cusp_pose,
        last_direction,
        result.samples.back().curvature,
        start_distance,
        true
      });

      result.samples.push_back({
        cusp_pose,
        segment.direction,
        segment.curvature,
        start_distance,
        true
      });
    }

    const int segment_intervals = std::max(
      1,
      static_cast<int>(
        std::ceil(segment.length / sample_resolution_)));

    for (int j = 1; j <= segment_intervals; ++j) {
      const double local_fraction =
        static_cast<double>(j) / segment_intervals;

      const double distance =
        start_distance + local_fraction * segment.length;

      const double t =
        result.total_length > 1e-12 ?
        std::min(1.0, distance / result.total_length) : 1.0;

      space.interpolate(from, to, t, sampled);

      result.samples.push_back({
        getPose(sampled),
        segment.direction,
        segment.curvature,
        distance,
        false
      });
    }

    last_direction = segment.direction;
  }

  if (result.segments.empty()) {
    result.samples.clear();
    result.samples.push_back({start, 0, 0.0, 0.0, false});
  }

  result.valid = !result.samples.empty();

  space.freeState(sampled);
  space.freeState(to);
  space.freeState(from);

  return result;
}

SteeringResult ReedsSheppSteering::connectLimited(
  const Pose2D & start,
  const Pose2D & goal,
  double max_length) const
{
  if (!std::isfinite(max_length) || max_length <= 0.0) {
    throw std::invalid_argument(
      "Maximum extension length must be positive and finite");
  }

  const SteeringResult full = connect(start, goal);

  if (!full.valid) {
    return full;
  }

  if (full.total_length <= max_length) {
    return full;
  }

  // The truncated result is a prefix of the original
  // Reeds-Shepp trajectory, not a newly optimized path.
  SteeringResult limited;

  limited.samples.push_back(full.samples.front());

  double remaining = max_length;
  int previous_direction = 0;

  for (const auto & segment : full.segments) {
    if (remaining <= 1e-10) {
      break;
    }

    const double used_length =
      std::min(segment.length, remaining);

    if (used_length <= 1e-10) {
      continue;
    }

    MotionSegment partial = segment;
    partial.length = used_length;

    if (previous_direction != 0) {
      ++limited.steering_transitions;

      if (previous_direction != partial.direction) {
        ++limited.gear_changes;
      }
    }

    limited.segments.push_back(partial);
    limited.total_length += used_length;
    limited.total_heading_change +=
      std::abs(partial.curvature) * used_length;

    if (partial.direction < 0) {
      limited.reverse_length += used_length;
    }

    previous_direction = partial.direction;
    remaining -= used_length;
  }

  // Retain original samples strictly before the limit.
  // Preserve cusp duplicates at identical distances.
  for (std::size_t i = 1; i < full.samples.size(); ++i) {
    const auto & sample = full.samples[i];

    if (sample.distance >= max_length - 1e-10) {
      break;
    }

    limited.samples.push_back(sample);
  }

  // Evaluate the endpoint on the original Reeds-Shepp path.
  // This avoids straight-line interpolation across a turn.
  namespace ob = ompl::base;

  ob::ReedsSheppStateSpace space(turning_radius_);

  auto * from = space.allocState();
  auto * to = space.allocState();
  auto * endpoint = space.allocState();

  setPose(from, start);
  setPose(to, goal);

  const double fraction =
    max_length / full.total_length;

  space.interpolate(from, to, fraction, endpoint);

  const Pose2D endpoint_pose = getPose(endpoint);

  space.freeState(endpoint);
  space.freeState(to);
  space.freeState(from);

  const auto & last_segment = limited.segments.back();

  limited.samples.push_back({
    endpoint_pose,
    last_segment.direction,
    last_segment.curvature,
    limited.total_length,
    false
  });

  limited.valid =
    !limited.segments.empty() &&
    limited.samples.size() >= 2;

  return limited;
}

bool ReedsSheppSteering::collisionFree(
  const SteeringResult & result,
  const std::function<bool(const Pose2D &)> & pose_is_free,
  const std::function<bool()> & should_stop) const
{
  if (!result.valid || !pose_is_free) {
    return false;
  }

  if (should_stop && should_stop()) {
    return false;
  }

  for (const auto & sample : result.samples) {
    if (should_stop && should_stop()) {
      return false;
    }

    if (!pose_is_free(sample.pose)) {
      return false;
    }
  }

  return !(should_stop && should_stop());
}

double ReedsSheppSteering::distance(
  const Pose2D & start,
  const Pose2D & goal) const
{
  RS space(turning_radius_);

  auto * from = space.allocState();
  auto * to = space.allocState();

  setPose(from, start);
  setPose(to, goal);

  const double value = space.distance(from, to);

  space.freeState(to);
  space.freeState(from);

  return value;
}

double ReedsSheppSteering::turningRadius() const
{
  return turning_radius_;
}

double ReedsSheppSteering::sampleResolution() const
{
  return sample_resolution_;
}

}  // namespace tugger_navigation
