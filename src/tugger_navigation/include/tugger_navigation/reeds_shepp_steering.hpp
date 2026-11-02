#ifndef TUGGER_NAVIGATION__REEDS_SHEPP_STEERING_HPP_
#define TUGGER_NAVIGATION__REEDS_SHEPP_STEERING_HPP_

#include <functional>
#include <vector>

namespace tugger_navigation
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct MotionSample
{
  Pose2D pose;
  int direction{0};
  double curvature{0.0};
  double distance{0.0};
  bool cusp{false};
};

struct MotionSegment
{
  enum class Type { LEFT, STRAIGHT, RIGHT };

  Type type{Type::STRAIGHT};
  int direction{1};
  double length{0.0};
  double curvature{0.0};
};

struct SteeringResult
{
  std::vector<MotionSegment> segments;
  std::vector<MotionSample> samples;

  double total_length{0.0};
  double reverse_length{0.0};
  double total_heading_change{0.0};

  int gear_changes{0};
  int steering_transitions{0};

  bool valid{false};
};

class ReedsSheppSteering
{
public:
  explicit ReedsSheppSteering(
    double turning_radius = 0.29,
    double sample_resolution = 0.05);

  SteeringResult connect(
    const Pose2D & start,
    const Pose2D & goal) const;

  SteeringResult connectLimited(
    const Pose2D & start,
    const Pose2D & goal,
    double max_length) const;

  bool collisionFree(
    const SteeringResult & result,
    const std::function<bool(const Pose2D &)> & pose_is_free,
    const std::function<bool()> & should_stop = {}) const;

  double distance(
    const Pose2D & start,
    const Pose2D & goal) const;

  double turningRadius() const;
  double sampleResolution() const;

private:
  double turning_radius_;
  double sample_resolution_;
};

}  // namespace tugger_navigation

#endif
