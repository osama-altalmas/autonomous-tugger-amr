#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <ompl/base/spaces/ReedsSheppStateSpace.h>

namespace ob = ompl::base;

constexpr double PI = 3.14159265358979323846;
constexpr double TURNING_RADIUS = 0.29;
constexpr double SAMPLE_RESOLUTION = 0.05;

struct Pose2D
{
  double x;
  double y;
  double yaw;
};

double angleDifference(double a, double b)
{
  return std::atan2(std::sin(a - b), std::cos(a - b));
}

const char * segmentName(
  ob::ReedsSheppStateSpace::ReedsSheppPathSegmentType type)
{
  switch (type) {
    case ob::ReedsSheppStateSpace::RS_LEFT:
      return "LEFT";
    case ob::ReedsSheppStateSpace::RS_RIGHT:
      return "RIGHT";
    case ob::ReedsSheppStateSpace::RS_STRAIGHT:
      return "STRAIGHT";
    default:
      return "NONE";
  }
}

int main()
{
  ob::ReedsSheppStateSpace space(TURNING_RADIUS);

  auto * start = space.allocState();
  auto * goal = space.allocState();
  auto * sample = space.allocState();

  const Pose2D start_pose{0.0, 0.0, 0.0};
  const Pose2D goal_pose{2.0, 1.0, PI / 2.0};

  auto setPose = [](ob::State * state, const Pose2D & pose) {
    auto * s = state->as<ob::SE2StateSpace::StateType>();
    s->setX(pose.x);
    s->setY(pose.y);
    s->setYaw(pose.yaw);
  };

  setPose(start, start_pose);
  setPose(goal, goal_pose);

  const auto path = space.getPath(start, goal);
  const double length = space.distance(start, goal);

  std::cout << std::fixed << std::setprecision(5);

  std::cout << "========== REEDS-SHEPP TEST ==========\n";
  std::cout << "Turning radius: " << TURNING_RADIUS << " m\n";
  std::cout << "Maximum curvature: "
            << 1.0 / TURNING_RADIUS << " 1/m\n";
  std::cout << "Path length: " << length << " m\n\n";

  std::cout << "Segments:\n";

  int forward_segments = 0;
  int reverse_segments = 0;
  int gear_changes = 0;
  int previous_direction = 0;
  double segment_length_sum = 0.0;

  for (int i = 0; i < 5; ++i) {
    const double signed_length = path.length_[i];

    if (path.type_[i] == ob::ReedsSheppStateSpace::RS_NOP ||
        std::abs(signed_length) < 1e-10) {
      continue;
    }

    const int direction = signed_length > 0.0 ? 1 : -1;

    if (direction > 0) {
      ++forward_segments;
    } else {
      ++reverse_segments;
    }

    if (previous_direction != 0 &&
        direction != previous_direction) {
      ++gear_changes;
    }

    previous_direction = direction;

    // OMPL stores normalized segment lengths.
    const double meters = signed_length * TURNING_RADIUS;
    segment_length_sum += std::abs(meters);

    std::cout << "  " << segmentName(path.type_[i])
              << " | " << (direction > 0 ? "FORWARD" : "REVERSE")
              << " | " << meters << " m\n";
  }

  std::cout << "\nForward segments: " << forward_segments << '\n';
  std::cout << "Reverse segments: " << reverse_segments << '\n';
  std::cout << "Gear changes: " << gear_changes << '\n';
  std::cout << "Segment length sum: " << segment_length_sum << " m\n";

  const int intervals = std::max(
    1, static_cast<int>(std::ceil(length / SAMPLE_RESOLUTION)));

  std::cout << "\n========== INTERPOLATED POSES ==========\n";
  std::cout << "Samples: " << intervals + 1 << '\n';

  Pose2D final_pose{};

  for (int i = 0; i <= intervals; ++i) {
    const double t = static_cast<double>(i) / intervals;

    space.interpolate(start, goal, t, sample);

    const auto * s =
      sample->as<ob::SE2StateSpace::StateType>();

    final_pose = {s->getX(), s->getY(), s->getYaw()};

    if (i == 0 || i == intervals || i % 5 == 0) {
      std::cout << "t=" << t
                << " x=" << final_pose.x
                << " y=" << final_pose.y
                << " yaw=" << final_pose.yaw << '\n';
    }
  }

  const double position_error = std::hypot(
    final_pose.x - goal_pose.x,
    final_pose.y - goal_pose.y);

  const double heading_error = std::abs(
    angleDifference(final_pose.yaw, goal_pose.yaw));

  const bool passed =
    std::abs(segment_length_sum - length) < 1e-5 &&
    position_error < 1e-5 &&
    heading_error < 1e-5;

  std::cout << "\n========== VALIDATION ==========\n";
  std::cout << "Position error: " << position_error << " m\n";
  std::cout << "Heading error: " << heading_error << " rad\n";
  std::cout << "Result: " << (passed ? "PASS" : "FAIL") << '\n';

  space.freeState(sample);
  space.freeState(goal);
  space.freeState(start);

  return passed ? 0 : 1;
}
