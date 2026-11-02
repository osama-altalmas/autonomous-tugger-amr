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

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  const auto connection = steering.connect(
    {0.0, 0.0, 0.0},
    {3.0, 0.0, 0.0});

  require(connection.valid, "Connection invalid");
  require(connection.samples.size() > 20, "Insufficient samples");

  int checked_poses = 0;

  const auto free_pose = [&](const Pose2D &) {
    ++checked_poses;
    return true;
  };

  // First verify normal operation.
  require(
    steering.collisionFree(connection, free_pose),
    "Collision-free trajectory rejected");

  require(
    checked_poses ==
      static_cast<int>(connection.samples.size()),
    "Normal collision check skipped samples");

  // Interrupt before examining the first pose.
  checked_poses = 0;

  const bool immediately_interrupted =
    steering.collisionFree(
      connection,
      free_pose,
      []() {return true;});

  require(
    !immediately_interrupted,
    "Immediate interruption was ignored");

  require(
    checked_poses == 0,
    "Pose evaluated after immediate interruption");

  // Interrupt after exactly ten checked poses.
  checked_poses = 0;

  const bool interrupted_midway =
    steering.collisionFree(
      connection,
      free_pose,
      [&]() {return checked_poses >= 10;});

  require(
    !interrupted_midway,
    "Mid-trajectory interruption was ignored");

  require(
    checked_poses == 10,
    "Unexpected number of poses checked before interruption");

  // Verify that an actual obstacle is still rejected.
  const bool obstacle_accepted =
    steering.collisionFree(
      connection,
      [](const Pose2D & pose) {
        return pose.x < 1.0;
      });

  require(
    !obstacle_accepted,
    "Colliding trajectory incorrectly accepted");

  std::cout << "Trajectory samples: "
            << connection.samples.size() << '\n';

  std::cout << "Immediate interruption: PASS\n";
  std::cout << "Mid-trajectory interruption: PASS\n";
  std::cout << "Obstacle rejection: PASS\n";
  std::cout << "Normal collision validation: PASS\n";

  std::cout << "\nINTERRUPTIBLE COLLISION: PASS\n";

  return 0;
}
