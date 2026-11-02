#ifndef TUGGER_NAVIGATION__RRT_STAR_TRAJECTORY_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_TRAJECTORY_HPP_

#include <vector>

#include "tugger_navigation/reeds_shepp_steering.hpp"

namespace tugger_navigation
{

std::vector<MotionSample> assembleTrajectory(
  const std::vector<SteeringResult> & connections);

}  // namespace tugger_navigation

#endif
