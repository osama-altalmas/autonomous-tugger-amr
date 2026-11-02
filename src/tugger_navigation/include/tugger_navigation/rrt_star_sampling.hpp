#ifndef TUGGER_NAVIGATION__RRT_STAR_SAMPLING_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_SAMPLING_HPP_

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "tugger_navigation/rrt_star_tree.hpp"

namespace tugger_navigation
{

struct SamplingBounds
{
  double min_x{-20.0};
  double max_x{20.0};
  double min_y{-12.5};
  double max_y{12.5};
};

class RRTStarSampler
{
public:
  RRTStarSampler(
    const SamplingBounds & bounds,
    double goal_bias = 0.10,
    std::uint32_t seed = 42);

  Pose2D sample(const Pose2D & goal);

  int nearest(
    const RRTStarTree & tree,
    const Pose2D & target,
    const ReedsSheppSteering & steering,
    const std::function<bool()> & should_stop = {}) const;

  std::vector<int> near(
    const RRTStarTree & tree,
    const Pose2D & target,
    const ReedsSheppSteering & steering,
    double radius,
    const std::function<bool()> & should_stop = {}) const;

private:
  SamplingBounds bounds_;
  double goal_bias_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> probability_;
  std::uniform_real_distribution<double> x_distribution_;
  std::uniform_real_distribution<double> y_distribution_;
  std::uniform_real_distribution<double> yaw_distribution_;
};

}  // namespace tugger_navigation

#endif
