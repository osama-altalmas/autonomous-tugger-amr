#ifndef TUGGER_NAVIGATION__RRT_STAR_COST_HPP_
#define TUGGER_NAVIGATION__RRT_STAR_COST_HPP_

#include <functional>

#include "tugger_navigation/reeds_shepp_steering.hpp"

namespace tugger_navigation
{

struct RRTStarCostWeights
{
  double distance{1.0};
  double reverse{0.5};
  double gear_change{0.5};
  double steering{0.10};
  double cardinal{0.8};
  double obstacle{0.0};
};

struct RRTStarCostBreakdown
{
  double distance{0.0};
  double reverse{0.0};
  double gear_change{0.0};
  double steering{0.0};
  double cardinal{0.0};
  double obstacle{0.0};

  double total() const;
};

class RRTStarCost
{
public:
  using ObstaclePenalty =
    std::function<double(const Pose2D &)>;

  explicit RRTStarCost(
    const RRTStarCostWeights & weights = {});

  RRTStarCostBreakdown evaluate(
    const SteeringResult & connection,
    int parent_arrival_direction = 0,
    const ObstaclePenalty & obstacle_penalty = {}) const;

  static double cardinalDeviation(double travel_heading);

private:
  RRTStarCostWeights weights_;
};

}  // namespace tugger_navigation

#endif
