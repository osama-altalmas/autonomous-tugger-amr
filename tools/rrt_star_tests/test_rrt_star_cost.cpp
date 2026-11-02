#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/rrt_star_cost.hpp"

using namespace tugger_navigation;

constexpr double PI = 3.14159265358979323846;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double a, double b, double tolerance = 1e-6)
{
  return std::abs(a - b) < tolerance;
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);
  RRTStarCost cost;

  const auto forward = steering.connect(
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0});

  const auto reverse = steering.connect(
    {0.0, 0.0, 0.0},
    {-2.0, 0.0, 0.0});

  const auto forward_cost = cost.evaluate(forward);
  const auto reverse_cost = cost.evaluate(reverse);

  require(
    near(forward_cost.total(), 2.0),
    "Incorrect straight-forward cost");

  require(
    near(reverse_cost.total(), 3.0),
    "Incorrect straight-reverse cost");

  require(
    reverse_cost.total() > forward_cost.total(),
    "Reverse movement was not penalized");

  std::cout << "Forward/reverse weighting: PASS\n";

  const auto transition_cost = cost.evaluate(reverse, 1);

  require(
    near(
      transition_cost.total() - reverse_cost.total(),
      0.5),
    "Parent-to-edge gear change not penalized");

  std::cout << "Inter-edge gear-change penalty: PASS\n";

  require(
    near(RRTStarCost::cardinalDeviation(0.0), 0.0) &&
    near(RRTStarCost::cardinalDeviation(PI / 2.0), 0.0) &&
    near(RRTStarCost::cardinalDeviation(PI / 4.0), 1.0),
    "Incorrect cardinal heading preference");

  std::cout << "Cardinal heading preference: PASS\n";

  const auto opposite = steering.connect(
    {0.0, 0.0, 0.0},
    {0.0, 0.0, PI});

  const auto opposite_cost = cost.evaluate(opposite);

  require(
    near(opposite_cost.gear_change, 1.0),
    "Internal gear changes not penalized");

  std::cout << "Internal gear-change penalty: PASS\n";

  RRTStarCostWeights obstacle_weights;
  obstacle_weights.obstacle = 2.0;

  RRTStarCost obstacle_cost(obstacle_weights);

  const auto penalized = obstacle_cost.evaluate(
    forward,
    0,
    [](const Pose2D &) {return 3.0;});

  require(
    near(penalized.obstacle, 12.0),
    "Incorrect obstacle integration");

  std::cout << "Obstacle penalty integration: PASS\n";

  std::cout << "\nALL RRT* COST TESTS PASSED\n";

  return 0;
}
