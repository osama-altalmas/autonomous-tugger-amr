#include <iostream>
#include <stdexcept>

#include "tugger_navigation/planning_interrupted.hpp"
#include "tugger_navigation/rrt_star_sampling.hpp"

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
  RRTStarSampler sampler(
    SamplingBounds{-5.0, 5.0, -5.0, 5.0},
    0.1,
    42);

  RRTStarTree tree({0.0, 0.0, 0.0});

  for (int i = 1; i <= 10; ++i) {
    const Pose2D pose{0.1 * i, 0.0, 0.0};

    const auto connection =
      steering.connect(tree.at(0).pose, pose);

    tree.addNode(
      pose, 0, connection, connection.total_length);
  }

  int checks = 0;
  bool nearest_interrupted = false;

  try {
    sampler.nearest(
      tree,
      {1.0, 0.0, 0.0},
      steering,
      [&]() {return ++checks >= 4;});
  } catch (const PlanningInterrupted &) {
    nearest_interrupted = true;
  }

  require(nearest_interrupted, "Nearest search not interrupted");
  require(checks == 4, "Nearest interruption check count incorrect");

  checks = 0;
  bool near_interrupted = false;

  try {
    sampler.near(
      tree,
      {1.0, 0.0, 0.0},
      steering,
      2.0,
      [&]() {return ++checks >= 4;});
  } catch (const PlanningInterrupted &) {
    near_interrupted = true;
  }

  require(near_interrupted, "Near search not interrupted");
  require(checks == 4, "Near interruption check count incorrect");

  require(tree.size() == 11, "Search modified the tree");

  const int nearest = sampler.nearest(
    tree, {1.0, 0.0, 0.0}, steering);

  require(nearest == 10, "Normal nearest search changed");

  std::cout << "Nearest interruption: PASS\n";
  std::cout << "Near interruption: PASS\n";
  std::cout << "Tree integrity: PASS\n";
  std::cout << "Backward compatibility: PASS\n";

  std::cout << "\nINTERRUPTIBLE NEIGHBORS: PASS\n";

  return 0;
}
