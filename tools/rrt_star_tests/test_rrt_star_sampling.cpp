#include <cmath>
#include <iostream>
#include <stdexcept>

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
  const SamplingBounds bounds{-5.0, 5.0, -3.0, 3.0};
  const Pose2D goal{4.0, 2.0, 1.57};

  RRTStarSampler uniform_sampler(bounds, 0.0, 42);

  for (int i = 0; i < 1000; ++i) {
    const auto pose = uniform_sampler.sample(goal);

    require(pose.x >= bounds.min_x && pose.x <= bounds.max_x,
            "Sample outside X bounds");
    require(pose.y >= bounds.min_y && pose.y <= bounds.max_y,
            "Sample outside Y bounds");
    require(pose.yaw >= -3.141592654 &&
            pose.yaw <= 3.141592654,
            "Sample outside yaw bounds");
  }

  std::cout << "Uniform SE(2) sampling: PASS\n";

  RRTStarSampler goal_sampler(bounds, 1.0, 42);

  for (int i = 0; i < 100; ++i) {
    const auto pose = goal_sampler.sample(goal);

    require(pose.x == goal.x &&
            pose.y == goal.y &&
            pose.yaw == goal.yaw,
            "Goal-biased sample incorrect");
  }

  std::cout << "Goal-biased sampling: PASS\n";

  RRTStarSampler sampler_a(bounds, 0.1, 123);
  RRTStarSampler sampler_b(bounds, 0.1, 123);

  for (int i = 0; i < 100; ++i) {
    const auto a = sampler_a.sample(goal);
    const auto b = sampler_b.sample(goal);

    require(a.x == b.x && a.y == b.y && a.yaw == b.yaw,
            "Random seed is not reproducible");
  }

  std::cout << "Deterministic random seed: PASS\n";

  ReedsSheppSteering steering(0.29, 0.05);
  RRTStarTree tree({0.0, 0.0, 0.0});

  const Pose2D node_pose{2.0, 0.0, 0.0};
  const auto connection = steering.connect(
    tree.at(0).pose, node_pose);

  const int node_index = tree.addNode(
    node_pose, 0, connection, connection.total_length);

  require(
    sampler_a.nearest(tree, node_pose, steering) == node_index,
    "Nearest-neighbor search returned wrong node");

  require(
    sampler_a.nearest(tree, {0.0, 0.0, 0.0}, steering) == 0,
    "Root nearest-neighbor search failed");

  std::cout << "Reeds-Shepp nearest neighbor: PASS\n";
  std::cout << "\nALL RRT* SAMPLING TESTS PASSED\n";

  return 0;
}
