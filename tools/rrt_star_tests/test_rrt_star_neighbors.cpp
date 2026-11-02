#include <algorithm>
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

bool contains(const std::vector<int> & nodes, int index)
{
  return std::find(nodes.begin(), nodes.end(), index)
    != nodes.end();
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  const SamplingBounds bounds{-5.0, 5.0, -5.0, 5.0};
  RRTStarSampler sampler(bounds, 0.1, 42);

  RRTStarTree tree({0.0, 0.0, 0.0});

  const Pose2D a{1.0, 0.0, 0.0};
  const Pose2D b{2.0, 0.0, 0.0};
  const Pose2D c{4.0, 0.0, 0.0};

  auto add = [&](const Pose2D & pose, int parent) {
    const auto connection =
      steering.connect(tree.at(parent).pose, pose);

    return tree.addNode(
      pose, parent, connection, connection.total_length);
  };

  const int node_a = add(a, 0);
  const int node_b = add(b, node_a);
  const int node_c = add(c, node_b);

  const auto neighbors = sampler.near(
    tree, {1.5, 0.0, 0.0}, steering, 1.0);

  require(
    contains(neighbors, node_a),
    "Node A missing from nearby set");

  require(
    contains(neighbors, node_b),
    "Node B missing from nearby set");

  require(
    !contains(neighbors, node_c),
    "Distant node C incorrectly included");

  require(
    !contains(neighbors, 0),
    "Root incorrectly included");

  std::cout << "Nearby-node selection: PASS\n";

  const auto small_radius = sampler.near(
    tree, {1.5, 0.0, 0.0}, steering, 0.1);

  require(
    small_radius.empty(),
    "Small radius unexpectedly returned neighbors");

  std::cout << "Radius filtering: PASS\n";

  const auto large_radius = sampler.near(
    tree, {1.5, 0.0, 0.0}, steering, 5.0);

  require(
    large_radius.size() == tree.size(),
    "Large radius did not include all nodes");

  std::cout << "Large-radius selection: PASS\n";

  std::cout << "\nALL RRT* NEIGHBOR TESTS PASSED\n";

  return 0;
}
