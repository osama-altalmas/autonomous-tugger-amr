#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/rrt_star_expansion.hpp"

using namespace tugger_navigation;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double a, double b)
{
  return std::abs(a - b) < 1e-6;
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);
  RRTStarCost cost_model;

  const SamplingBounds bounds{-5.0, 5.0, -5.0, 5.0};
  RRTStarSampler sampler(bounds, 0.1, 42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0);

  const auto free_space = [](const Pose2D &) {
    return true;
  };

  RRTStarTree tree({0.0, 0.0, 0.0});

  // Test 1: Expand toward a distant target.
  const auto first = expansion.expand(
    tree, {5.0, 0.0, 0.0}, free_space);

  require(first.inserted, "First expansion rejected");
  require(tree.size() == 2, "Tree size incorrect");
  require(first.parent_index == 0, "Incorrect parent");

  require(
    near(tree.at(first.node_index).pose.x, 1.0),
    "Expansion endpoint incorrect");

  require(
    near(
      tree.at(first.node_index).incoming_connection.total_length,
      1.0),
    "Maximum extension length violated");

  std::cout << "Bounded tree expansion: PASS\n";

  // Test 2: Expand again toward the same target.
  const auto second = expansion.expand(
    tree, {5.0, 0.0, 0.0}, free_space);

  require(second.inserted, "Second expansion rejected");
  require(tree.size() == 3, "Second insertion failed");

  require(
    second.parent_index == first.node_index,
    "Nearest parent selection incorrect");

  require(
    near(tree.at(second.node_index).pose.x, 2.0),
    "Second expansion endpoint incorrect");

  std::cout << "Repeated expansion: PASS\n";

  // Test 3: Collision rejection.
  const auto blocked = [](const Pose2D & pose) {
    return pose.x <= 2.5;
  };

  const std::size_t size_before = tree.size();

  const auto rejected = expansion.expand(
    tree, {5.0, 0.0, 0.0}, blocked);

  require(!rejected.inserted, "Colliding expansion accepted");

  require(
    tree.size() == size_before,
    "Rejected expansion modified tree");

  std::cout << "Collision rejection: PASS\n";

  // Test 4: Cost consistency.
  const auto & node = tree.at(second.node_index);
  const auto & parent = tree.at(node.parent);

  require(
    near(
      node.cost_from_start,
      parent.cost_from_start + node.incoming_edge_cost),
    "Cumulative node cost inconsistent");

  std::cout << "Cumulative cost consistency: PASS\n";

  std::cout << "\nALL RRT* EXPANSION TESTS PASSED\n";

  return 0;
}
