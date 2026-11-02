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

bool near(double a, double b, double tolerance = 1e-6)
{
  return std::abs(a - b) < tolerance;
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);
  RRTStarCost cost_model;

  RRTStarSampler sampler(
    SamplingBounds{-5.0, 5.0, -5.0, 5.0},
    0.0, 42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0, 2.0);

  const auto free_space = [](const Pose2D &) {
    return true;
  };

  RRTStarTree tree({0.0, 0.0, 0.0});

  // Cheap branch: root -> A.
  const Pose2D a{1.0, 0.0, 0.0};

  const auto root_to_a =
    steering.connect(tree.at(0).pose, a);

  const int node_a = tree.addNode(
    a, 0, root_to_a,
    cost_model.evaluate(root_to_a).total());

  // Deliberately expensive branch: root -> B -> C.
  const Pose2D b{2.0, 0.0, 0.0};
  const Pose2D c{2.5, 0.0, 0.0};

  const auto root_to_b =
    steering.connect(tree.at(0).pose, b);

  const int node_b = tree.addNode(
    b, 0, root_to_b, 100.0);

  const auto b_to_c = steering.connect(b, c);

  const int node_c = tree.addNode(
    c, node_b, b_to_c,
    cost_model.evaluate(
      b_to_c, tree.at(node_b).arrival_direction).total());

  const double old_b_cost =
    tree.at(node_b).cost_from_start;

  const double old_c_cost =
    tree.at(node_c).cost_from_start;

  // The nearest node to this target is B, but A is
  // the cheapest feasible parent.
  const Pose2D target{1.6, 0.0, 0.0};

  require(
    sampler.nearest(tree, target, steering) == node_b,
    "Test setup: B should be nearest");

  const auto result =
    expansion.expand(tree, target, free_space);

  require(result.inserted, "Expansion failed");

  require(
    result.parent_index == node_a,
    "New node did not choose cheap parent A");

  require(
    result.rewired_nodes >= 1,
    "No existing nodes were rewired");

  require(
    tree.at(node_b).parent == result.node_index,
    "Expensive node B was not rewired");

  require(
    tree.at(node_b).cost_from_start < old_b_cost,
    "Node B cost did not improve");

  require(
    tree.at(node_c).cost_from_start < old_c_cost,
    "Descendant C cost did not improve");

  require(
    near(
      tree.at(node_c).cost_from_start,
      tree.at(node_b).cost_from_start +
        tree.at(node_c).incoming_edge_cost),
    "Descendant cost propagation inconsistent");

  std::cout << "New node index: "
            << result.node_index << '\n';

  std::cout << "New node parent: "
            << result.parent_index << '\n';

  std::cout << "Rewired nodes: "
            << result.rewired_nodes << '\n';

  std::cout << "B cost before: "
            << old_b_cost << '\n';

  std::cout << "B cost after: "
            << tree.at(node_b).cost_from_start << '\n';

  std::cout << "C cost before: "
            << old_c_cost << '\n';

  std::cout << "C cost after: "
            << tree.at(node_c).cost_from_start << '\n';

  std::cout << "\nAUTOMATIC RRT* REWIRING: PASS\n";

  return 0;
}
