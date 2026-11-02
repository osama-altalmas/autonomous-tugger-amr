#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/rrt_star_tree.hpp"

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

  RRTStarCostWeights weights;
  weights.distance = 1.0;
  weights.reverse = 0.0;
  weights.gear_change = 10.0;
  weights.steering = 0.0;
  weights.cardinal = 0.0;
  weights.obstacle = 0.0;

  RRTStarCost cost_model(weights);
  RRTStarTree tree({0.0, 0.0, 0.0});

  const Pose2D a{1.0, 0.0, 0.0};
  const Pose2D b{2.0, 0.0, 0.0};
  const Pose2D c{3.0, 0.0, 0.0};
  const Pose2D d{3.0, 0.0, 0.0};

  const auto root_to_a =
    steering.connect(tree.at(0).pose, a);

  // Deliberately assign an expensive initial edge.
  // This creates a deterministic improvement opportunity.
  const int node_a = tree.addNode(
    a, 0, root_to_a, 100.0);

  const auto a_to_b = steering.connect(a, b);

  const int node_b = tree.addNode(
    b, node_a, a_to_b,
    cost_model.evaluate(a_to_b, 1).total());

  const auto b_to_c = steering.connect(b, c);

  const int node_c = tree.addNode(
    c, node_b, b_to_c,
    cost_model.evaluate(b_to_c, 1).total());

  // D is located ahead of B. Reaching B from D
  // requires reversing.
  const auto root_to_d =
    steering.connect(tree.at(0).pose, d);

  const int node_d = tree.addNode(
    d, 0, root_to_d,
    cost_model.evaluate(root_to_d).total());

  const auto d_to_b = steering.connect(d, b);

  require(
    d_to_b.segments.front().direction == -1,
    "D-to-B connection does not begin in reverse");

  require(
    d_to_b.segments.back().direction == -1,
    "D-to-B connection does not end in reverse");

  const double old_b_cost =
    tree.at(node_b).cost_from_start;

  const double old_c_edge =
    tree.at(node_c).incoming_edge_cost;

  const bool rewired = tree.rewireWithCostModel(
    node_b, node_d, d_to_b, cost_model);

  require(rewired, "Expected improving rewire was rejected");

  require(
    tree.at(node_b).parent == node_d,
    "B parent was not updated");

  require(
    tree.at(node_b).arrival_direction == -1,
    "B arrival direction was not updated");

  const double expected_c_edge = cost_model.evaluate(
    tree.at(node_c).incoming_connection,
    tree.at(node_b).arrival_direction).total();

  require(
    near(tree.at(node_c).incoming_edge_cost, expected_c_edge),
    "C incoming edge cost was not recalculated");

  require(
    near(tree.at(node_c).incoming_edge_cost - old_c_edge, 10.0),
    "Expected additional gear-change penalty of 10");

  require(
    near(
      tree.at(node_c).cost_from_start,
      tree.at(node_b).cost_from_start + expected_c_edge),
    "C cumulative cost is inconsistent");

  require(
    tree.at(node_b).cost_from_start < old_b_cost,
    "B cost did not improve");

  std::cout << "Rewire accepted: yes\n";
  std::cout << "B cost before: " << old_b_cost << '\n';
  std::cout << "B cost after: "
            << tree.at(node_b).cost_from_start << '\n';

  std::cout << "B arrival direction: "
            << tree.at(node_b).arrival_direction << '\n';

  std::cout << "C incoming edge before: "
            << old_c_edge << '\n';

  std::cout << "C incoming edge after: "
            << tree.at(node_c).incoming_edge_cost << '\n';

  std::cout << "Additional C gear-change cost: "
            << tree.at(node_c).incoming_edge_cost - old_c_edge
            << '\n';

  std::cout << "\nDIRECTION-AWARE REWIRING: PASS\n";

  return 0;
}
