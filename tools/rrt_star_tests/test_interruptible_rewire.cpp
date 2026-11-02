#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
  return std::abs(a - b) < 1e-9;
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
  const Pose2D e{4.0, 0.0, 0.0};
  const Pose2D d{3.0, 0.0, 0.0};

  const auto root_to_a = steering.connect(
    tree.at(0).pose, a);

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

  const auto c_to_e = steering.connect(c, e);
  const int node_e = tree.addNode(
    e, node_c, c_to_e,
    cost_model.evaluate(c_to_e, 1).total());

  const auto root_to_d = steering.connect(
    tree.at(0).pose, d);

  const int node_d = tree.addNode(
    d, 0, root_to_d,
    cost_model.evaluate(root_to_d).total());

  const auto d_to_b = steering.connect(d, b);

  require(
    d_to_b.segments.back().direction == -1,
    "Expected reverse arrival at B");

  // Snapshot every node before attempting the rewire.
  struct Snapshot
  {
    int parent;
    std::vector<int> children;
    double total_cost;
    double edge_cost;
    int arrival_direction;
  };

  std::vector<Snapshot> before;

  for (std::size_t i = 0; i < tree.size(); ++i) {
    const auto & node = tree.at(static_cast<int>(i));

    before.push_back({
      node.parent,
      node.children,
      node.cost_from_start,
      node.incoming_edge_cost,
      node.arrival_direction
    });
  }

  // Interrupt during subtree traversal, after several
  // cost calculations but before the commit.
  int checks = 0;

  const bool interrupted_rewire = tree.rewireWithCostModel(
    node_b,
    node_d,
    d_to_b,
    cost_model,
    {},
    [&]() {
      return ++checks >= 4;
    });

  require(!interrupted_rewire, "Interrupted rewire was committed");
  require(checks == 4, "Unexpected interruption check count");
  require(tree.size() == before.size(), "Tree size changed");

  for (std::size_t i = 0; i < tree.size(); ++i) {
    const auto & node = tree.at(static_cast<int>(i));
    const auto & original = before[i];

    require(node.parent == original.parent, "Parent changed");
    require(node.children == original.children, "Children changed");

    require(
      near(node.cost_from_start, original.total_cost),
      "Accumulated cost changed");

    require(
      near(node.incoming_edge_cost, original.edge_cost),
      "Incoming edge cost changed");

    require(
      node.arrival_direction == original.arrival_direction,
      "Arrival direction changed");
  }

  // Confirm that interruption did not damage the tree.
  const bool successful_rewire = tree.rewireWithCostModel(
    node_b, node_d, d_to_b, cost_model);

  require(successful_rewire, "Uninterrupted rewire failed");

  require(
    tree.at(node_b).parent == node_d,
    "Parent not updated after successful rewire");

  require(
    tree.at(node_c).parent == node_b,
    "Descendant C lost");

  require(
    tree.at(node_e).parent == node_c,
    "Descendant E lost");

  std::cout << "Interrupted rewire rejected: PASS\n";
  std::cout << "Tree snapshot preserved: PASS\n";
  std::cout << "Uninterrupted rewire accepted: PASS\n";
  std::cout << "Descendant relationships preserved: PASS\n";
  std::cout << "\nINTERRUPTIBLE REWIRE: PASS\n";

  return 0;
}
