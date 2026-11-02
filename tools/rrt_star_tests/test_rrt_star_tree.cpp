#include <algorithm>
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
  return std::abs(a - b) < 1e-8;
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  RRTStarTree tree({0.0, 0.0, 0.0});

  const Pose2D a{1.0, 0.0, 0.0};
  const Pose2D b{2.0, 0.0, 0.0};
  const Pose2D c{3.0, 0.0, 0.0};
  const Pose2D d{1.0, 1.0, 0.0};

  const int node_a = tree.addNode(
    a, 0,
    steering.connect(tree.at(0).pose, a),
    5.0);

  const int node_b = tree.addNode(
    b, node_a,
    steering.connect(a, b),
    4.0);

  const int node_c = tree.addNode(
    c, node_b,
    steering.connect(b, c),
    3.0);

  const int node_d = tree.addNode(
    d, 0,
    steering.connect(tree.at(0).pose, d),
    2.0);

  require(tree.size() == 5, "Incorrect tree size");

  require(near(tree.at(node_a).cost_from_start, 5.0),
          "Incorrect A cost");

  require(near(tree.at(node_b).cost_from_start, 9.0),
          "Incorrect B cost");

  require(near(tree.at(node_c).cost_from_start, 12.0),
          "Incorrect C cost");

  std::cout << "Initial tree construction: PASS\n";

  // Rewire B through D with a lower edge cost.
  require(
    tree.rewire(
      node_b,
      node_d,
      steering.connect(d, b),
      1.5),
    "Valid rewire rejected");

  require(
    near(tree.at(node_b).cost_from_start, 3.5),
    "B cost not updated");

  require(
    near(tree.at(node_c).cost_from_start, 6.5),
    "Descendant C cost not propagated");

  require(
    tree.at(node_b).parent == node_d,
    "B parent not updated");

  const auto & old_children = tree.at(node_a).children;
  const auto & new_children = tree.at(node_d).children;

  require(
    std::find(
      old_children.begin(), old_children.end(), node_b) ==
      old_children.end(),
    "Old parent still contains B");

  require(
    std::find(
      new_children.begin(), new_children.end(), node_b) !=
      new_children.end(),
    "New parent does not contain B");

  std::cout << "Rewiring and descendant propagation: PASS\n";

  // C is a descendant of B. Making B a child of C
  // would create a cycle.
  require(
    !tree.rewire(
      node_b,
      node_c,
      steering.connect(c, b),
      1.0),
    "Cycle-producing rewire accepted");

  require(
    !tree.rewire(
      0,
      node_c,
      steering.connect(c, tree.at(0).pose),
      1.0),
    "Root rewiring accepted");

  std::cout << "Cycle prevention: PASS\n";

  const std::vector<int> expected_path{
    0, node_d, node_b, node_c
  };

  require(
    tree.pathToRoot(node_c) == expected_path,
    "Incorrect root-to-node path");

  std::cout << "Path extraction: PASS\n";

  std::cout << "\nALL RRT* TREE TESTS PASSED\n";

  return 0;
}
