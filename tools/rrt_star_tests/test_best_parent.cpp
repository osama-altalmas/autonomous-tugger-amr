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

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);
  RRTStarCost cost_model;

  const SamplingBounds bounds{-5.0, 5.0, -5.0, 5.0};
  RRTStarSampler sampler(bounds, 0.0, 42);

  RRTStarExpansion expansion(
    steering, cost_model, sampler,
    1.0, 2.0);

  const auto free_space = [](const Pose2D &) {
    return true;
  };

  RRTStarTree tree({-3.0, 0.0, 0.0});

  const Pose2D expensive_pose{0.5, 0.0, 0.0};
  const Pose2D cheap_pose{0.0, 0.0, 0.0};
  const Pose2D target{0.8, 0.0, 0.0};

  // Construct a deliberately expensive node.
  const auto expensive_connection = steering.connect(
    tree.at(0).pose, expensive_pose);

  const int expensive_index = tree.addNode(
    expensive_pose,
    0,
    expensive_connection,
    100.0);

  // Construct a cheaper alternative parent.
  const auto cheap_connection = steering.connect(
    tree.at(0).pose, cheap_pose);

  const int cheap_index = tree.addNode(
    cheap_pose,
    0,
    cheap_connection,
    3.0);

  const int nearest_index =
    sampler.nearest(tree, target, steering);

  require(
    nearest_index == expensive_index,
    "Test setup failed: expensive node is not nearest");

  std::cout << "Nearest node: "
            << nearest_index << '\n';

  std::cout << "Expensive node cost: "
            << tree.at(expensive_index).cost_from_start << '\n';

  std::cout << "Cheap node cost: "
            << tree.at(cheap_index).cost_from_start << '\n';

  const auto result = expansion.expand(
    tree, target, free_space);

  require(result.inserted, "Expansion was rejected");

  require(
    result.parent_index == cheap_index,
    "Best-parent selection chose the wrong parent");

  const auto & new_node = tree.at(result.node_index);

  require(
    new_node.incoming_connection.total_length <= 1.0 + 1e-8,
    "Selected connection exceeds extension limit");

  require(
    new_node.cost_from_start <
      tree.at(expensive_index).cost_from_start,
    "Selected path was not cheaper");

  std::cout << "Selected parent: "
            << result.parent_index << '\n';

  std::cout << "New node cost: "
            << new_node.cost_from_start << '\n';

  std::cout << "Connection length: "
            << new_node.incoming_connection.total_length << '\n';

  std::cout << "\nBEST-PARENT SELECTION: PASS\n";

  return 0;
}
