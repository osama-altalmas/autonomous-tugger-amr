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
  RRTStarSampler sampler(
    SamplingBounds{-5.0, 5.0, -5.0, 5.0}, 0.1, 42);
  RRTStarExpansion expansion(
    steering, cost_model, sampler, 1.0, 2.0);

  const auto free_pose = [](const Pose2D &) {
    return true;
  };

  // Interrupt before any node can be inserted.
  RRTStarTree before_tree({0.0, 0.0, 0.0});

  const auto before = expansion.expand(
    before_tree, {3.0, 0.0, 0.0},
    free_pose, {}, []() {return true;});

  require(before.interrupted, "Early interruption not detected");
  require(!before.inserted, "Node inserted after early interruption");
  require(before_tree.size() == 1, "Early interruption changed tree");

  // Trigger interruption after insertion, during the
  // second nearby-node search.
  RRTStarTree after_tree({0.0, 0.0, 0.0});

  int checks = 0;
  const auto after = expansion.expand(
    after_tree,
    {3.0, 0.0, 0.0},
    free_pose,
    {},
    [&]() {
      ++checks;
      return after_tree.size() > 1;
    });

  require(after.interrupted, "Late interruption not detected");
  require(after.inserted, "Inserted node was not reported");
  require(after_tree.size() == 2, "Inserted node was lost");
  require(after.node_index == 1, "Incorrect inserted node index");
  require(
    after_tree.at(1).parent == 0,
    "Inserted node has incorrect parent");

  require(
    after_tree.at(1).cost_from_start >= 0.0,
    "Inserted node has invalid cost");

  std::cout << "Interruption before insertion: PASS\n";
  std::cout << "Interruption after insertion: PASS\n";
  std::cout << "Inserted node preserved: PASS\n";
  std::cout << "Tree consistency: PASS\n";
  std::cout << "\nINTERRUPTIBLE EXPANSION: PASS\n";
  return 0;
}
