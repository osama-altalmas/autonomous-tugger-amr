#include "tugger_navigation/rrt_star_tree.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tugger_navigation
{

namespace
{

int finalDirection(const SteeringResult & connection)
{
  if (connection.segments.empty()) {
    return 0;
  }

  return connection.segments.back().direction;
}

bool validEdgeCost(double cost)
{
  return std::isfinite(cost) && cost >= 0.0;
}

}  // namespace

RRTStarTree::RRTStarTree(const Pose2D & start)
{
  RRTStarNode root;
  root.pose = start;
  root.parent = -1;
  root.cost_from_start = 0.0;
  root.incoming_edge_cost = 0.0;
  root.arrival_direction = 0;

  nodes_.push_back(std::move(root));
}

int RRTStarTree::addNode(
  const Pose2D & pose,
  int parent,
  const SteeringResult & connection,
  double edge_cost)
{
  if (parent < 0 ||
      static_cast<std::size_t>(parent) >= nodes_.size()) {
    throw std::out_of_range("Invalid RRT* parent index");
  }

  if (!connection.valid || connection.segments.empty()) {
    throw std::invalid_argument("Invalid incoming connection");
  }

  if (!validEdgeCost(edge_cost)) {
    throw std::invalid_argument("Invalid incoming edge cost");
  }

  RRTStarNode node;
  node.pose = pose;
  node.parent = parent;
  node.incoming_edge_cost = edge_cost;
  node.cost_from_start =
    nodes_[parent].cost_from_start + edge_cost;
  node.arrival_direction = finalDirection(connection);
  node.incoming_connection = connection;

  const int index = static_cast<int>(nodes_.size());

  nodes_.push_back(std::move(node));
  nodes_[parent].children.push_back(index);

  return index;
}

bool RRTStarTree::isAncestor(
  int ancestor,
  int node) const
{
  if (ancestor < 0 || node < 0 ||
      static_cast<std::size_t>(ancestor) >= nodes_.size() ||
      static_cast<std::size_t>(node) >= nodes_.size()) {
    return false;
  }

  int current = node;

  while (current != -1) {
    if (current == ancestor) {
      return true;
    }

    current = nodes_[current].parent;
  }

  return false;
}

bool RRTStarTree::rewire(
  int node,
  int new_parent,
  const SteeringResult & connection,
  double edge_cost)
{
  if (node <= 0 || new_parent < 0 ||
      static_cast<std::size_t>(node) >= nodes_.size() ||
      static_cast<std::size_t>(new_parent) >= nodes_.size()) {
    return false;
  }

  if (node == new_parent ||
      isAncestor(node, new_parent)) {
    return false;
  }

  if (!connection.valid || connection.segments.empty() ||
      !validEdgeCost(edge_cost)) {
    return false;
  }

  const int old_parent = nodes_[node].parent;

  if (old_parent == new_parent) {
    return false;
  }

  auto & old_children = nodes_[old_parent].children;

  old_children.erase(
    std::remove(
      old_children.begin(),
      old_children.end(),
      node),
    old_children.end());

  nodes_[node].parent = new_parent;
  nodes_[node].incoming_edge_cost = edge_cost;
  nodes_[node].incoming_connection = connection;
  nodes_[node].arrival_direction = finalDirection(connection);

  nodes_[new_parent].children.push_back(node);

  propagateCosts(node);

  return true;
}

bool RRTStarTree::rewireWithCostModel(
  int node,
  int new_parent,
  const SteeringResult & connection,
  const RRTStarCost & cost_model,
  const RRTStarCost::ObstaclePenalty & obstacle_penalty,
  const std::function<bool()> & should_stop)
{
  if (node <= 0 || new_parent < 0 ||
      static_cast<std::size_t>(node) >= nodes_.size() ||
      static_cast<std::size_t>(new_parent) >= nodes_.size() ||
      node == new_parent ||
      isAncestor(node, new_parent) ||
      nodes_[node].parent == new_parent ||
      !connection.valid || connection.segments.empty()) {
    return false;
  }

  if (should_stop && should_stop()) {
    return false;
  }

  const double new_edge_cost = cost_model.evaluate(
    connection,
    nodes_[new_parent].arrival_direction,
    obstacle_penalty).total();

  const double new_total_cost =
    nodes_[new_parent].cost_from_start + new_edge_cost;

  if (!std::isfinite(new_total_cost) ||
      new_total_cost >= nodes_[node].cost_from_start - 1e-9) {
    return false;
  }

  // Calculate the entire affected subtree before modifying
  // parent-child relationships.
  struct CostUpdate {
    int index;
    double edge_cost;
    double total_cost;
  };

  std::vector<CostUpdate> updates;
  updates.push_back({node, new_edge_cost, new_total_cost});

  std::size_t cursor = 0;

  while (cursor < updates.size()) {
    if (should_stop && should_stop()) {
      return false;
    }

    const auto current = updates[cursor++];

    for (const int child : nodes_[current.index].children) {
      if (should_stop && should_stop()) {
        return false;
      }

      const double child_edge_cost = cost_model.evaluate(
        nodes_[child].incoming_connection,
        finalDirection(
          current.index == node ?
          connection : nodes_[current.index].incoming_connection),
        obstacle_penalty).total();

      const double child_total_cost =
        current.total_cost + child_edge_cost;

      if (!std::isfinite(child_total_cost)) {
        return false;
      }

      updates.push_back({
        child,
        child_edge_cost,
        child_total_cost
      });
    }
  }

  if (should_stop && should_stop()) {
    return false;
  }

  const int old_parent = nodes_[node].parent;

  auto & old_children = nodes_[old_parent].children;

  old_children.erase(
    std::remove(
      old_children.begin(),
      old_children.end(),
      node),
    old_children.end());

  nodes_[new_parent].children.push_back(node);

  nodes_[node].parent = new_parent;
  nodes_[node].incoming_connection = connection;
  nodes_[node].arrival_direction = finalDirection(connection);

  for (const auto & update : updates) {
    nodes_[update.index].incoming_edge_cost = update.edge_cost;
    nodes_[update.index].cost_from_start = update.total_cost;
  }

  return true;
}

void RRTStarTree::propagateCosts(int node)
{
  // Iterative traversal avoids recursive stack overflow
  // when the RRT* tree becomes deep.
  std::vector<int> pending;
  pending.push_back(node);

  while (!pending.empty()) {
    const int current = pending.back();
    pending.pop_back();

    auto & current_node = nodes_[current];

    if (current_node.parent >= 0) {
      current_node.cost_from_start =
        nodes_[current_node.parent].cost_from_start +
        current_node.incoming_edge_cost;
    }

    for (const int child : current_node.children) {
      pending.push_back(child);
    }
  }
}

std::vector<int> RRTStarTree::pathToRoot(int node) const
{
  if (node < 0 ||
      static_cast<std::size_t>(node) >= nodes_.size()) {
    throw std::out_of_range("Invalid RRT* node index");
  }

  std::vector<int> path;

  int current = node;

  while (current != -1) {
    path.push_back(current);
    current = nodes_[current].parent;
  }

  std::reverse(path.begin(), path.end());

  return path;
}

const RRTStarNode & RRTStarTree::at(int index) const
{
  return nodes_.at(static_cast<std::size_t>(index));
}

std::size_t RRTStarTree::size() const
{
  return nodes_.size();
}

}  // namespace tugger_navigation
