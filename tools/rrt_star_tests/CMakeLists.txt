cmake_minimum_required(VERSION 3.16)
project(rrt_star_steering_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ompl REQUIRED)

add_library(reeds_shepp_steering STATIC
  ../../src/tugger_navigation/src/reeds_shepp_steering.cpp
)

target_include_directories(reeds_shepp_steering PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(reeds_shepp_steering PUBLIC
  ompl::ompl
)

add_executable(test_reeds_shepp test_reeds_shepp.cpp)
target_link_libraries(test_reeds_shepp PRIVATE ompl::ompl)

add_executable(test_steering_module test_steering_module.cpp)
target_link_libraries(test_steering_module PRIVATE reeds_shepp_steering)

enable_testing()

add_test(
  NAME steering_module
  COMMAND test_steering_module
)

add_test(
  NAME original_reeds_shepp
  COMMAND test_reeds_shepp
)

add_executable(
  test_steering_validation
  test_steering_validation.cpp
)

target_link_libraries(
  test_steering_validation
  PRIVATE reeds_shepp_steering
)

add_test(
  NAME steering_validation
  COMMAND test_steering_validation
)

add_library(rrt_star_tree STATIC
  ../../src/tugger_navigation/src/rrt_star_tree.cpp
)

target_include_directories(rrt_star_tree PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(rrt_star_tree PUBLIC
  reeds_shepp_steering
  rrt_star_cost
)

add_executable(test_rrt_star_tree
  test_rrt_star_tree.cpp
)

target_link_libraries(test_rrt_star_tree PRIVATE
  rrt_star_tree
)

add_test(
  NAME rrt_star_tree
  COMMAND test_rrt_star_tree
)

add_library(rrt_star_cost STATIC
  ../../src/tugger_navigation/src/rrt_star_cost.cpp
)

target_include_directories(rrt_star_cost PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(rrt_star_cost PUBLIC
  reeds_shepp_steering
)

add_executable(test_rrt_star_cost
  test_rrt_star_cost.cpp
)

target_link_libraries(test_rrt_star_cost PRIVATE
  rrt_star_cost
)

add_test(
  NAME rrt_star_cost
  COMMAND test_rrt_star_cost
)

add_executable(test_direction_aware_rewire
  test_direction_aware_rewire.cpp
)

target_link_libraries(test_direction_aware_rewire PRIVATE
  rrt_star_tree
  rrt_star_cost
)

add_test(
  NAME direction_aware_rewire
  COMMAND test_direction_aware_rewire
)

add_library(rrt_star_sampling STATIC
  ../../src/tugger_navigation/src/rrt_star_sampling.cpp
)

target_include_directories(rrt_star_sampling PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(rrt_star_sampling PUBLIC
  rrt_star_tree
)

add_executable(test_rrt_star_sampling
  test_rrt_star_sampling.cpp
)

target_link_libraries(test_rrt_star_sampling PRIVATE
  rrt_star_sampling
)

add_test(
  NAME rrt_star_sampling
  COMMAND test_rrt_star_sampling
)

add_executable(test_limited_steering
  test_limited_steering.cpp
)

target_link_libraries(test_limited_steering PRIVATE
  reeds_shepp_steering
)

add_test(
  NAME limited_steering
  COMMAND test_limited_steering
)

add_executable(test_limited_cusps
  test_limited_cusps.cpp
)

target_link_libraries(test_limited_cusps PRIVATE
  reeds_shepp_steering
)

add_test(
  NAME limited_cusps
  COMMAND test_limited_cusps
)

add_library(rrt_star_expansion STATIC
  ../../src/tugger_navigation/src/rrt_star_expansion.cpp
)

target_include_directories(rrt_star_expansion PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(rrt_star_expansion PUBLIC
  rrt_star_sampling
  rrt_star_cost
)

add_executable(test_rrt_star_expansion
  test_rrt_star_expansion.cpp
)

target_link_libraries(test_rrt_star_expansion PRIVATE
  rrt_star_expansion
)

add_test(
  NAME rrt_star_expansion
  COMMAND test_rrt_star_expansion
)

add_executable(test_rrt_star_neighbors
  test_rrt_star_neighbors.cpp
)

target_link_libraries(test_rrt_star_neighbors PRIVATE
  rrt_star_sampling
)

add_test(
  NAME rrt_star_neighbors
  COMMAND test_rrt_star_neighbors
)

add_executable(test_best_parent
  test_best_parent.cpp
)

target_link_libraries(test_best_parent PRIVATE
  rrt_star_expansion
)

add_test(
  NAME best_parent
  COMMAND test_best_parent
)

add_executable(test_expansion_rewiring
  test_expansion_rewiring.cpp
)

target_link_libraries(test_expansion_rewiring PRIVATE
  rrt_star_expansion
)

add_test(
  NAME expansion_rewiring
  COMMAND test_expansion_rewiring
)

add_library(rrt_star_planner_core STATIC
  ../../src/tugger_navigation/src/rrt_star_planner_core.cpp
)

target_include_directories(rrt_star_planner_core PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(rrt_star_planner_core PUBLIC
  rrt_star_expansion
)

add_executable(test_rrt_star_planner_core
  test_rrt_star_planner_core.cpp
)

target_link_libraries(test_rrt_star_planner_core PRIVATE
  rrt_star_planner_core
)

add_test(
  NAME rrt_star_planner_core
  COMMAND test_rrt_star_planner_core
)

add_executable(test_trajectory_geometry
  test_trajectory_geometry.cpp
)

target_link_libraries(test_trajectory_geometry PRIVATE
  rrt_star_planner_core
)

add_test(
  NAME trajectory_geometry
  COMMAND test_trajectory_geometry
)

add_executable(test_obstacle_trajectory
  test_obstacle_trajectory.cpp
)

target_link_libraries(test_obstacle_trajectory PRIVATE
  rrt_star_planner_core
)

add_test(
  NAME obstacle_trajectory
  COMMAND test_obstacle_trajectory
)

add_library(rrt_star_trajectory STATIC
  ../../src/tugger_navigation/src/rrt_star_trajectory.cpp
)

target_include_directories(rrt_star_trajectory PUBLIC
  ../../src/tugger_navigation/include
)

target_link_libraries(rrt_star_trajectory PUBLIC
  reeds_shepp_steering
)

target_link_libraries(rrt_star_planner_core PUBLIC
  rrt_star_trajectory
)

add_executable(test_boundary_cusp
  test_boundary_cusp.cpp
)

target_link_libraries(test_boundary_cusp PRIVATE
  rrt_star_trajectory
)

add_test(
  NAME boundary_cusp
  COMMAND test_boundary_cusp
)

add_executable(test_planning_deadline
  test_planning_deadline.cpp
)

target_link_libraries(test_planning_deadline PRIVATE
  rrt_star_planner_core
)

add_test(
  NAME planning_deadline
  COMMAND test_planning_deadline
)

add_executable(test_unreachable_deadline
  test_unreachable_deadline.cpp
)

target_link_libraries(test_unreachable_deadline PRIVATE
  rrt_star_planner_core
)

add_test(
  NAME unreachable_deadline
  COMMAND test_unreachable_deadline
)

add_executable(test_interruptible_collision
  test_interruptible_collision.cpp
)

target_link_libraries(test_interruptible_collision PRIVATE
  reeds_shepp_steering
)

add_test(
  NAME interruptible_collision
  COMMAND test_interruptible_collision
)

add_executable(test_interruptible_neighbors
  test_interruptible_neighbors.cpp
)

target_link_libraries(test_interruptible_neighbors PRIVATE
  rrt_star_sampling
)

add_test(
  NAME interruptible_neighbors
  COMMAND test_interruptible_neighbors
)

add_executable(test_interruptible_expansion
  test_interruptible_expansion.cpp
)

target_link_libraries(test_interruptible_expansion PRIVATE
  rrt_star_expansion
)

add_test(
  NAME interruptible_expansion
  COMMAND test_interruptible_expansion
)

add_executable(test_initial_goal_cancellation
  test_initial_goal_cancellation.cpp
)

target_link_libraries(test_initial_goal_cancellation PRIVATE
  rrt_star_planner_core
)

add_test(
  NAME initial_goal_cancellation
  COMMAND test_initial_goal_cancellation
)

add_executable(test_interruptible_rewire
  test_interruptible_rewire.cpp
)

target_link_libraries(test_interruptible_rewire PRIVATE
  rrt_star_tree
)

add_test(
  NAME interruptible_rewire
  COMMAND test_interruptible_rewire
)
