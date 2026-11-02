cmake_minimum_required(VERSION 3.10)
project(tugger_navigation)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_smac_planner REQUIRED)
find_package(nav2_ros_common REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ompl REQUIRED)

add_library(tugger_bidirectional_planner SHARED
  src/bidirectional_planner.cpp
)

target_include_directories(tugger_bidirectional_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(tugger_bidirectional_planner PUBLIC
  rclcpp::rclcpp
  pluginlib::pluginlib
  nav2_core::nav2_core
  nav2_costmap_2d::nav2_costmap_2d_core
  nav2_smac_planner::nav2_smac_planner
  nav2_ros_common::nav2_ros_common
  ${nav_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  tf2::tf2
  ${tf2_geometry_msgs_TARGETS}
)


# Custom RRT* Reeds-Shepp Nav2 global planner.
add_library(tugger_rrt_star_planner SHARED
  src/rrt_star_reeds_shepp_planner.cpp
  src/reeds_shepp_steering.cpp
  src/rrt_star_cost.cpp
  src/rrt_star_tree.cpp
  src/rrt_star_sampling.cpp
  src/rrt_star_expansion.cpp
  src/rrt_star_planner_core.cpp
  src/rrt_star_trajectory.cpp
)

target_include_directories(tugger_rrt_star_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(tugger_rrt_star_planner PUBLIC
  rclcpp::rclcpp
  pluginlib::pluginlib
  nav2_core::nav2_core
  nav2_costmap_2d::nav2_costmap_2d_core
  nav2_ros_common::nav2_ros_common
  ompl::ompl
  ${nav_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  tf2::tf2
  ${tf2_geometry_msgs_TARGETS}
)

pluginlib_export_plugin_description_file(
  nav2_core
  tugger_navigation_plugins.xml
)

install(
  TARGETS tugger_bidirectional_planner tugger_rrt_star_planner
  EXPORT export_tugger_bidirectional_planner
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY
    launch
    config
    behavior_trees
    maps
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(
  export_tugger_bidirectional_planner
  HAS_LIBRARY_TARGET
)

ament_export_dependencies(
  rclcpp
  pluginlib
  nav2_core
  nav2_costmap_2d
  nav2_smac_planner
  nav2_ros_common
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
  ompl
)


install(PROGRAMS
  scripts/slam_scan_filter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
