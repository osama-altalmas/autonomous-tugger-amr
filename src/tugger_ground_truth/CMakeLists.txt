cmake_minimum_required(VERSION 3.16)
project(tugger_ground_truth)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(gz-transport REQUIRED CONFIG)
find_package(gz-msgs REQUIRED CONFIG)

add_executable(
  ground_truth_bridge
  src/ground_truth_bridge.cpp
)

target_link_libraries(
  ground_truth_bridge
  PRIVATE
    rclcpp::rclcpp
    ${geometry_msgs_TARGETS}
    gz-transport::gz-transport
    gz-msgs::gz-msgs
)

install(
  TARGETS ground_truth_bridge
  DESTINATION lib/${PROJECT_NAME}
)


install(
  PROGRAMS
    scripts/ground_truth_localization.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
