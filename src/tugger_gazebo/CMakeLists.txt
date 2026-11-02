cmake_minimum_required(VERSION 3.8)
project(tugger_gazebo)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    worlds
    config
    media
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
