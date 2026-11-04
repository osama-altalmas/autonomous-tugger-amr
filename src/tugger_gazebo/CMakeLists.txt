cmake_minimum_required(VERSION 3.8)
project(tugger_gazebo)

find_package(ament_cmake REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)
find_package(gz-transport REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(tugger_guarded_cargo_joint SHARED src/guarded_cargo_joint.cpp)
target_compile_features(tugger_guarded_cargo_joint PRIVATE cxx_std_17)
target_link_libraries(tugger_guarded_cargo_joint PRIVATE
  gz-sim::gz-sim gz-plugin::register gz-transport::gz-transport)
install(TARGETS tugger_guarded_cargo_joint LIBRARY DESTINATION lib)

add_library(tugger_collision_scene SHARED src/collision_scene.cpp)
target_compile_features(tugger_collision_scene PRIVATE cxx_std_17)
target_link_libraries(tugger_collision_scene PRIVATE
  gz-sim::gz-sim gz-plugin::register gz-transport::gz-transport nlohmann_json::nlohmann_json)
install(TARGETS tugger_collision_scene LIBRARY DESTINATION lib)

install(
  DIRECTORY
    launch
    worlds
    config
    media
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
