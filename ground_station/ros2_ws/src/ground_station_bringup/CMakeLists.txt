cmake_minimum_required(VERSION 3.8)
project(ground_station_bringup)
 
find_package(ament_cmake REQUIRED)
 
install(DIRECTORY launch config scripts
  DESTINATION share/${PROJECT_NAME}
)
 
install(PROGRAMS
  scripts/tf_broadcaster.py
  DESTINATION lib/${PROJECT_NAME}
)
 
ament_package()
 