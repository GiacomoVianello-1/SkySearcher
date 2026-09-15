cmake_minimum_required(VERSION 3.8)
project(interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/StartMission.action"
  "action/MoveToPosition.action"
  "msg/DetectionPositions.msg"
  "msg/PosteriorField.msg"
  "msg/SemanticBBox.msg"
  "msg/CoverageField.msg"
  "msg/CoveragePolygon.msg"
  "srv/VlmQuery.srv"
  "srv/VlmScoringQuery.srv"
  "srv/AltitudeInference.srv"
  "srv/ExtractConcepts.srv"
  "srv/ExtractCluesBenchmark.srv"
  "srv/ExtractClues.srv"
  "srv/ExtractClueScales.srv"
  "srv/ConfigurePrompts.srv"
  "srv/DetectSemantics.srv"
  "srv/UpdatePosterior.srv"
  "srv/RotateToYaw.srv"
  "srv/SpawnRandom.srv"
  "srv/Spawn.srv"
  "srv/SpawnByPose.srv"
  "srv/TeleportNED.srv"
  "srv/AcquireObservation.srv"
  "srv/VerifyTarget.srv"
  "srv/VerifyTargetBaseline.srv"
  "srv/VisualServoingVerify.srv"
  "srv/GenerateWaypoints.srv"
  "srv/EvaluateWaypoints.srv"
  "srv/GetNextEpisode.srv"
  "srv/ResetBelief.srv"
  DEPENDENCIES sensor_msgs std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
