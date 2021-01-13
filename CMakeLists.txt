cmake_minimum_required(VERSION 2.8.3)
project(home_coverage_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(ROS_PKG_DEPS
	genmsg
	rospy
	std_msgs
	geometry_msgs
	actionlib_msgs
)

find_package(catkin REQUIRED COMPONENTS ${ROS_PKG_DEPS})

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate services in the 'srv' folder
add_service_files(
     FILES
     HvtInfoGet.srv
     HvtInfoAdd.srv
	HvtInfoDel.srv
     SearchPolygon.srv
	StdDataType.srv
)
add_message_files(
	DIRECTORY msg
	FILES HighValueTarget.msg
)
add_action_files(
	DIRECTORY action
	FILES CoveragePlanner.action
)

generate_messages(
     DEPENDENCIES
     std_msgs
     geometry_msgs
	actionlib_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES path_coverage
  CATKIN_DEPENDS message_runtime ${${ROS_PKG_DEPS}}
)

###########
## Build ##
###########

include_directories(${catkin_INCLUDE_DIRS} )
