cmake_minimum_required(VERSION 3.0.2)
project(hybrid_vlo)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 14)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(OpenCV REQUIRED)

set(PCL_DIR /usr/local/pcl_1.7/share/pcl-1.7)
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})


find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIRS}")

# find_package(Sophus REQUIRED)
set(Sophus_INCLUDE_DIRS /usr/local/Sophus/include)
set(Sophus_LIBRARIES /usr/local/Sophus/lib/libSophus.so)


#vikit_common
set(vikit_common /home/kinggreat24/ORB_SLAM2/Thirdparty/vikit_common/build)
find_package(vikit_common REQUIRED)


find_package(CUDA REQUIRED)

# SGM
set(SGM_INCLUDE_DIRS "/home/kinggreat24/cmake_ws/libSGM/include")
set(SGM_LIBRARIES "/home/kinggreat24/cmake_ws/libSGM/build/src/libsgm.a")

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES hybrid_vlo
#  CATKIN_DEPENDS cv_bridge nav_msgs roscpp rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  include/hybrid_vlo
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${vikit_common_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${SGM_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/Thirdparty/fadeRelease_v1.82
)

# ceres视觉激光直接法
cuda_add_executable(hvlo 
    src/hybrid_visual_odometry.cpp 
    src/Frame.cpp 
    src/ORBextractor.cc 
    src/utils.cpp 
    src/lidar_sparse_align/FeatureAlign.cc
    src/lidar_sparse_align/SparseLidarAlign.cc
    src/lidar_sparse_align/WeightFunction.cpp
)
target_link_libraries(hvlo 
    ${CUDA_LIBRARIES}   
    ${SGM_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${EIGEN3__LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${vikit_common_LIBRARIES}
    ${Sophus_LIBRARIES} 
    ${catkin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/fadeRelease_v1.82/lib_ubuntu18.04_x86_64/libfade25d.so
)
