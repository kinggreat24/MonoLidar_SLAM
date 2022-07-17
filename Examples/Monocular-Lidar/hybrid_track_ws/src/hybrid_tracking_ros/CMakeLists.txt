cmake_minimum_required(VERSION 2.8.3)
project(hybrid_tracking_ros)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  tf
  eigen_conversions
  cv_bridge
)


set(EIGEN3_FOUND true)
find_package(Eigen3 REQUIRED)
IF(EIGEN3_FOUND) 
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
MESSAGE("EIGEN3 [OK]:" ${EIGEN3_INCLUDE_DIR})
ENDIF(EIGEN3_FOUND)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
# INCLUDE_DIRECTORIES(${OpenCV_INC})

find_package(PCL REQUIRED)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
message(STATUS "pcl incldue dir : ${PCL_INCLUDE_DIRS}")

#set(Ceres_DIR /usr/local/ceres-1.14/lib/cmake/Ceres)
#find_package(Ceres REQUIRED )
# FIND_PACKAGE(Sophus REQUIRED)
set(Sophus_INCLUDE_DIRS /usr/local/Sophus/include)
set(Sophus_LIBRARIES /usr/local/Sophus/lib/libSophus.so)

set(vikit_common ${PROJECT_SOURCE_DIR}/Thirdparty/vikit_common/build)
find_package(vikit_common REQUIRED)

find_package(Pangolin REQUIRED)

message(STATUS "pcl dir: ${PCL_INCLUDE_DIRS}")

#set(g2o_DIR /usr/local/g2o_20200410/lib/cmake/g2o)
#find_package(g2o REQUIRED)
#set(G2O_INCLUDE_DIRS /usr/local/g2o_20200410/include)
#set(G2O_LIB_DIR /usr/local/g2o_20200410/lib) 
#SET(G2O_LIBRARIES g2o_cli g2o_core g2o_csparse_extension g2o_ext_freeglut_minimal 
#             g2o_incremental g2o_interactive g2o_interface g2o_opengl_helper 
#             g2o_parser g2o_simulator g2o_solver_cholmod g2o_solver_csparse 
#             g2o_solver_dense g2o_solver_eigen g2o_solver_pcg g2o_solver_slam2d_linear 
#             g2o_solver_structure_only g2o_stuff g2o_types_data g2o_types_icp 
#             g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d
#             g2o_types_slam2d_addons g2o_types_slam3d g2o_types_slam3d_addons g2o_viewer)
#IF(G2O_FOUND)
#    message("G2O lib is found:" ${G2O_INCLUDE_DIRS}) 
#    message("G2O lib:" ${G2O_LIBRARIES})
#ENDIF(G2O_FOUND)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
#  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
#  ${G2O_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  /home/kinggreat24/ORB_SLAM2
  /home/kinggreat24/ORB_SLAM2/include
  /home/kinggreat24/ORB_SLAM2/include/depth_clustering
  /home/kinggreat24/ORB_SLAM2/Thirdparty/fadeRelease_v1.82
  /home/kinggreat24/ORB_SLAM2/Thirdparty/vikit_common/include
  /home/kinggreat24/ORB_SLAM2/Thirdparty/monolidar_fusion/include
)

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
#  ${CERES_LIBRARY_DIRS}
#  ${G2O_LIB_DIR}
)


catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
#  INCLUDE_DIRS include
)

add_executable(hybrid_tracking_node src/hybrid_tracking_example.cc)
target_link_libraries(hybrid_tracking_node 
  ${EIGEN3_LIBRARIES} 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${vikit_common_LIBRARIES}
  /home/kinggreat24/ORB_SLAM2/Thirdparty/DBoW2/lib/libDBoW2.so
  /home/kinggreat24/ORB_SLAM2/Thirdparty/g2o/lib/libg2o.so
  /home/kinggreat24/ORB_SLAM2/Thirdparty/fadeRelease_v1.82/lib_ubuntu18.04_x86_64/libfade25d.so
  /home/kinggreat24/ORB_SLAM2/Thirdparty/monolidar_fusion/lib/libmonolidar_fusion.so
  /home/kinggreat24/ORB_SLAM2/lib/libORB_SLAM2.so
)

