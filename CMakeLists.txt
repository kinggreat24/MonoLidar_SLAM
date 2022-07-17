cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

# SET(CMAKE_BUILD_TYPE Debug)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

set(PCL_DIR /usr/local/pcl_1.7/share/pcl-1.7)
find_package(PCL REQUIRED)
add_definitions(-DPCL_FOUND)
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP REQUIRED)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# FIND_PACKAGE(Sophus REQUIRED)
set(Sophus_INCLUDE_DIRS /usr/local/Sophus/include)
set(Sophus_LIBRARIES /usr/local/Sophus/lib/libSophus.so)

##VIKIT_COMMON
set(vikit_common ${PROJECT_SOURCE_DIR}/Thirdparty/vikit_common/build)
find_package(vikit_common REQUIRED)

# GFLAG (neccessary)
find_package(GFLAGS REQUIRED QUIET)
if(GFLAGS_FOUND)
include_directories(${GFLAGS_INCLUDE_DIRS})
message("GFLAGS [OK]:" ${GFLAGS_INCLUDE_DIRS}) 
endif(GFLAGS_FOUND)

#GLOG (neccessary)
find_package(GLOG REQUIRED QUIET)
if(GLOG_FOUND)
include_directories(${GLOG_INCLUDE_DIRS})
message("GLOG [OK]:" ${GLOG_INCLUDE_DIRS}) 
endif(GLOG_FOUND)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/mulls_feature_extraction
${PROJECT_SOURCE_DIR}/include/baseline_reg
${PROJECT_SOURCE_DIR}/include/depth_clustering
${PROJECT_SOURCE_DIR}/Thirdparty/fadeRelease_v1.82
${PROJECT_SOURCE_DIR}/Thirdparty/monolidar_fusion/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
${vikit_common_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc

# lidar depth extraction
src/LidarFeatureExtraction.cc
src/LidarDepthExtraction.cc

#lidar projections
src/depth_clustering/utils/bbox.cc
src/depth_clustering/utils/cloud.cc
src/depth_clustering/utils/folder_reader.cc
src/depth_clustering/utils/rich_point.cc
src/depth_clustering/utils/velodyne_utils.cc
src/depth_clustering/projections/projection_params.cc
src/depth_clustering/projections/cloud_projection.cc
src/depth_clustering/projections/ring_projection.cc
src/depth_clustering/projections/spherical_projection.cc
src/depth_clustering/ground_removal/depth_ground_remover.cpp
src/depth_clustering/image_labelers/abstract_image_labeler.cpp
src/depth_clustering/image_labelers/diff_helpers/angle_diff.cpp
src/depth_clustering/image_labelers/diff_helpers/line_dist_diff.cpp


src/lidar_sparse_align/SparseLidarAlign.cc
src/lidar_sparse_align/HybridSparseAlign.cc
src/lidar_sparse_align/SparseReprojAlign.cc
src/lidar_sparse_align/WeightFunction.cpp
)

add_definitions(-DUSE_LIDAR)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PCL_LIBRARIES}
${Pangolin_LIBRARIES}
${GLOG_LIBRARIES}
${GFLAGS_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/fadeRelease_v1.82/lib_ubuntu18.04_x86_64/libfade25d.so
${PROJECT_SOURCE_DIR}/Thirdparty/monolidar_fusion/lib/libmonolidar_fusion.so
${Sophus_LIBRARIES}
${vikit_common_LIBRARIES}
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_se2vlam
Examples/Monocular/mono_se2vlam.cc)
target_link_libraries(mono_se2vlam ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Lidar)
add_executable(monolidar_tum
Examples/Monocular-Lidar/monocular_lidar_tum.cc)
target_link_libraries(monolidar_tum ${PROJECT_NAME})

add_executable(monolidar_kitti
Examples/Monocular-Lidar/monocular_lidar_kitti.cc)
target_link_libraries(monolidar_kitti ${PROJECT_NAME})
