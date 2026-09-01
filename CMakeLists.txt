cmake_minimum_required(VERSION 3.22)
project(Turret LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Pin to the /usr/local source build, not the JetPack system OpenCV.
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer REQUIRED)

# --- libraries ---
add_library(preprocess src/preprocess/Preprocessor.cpp)
target_link_libraries(preprocess PUBLIC ${OpenCV_LIBS})
target_include_directories(preprocess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(detect src/detect/Detector.cpp)
target_link_libraries(detect PUBLIC preprocess PRIVATE ${NVINFER_LIB} CUDA::cudart)

add_library(aim src/aim/Aimer.cpp)
target_link_libraries(aim PUBLIC opencv_core opencv_calib3d)
target_include_directories(aim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- apps ---
add_executable(livecam apps/LiveCam.cpp)
target_link_libraries(livecam PRIVATE detect aim track link)

# --- tools ---
add_executable(preprocess_check tools/preprocess_check.cpp)
target_link_libraries(preprocess_check PRIVATE preprocess)

add_executable(det_check tools/det_check.cpp)
target_link_libraries(det_check PRIVATE detect)

add_executable(calibrate tools/calibrate.cpp)
target_link_libraries(calibrate PRIVATE ${OpenCV_LIBS})

add_executable(aim_check tools/aim_check.cpp)
target_link_libraries(aim_check PRIVATE aim)

add_library(track src/track/Tracker.cpp)
target_include_directories(track PUBLIC ${CMAKE_SOURCE_DIR}/src/track)
target_link_libraries(track PUBLIC opencv_core)

add_executable(kalman_test tools/kalman_test.cpp)
target_link_libraries(kalman_test PRIVATE track)

add_executable(capture_dataset tools/capture_dataset.cpp)
target_link_libraries(capture_dataset ${OpenCV_LIBS})
target_compile_definitions(capture_dataset PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(serial_check tools/serial_check.cpp)

add_library(link src/link/TurretLink.cpp)
target_include_directories(link PUBLIC src)
