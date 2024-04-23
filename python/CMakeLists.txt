cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本号
project(litesdk VERSION 1.0)

# 添加可执行文件并指定源文件
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(litesdk ${SOURCES})

# 可选：指定C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_PYTHON_API "build python api"                                          ON)


set(OPENCV_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty/third_party/opencv")
set(MNN_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty/third_party/MNN")

#pybind11
set(Pybind11_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/third_party/pybind11)
if(BUILD_PYTHON_API)
  # Python3_FOUND
  # Python3_EXECUTABLE
  # Python3_INCLUDE_DIRS
  # Python3_LINK_OPTIONS
  # Python3_LIBRARIES
  # Python3_LIBRARY_DIRS
  # Python3_VERSION
  # Python3_VERSION_MAJOR
  # Python3_VERSION_MINOR
  # Python3_VERSION_PATCH
  # Python3_NumPy_FOUND
  # Python3_NumPy_INCLUDE_DIRS
  # Python3_NumPy_VERSION

  find_package (Python3 COMPONENTS Interpreter Development)
  message("### Python version: ${Python3_VERSION}")
  message("### Python path:${Python3_EXECUTABLE}")
  include_directories(${Python3_INCLUDE_DIRS})
  link_directories(${Python3_LIBRARY_DIRS})

  add_definitions(-D_PYTHON_API)
  include_directories(${Pybind11_DIR}/include)
endif()

if(BUILD_PYTHON_API)
    add_subdirectory(${Pybind11_DIR})
    pybind11_add_module(pylitesdk ${CMAKE_CURRENT_LIST_DIR}/litesdk_pybind11.cxx)
endif()

if(BUILD_PYTHON_API)
target_link_libraries(pylitesdk PRIVATE litesdk)
endif()

# config opencv
set(OpenCV_DIR ${OPENCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV CONFIG REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found: (${OpenCV_CONFIG_PATH})")
  message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
  add_definitions(-DUSE_OPENCV -DOPENCV_VERSION=${OpenCV_VERSION_MAJOR})
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND OPENCV_LIBRARIES ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found")
endif()


set_target_properties(litesdk PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(DLLEXPORT "__declspec(dllexport)")

if (MSVC)
  add_compile_definitions(DLLEXPORT=${DLLEXPORT})
else()
  add_compile_definitions(DLLEXPORT=)
endif()

include_directories(
    ${MNN_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include
)

target_link_libraries(litesdk
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib/liblite.ai.toolkit.so
    ${MNN_DIR}/lib/libMNN.so
    ${OPENCV_LIBRARIES}
)
