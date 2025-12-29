cmake_minimum_required(VERSION 3.10)
project(face_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode with optimizations")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
else()
    message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")
endif()

find_package(OpenCV REQUIRED)

# Try to find dlib
find_package(dlib QUIET)
if(dlib_FOUND)
    message(STATUS "dlib found: ${dlib_VERSION}")
else()
    # Try common installation paths
    find_path(DLIB_INCLUDE_DIR dlib/image_processing/frontal_face_detector.h
        PATHS
        /usr/local/include
        /opt/homebrew/include
        /usr/include
    )
    find_library(DLIB_LIBRARY
        NAMES dlib
        PATHS
        /usr/local/lib
        /opt/homebrew/lib
        /usr/lib
    )
    if(DLIB_INCLUDE_DIR AND DLIB_LIBRARY)
        set(dlib_FOUND TRUE)
        message(STATUS "dlib found manually: ${DLIB_INCLUDE_DIR}")
    else()
        message(WARNING "dlib not found. Install with: brew install dlib")
    endif()
endif()

add_executable(face_app
    main.cpp
)

add_executable(metrics
    metrics.cpp
)

add_executable(recognition
    recognition.cpp
)

target_include_directories(face_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_app PRIVATE ${OpenCV_LIBS})
if(dlib_FOUND)
    if(TARGET dlib::dlib)
        target_link_libraries(face_app PRIVATE dlib::dlib)
    else()
        target_include_directories(face_app PRIVATE ${DLIB_INCLUDE_DIR})
        target_link_libraries(face_app ${DLIB_LIBRARY})
    endif()
    target_compile_definitions(face_app PRIVATE HAVE_DLIB)
endif()

target_include_directories(metrics PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(metrics PRIVATE ${OpenCV_LIBS})
if(dlib_FOUND)
    if(TARGET dlib::dlib)
        target_link_libraries(metrics PRIVATE dlib::dlib)
    else()
        target_include_directories(metrics PRIVATE ${DLIB_INCLUDE_DIR})
        target_link_libraries(metrics ${DLIB_LIBRARY})
    endif()
    target_compile_definitions(metrics PRIVATE HAVE_DLIB)
endif()

target_include_directories(recognition PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(recognition PRIVATE ${OpenCV_LIBS})
