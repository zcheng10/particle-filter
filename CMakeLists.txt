cmake_minimum_required(VERSION 3.27)

set (SRC
	src/particle_filter.cc
	src/main.cpp
)

project(
	particle_filter
	# LANGUAGE CXX
)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories (
	${PROJECT_NAME} PRIVATE
	./3rdparty/opencv/include
	./3rdparty/spdlog/include
	./
)

target_link_directories(
	${PROJECT_NAME} PRIVATE
	./3rdparty/opencv/x64/vc16/lib
	./3rdparty/spdlog/lib
)

target_link_libraries(
	${PROJECT_NAME} PRIVATE
	opencv_world490
	spdlog
)

set_target_properties( ${PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

