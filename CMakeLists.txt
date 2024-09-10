set (SRC
	src/particle_filter.h
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
)

target_link_directories(
	${PROJECT_NAME} PRIVATE
	./3rdparty/opencv/x64/vc16/lib
)

target_link_libraries(
	${PROJECT_NAME} PRIVATE
	opencv_world490
)

set_target_properties( ${PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

