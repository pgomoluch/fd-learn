cmake_minimum_required(VERSION 2.8.12)

project(rsvm_heuristic)

include(../../../dlib/dlib/cmake)

include_directories(../servers)

add_executable(rsvm learner.cc)
target_link_libraries(rsvm dlib::dlib)

add_executable(server server.cc ../servers/heuristic_server.cc)
target_link_libraries(server dlib::dlib)
