cmake_minimum_required(VERSION 3.21)

project(HPXFFTTests LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
  find_package(HPXFFT REQUIRED)
  enable_testing()
endif()

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0)

FetchContent_MakeAvailable(Catch2)

# ---- Tests ----
add_executable(test_naive src/test_naive.cpp)
target_link_libraries(
  test_naive
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_naive PRIVATE cxx_std_17)

add_test(
  NAME test_naive
  COMMAND test_naive
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_agas src/test_shared_agas.cpp)
target_link_libraries(
  test_shared_agas
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_agas PRIVATE cxx_std_17)

add_test(
  NAME test_shared_agas
  COMMAND test_shared_agas
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_loop src/test_shared_loop.cpp)
target_link_libraries(
  test_shared_loop
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_loop PRIVATE cxx_std_17)

add_test(
  NAME test_shared_loop
  COMMAND test_shared_loop
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_sync src/test_shared_sync.cpp)
target_link_libraries(
  test_shared_sync
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_sync PRIVATE cxx_std_17)

add_test(
  NAME test_shared_sync
  COMMAND test_shared_sync
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_opt src/test_shared_opt.cpp)
target_link_libraries(
  test_shared_opt
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_opt PRIVATE cxx_std_17)

add_test(
  NAME test_shared_opt
  COMMAND test_shared_opt
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_distributed_loop src/test_distributed_loop.cpp)
target_link_libraries(
  test_distributed_loop
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_distributed_loop PRIVATE cxx_std_17)

add_test(
  NAME test_distributed_loop
  COMMAND test_distributed_loop 
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_distributed_agas src/test_distributed_agas.cpp)
target_link_libraries(
  test_distributed_agas
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_distributed_agas PRIVATE cxx_std_17)

add_test(
  NAME test_distributed_agas
  COMMAND test_distributed_agas 
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_vector_2d src/test_vector_2d.cpp)
target_link_libraries(
  test_vector_2d
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_vector_2d PUBLIC cxx_std_20)

add_test(
  NAME test_vector_2d
  COMMAND test_vector_2d
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
