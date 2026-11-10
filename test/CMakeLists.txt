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

add_executable(test_vector_3d src/test_vector_3d.cpp)
target_link_libraries(
  test_vector_3d
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_vector_3d PUBLIC cxx_std_20)

add_test(
  NAME test_vector_3d
  COMMAND test_vector_3d
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_loop_3d src/test_shared_loop_3d.cpp)
target_link_libraries(
  test_shared_loop_3d
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_loop_3d PRIVATE cxx_std_17)

add_test(
  NAME test_shared_loop_3d
  COMMAND test_shared_loop_3d
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_naive_3d src/test_shared_naive_3d.cpp)
target_link_libraries(
  test_shared_naive_3d
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_naive_3d PRIVATE cxx_std_17)

add_test(
  NAME test_shared_naive_3d
  COMMAND test_shared_naive_3d
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(test_shared_sync_3d src/test_shared_sync_3d.cpp)
target_link_libraries(
  test_shared_sync_3d
  PRIVATE HPXFFT::hpxfft Catch2::Catch2WithMain
  PUBLIC HPX::hpx)
target_compile_features(test_shared_sync_3d PRIVATE cxx_std_17)

add_test(
  NAME test_shared_sync_3d
  COMMAND test_shared_sync_3d
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

# ---- Distributed correctness tests: several localities per test, naive DFT reference ----
add_executable(test_distributed_fft src/test_distributed_fft.cpp)
target_link_libraries(
  test_distributed_fft
  PRIVATE HPXFFT::hpxfft
  PUBLIC HPX::hpx)
target_compile_features(test_distributed_fft PRIVATE cxx_std_17)

# name suffix, localities, then the arguments of the test program
function(add_distributed_test name localities)
  add_test(
    NAME ${name}
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/run_localities.sh ${localities}
            $<TARGET_FILE:test_distributed_fft> ${ARGN})
  set_tests_properties(${name} PROPERTIES TIMEOUT 200)
endfunction()

# a case is "nx,ny,nz"
function(add_distributed_cases mode localities cases flags)
  foreach(case ${cases})
    string(REPLACE "," ";" dims ${case})
    list(GET dims 0 nx)
    list(GET dims 1 ny)
    list(GET dims 2 nz)
    foreach(flag ${flags})
      foreach(tr 0 1)
        add_distributed_test(
          dist_${mode}_L${localities}_${nx}x${ny}x${nz}_${flag}_t${tr} ${localities}
          --mode=${mode} --nx=${nx} --ny=${ny} --nz=${nz} --run=${flag} --transposed=${tr})
      endforeach()
    endforeach()
  endforeach()
endfunction()

set(DIST_FLAGS scatter all_to_all scatter_hierarchical)
foreach(l 1 2 3 4)
  add_distributed_cases(slab ${l} "12,12,10;12,12,12;12,12,9;12,13,10;12,7,9" "${DIST_FLAGS}")
endforeach()
add_distributed_cases(pencil 4 "8,8,6;8,8,10;8,8,11;8,8,8;8,8,12" "${DIST_FLAGS}")
add_distributed_cases(pencil 9 "12,12,10;12,12,8" "${DIST_FLAGS}")
add_distributed_cases(pencil 16 "16,16,12;16,16,11" "${DIST_FLAGS}")
# sizes beyond one 16x16 transpose tile, with partial tiles and uneven blocks
add_distributed_cases(slab 4 "40,37,35;36,40,34" "${DIST_FLAGS}")
add_distributed_cases(pencil 4 "40,36,35" "${DIST_FLAGS}")
# 2D: the case is "nx,ny,0"; ny / 2 + 1 (5, 5 and 6 here) need not be divisible by the number of localities
foreach(l 1 2 4)
  add_distributed_cases(2d ${l} "8,6,0;8,14,0;8,8,0;8,9,0;8,11,0" "${DIST_FLAGS}")
endforeach()
add_distributed_cases(2d 2 "8,10,0" "${DIST_FLAGS}")
add_distributed_cases(2d 4 "40,37,0;40,34,0" "${DIST_FLAGS}")
add_distributed_cases(2d 3 "12,9,0" "${DIST_FLAGS}")
add_distributed_cases(2d 4 "8,9,0;40,37,0" "all_to_all_routed")

# remaining run flags of the 3D loops
add_distributed_cases(slab 3 "12,13,10;12,7,9" "scatter_async;all_to_all_routed")
add_distributed_cases(pencil 4 "8,8,11;8,8,6" "scatter_async;all_to_all_routed")
add_distributed_cases(pencil 9 "12,12,10" "scatter_async;all_to_all_routed")

# sizes that must be rejected on every locality
add_distributed_test(dist_reject_slab_nz_mismatch 2 --mode=slab --nx=8 --ny=8 --nz=6 --size_arg=8 --expect_throw=1)
add_distributed_test(dist_reject_pencil_nz_mismatch 4 --mode=pencil --nx=8 --ny=8 --nz=6 --size_arg=8 --expect_throw=1)
add_distributed_test(dist_reject_2d_ny_mismatch 2 --mode=2d --nx=8 --ny=6 --size_arg=9 --expect_throw=1)
add_distributed_test(dist_reject_pencil_not_square 2 --mode=pencil --nx=8 --ny=8 --nz=6 --expect_throw=1)
add_distributed_test(dist_reject_pencil_small_z 4 --mode=pencil --nx=8 --ny=8 --nz=1 --expect_throw=1)
add_distributed_test(dist_reject_slab_y 4 --mode=slab --nx=8 --ny=3 --nz=6 --expect_throw=1)
add_distributed_test(dist_reject_2d_small_y 4 --mode=2d --nx=8 --ny=2 --nz=0 --expect_throw=1)
add_distributed_test(dist_reject_slab_unequal_x 2 --mode=slab --nx=8 --ny=8 --nz=6 --uneven_x=1 --expect_throw=1)
add_distributed_test(dist_reject_2d_unequal_x 2 --mode=2d --nx=8 --ny=6 --uneven_x=1 --expect_throw=1)

# shared memory loop, one locality, including odd nz
foreach(case "6,5,8" "6,5,9" "4,7,7" "5,6,10")
  string(REPLACE "," ";" dims ${case})
  list(GET dims 0 nx)
  list(GET dims 1 ny)
  list(GET dims 2 nz)
  foreach(variant par seq naive sync)
    add_distributed_test(dist_shared_${variant}_${nx}x${ny}x${nz} 1 --mode=shared --run=${variant} --nx=${nx} --ny=${ny}
                         --nz=${nz})
  endforeach()
endforeach()
foreach(case "6,8" "5,9" "7,10" "20,37")
  string(REPLACE "," ";" dims ${case})
  list(GET dims 0 nx)
  list(GET dims 1 ny)
  foreach(variant par seq naive sync opt)
    add_distributed_test(dist_shared2d_${variant}_${nx}x${ny} 1 --mode=shared2d --run=${variant} --nx=${nx} --ny=${ny})
  endforeach()
endforeach()
