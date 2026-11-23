# Same as in the root CMakeLists.txt
find_package(HPX 2.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

include("${CMAKE_CURRENT_LIST_DIR}/HPXFFTTargets.cmake")
