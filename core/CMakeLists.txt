set(SOURCE_FILES
    src/2D/shared/loop.cpp
    src/2D/shared/sync.cpp
    src/2D/shared/opt.cpp
    src/2D/shared/naive.cpp
    src/2D/shared/agas.cpp
    src/2D/distributed/loop.cpp
    src/2D/distributed/agas.cpp
    src/3D/shared/loop.cpp
    src/3D/shared/naive.cpp
    src/3D/shared/sync.cpp
    src/util/adapter_fftw.cpp
    src/util/create_dir.cpp
    src/3D/distributed/loop_slab.cpp
    src/3D/distributed/loop_pencil.cpp)

add_library(hpxfft STATIC ${SOURCE_FILES})

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/hpxfft/shared/*.hpp
     include/hpxfft/distributed/*.hpp)
target_sources(hpxfft PRIVATE ${header_files})

# Link HPX
target_link_libraries(hpxfft PUBLIC HPX::hpx)
# Link FFTW backend
target_link_libraries(hpxfft PUBLIC PkgConfig::FFTW)

# Include directories
target_include_directories(
  hpxfft PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

target_compile_features(hpxfft PUBLIC cxx_std_20)

# Turn on PIC for shared libraries
set_property(TARGET hpxfft PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # We need to manually install those into CMAKE_INSTALL_INCLUDEDIR. Below
  # install(TARGETS ...) only setups the paths for the exported targets.
  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/core/include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Development)

  install(
    TARGETS hpxfft
    EXPORT HPXFFTTargets
    RUNTIME COMPONENT Runtime
    LIBRARY COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE COMPONENT Development
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
