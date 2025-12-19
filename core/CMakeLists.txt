set(SOURCE_FILES
    src/shared/loop.cpp
    src/shared/sync.cpp
    src/shared/opt.cpp
    src/shared/naive.cpp
    src/shared/agas.cpp
    src/distributed/loop.cpp
    src/distributed/agas.cpp
    src/util/adapter_fftw.cpp)

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
