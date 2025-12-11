cmake_minimum_required(VERSION 3.23)

project(
  HPXFFT
  VERSION 0.1.0
  DESCRIPTION "Distributed 2D FFT using HPX"
  HOMEPAGE_URL "https://github.com/constracktor/HPX-FFT"
  LANGUAGES CXX)

option(HPXFFT_ENABLE_FORMAT_TARGETS
       "Enable clang-format / cmake-format targets" ${PROJECT_IS_TOP_LEVEL})

option(HPXFFT_BUILD_CORE "Build the core library" ON)
option(HPXFFT_ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

if(HPXFFT_ENABLE_FORMAT_TARGETS)
  find_package(format QUIET)
  if(NOT format_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      format
      GIT_REPOSITORY https://github.com/TheLartians/Format.cmake.git
      GIT_TAG v1.8.1
      QUIET)
    FetchContent_MakeAvailable(format)
  endif()
endif()
if(HPXFFT_BUILD_CORE)
  find_package(HPX REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
  if(NOT CMAKE_SKIP_INSTALL_RULES)
    # Our installs follow the standard GNU directory layout. This include needs
    # to come first since we need the CMAKE_INSTALL_* in the CMakeLists.txt of
    # each target.
    include(GNUInstallDirs)
  endif()

  add_subdirectory(core)

  if(NOT CMAKE_SKIP_INSTALL_RULES)
    include(CMakePackageConfigHelpers)
    # find_package(<package>) call for consumers to find this project
    set(package HPXFFT)

    write_basic_package_version_file("${package}ConfigVersion.cmake"
                                     COMPATIBILITY SameMajorVersion)
    # Allow package maintainers to freely override the path for the configs
    set(HPXFFT_INSTALL_CMAKEDIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
        CACHE STRING
              "CMake package config location relative to the install prefix")
    set_property(CACHE HPXFFT_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
    mark_as_advanced(HPXFFT_INSTALL_CMAKEDIR)

    install(
      FILES cmake/install-config.cmake
      DESTINATION "${HPXFFT_INSTALL_CMAKEDIR}"
      RENAME "${package}Config.cmake"
      COMPONENT Development)

    install(
      FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
      DESTINATION "${HPXFFT_INSTALL_CMAKEDIR}"
      COMPONENT Development)

    install(
      EXPORT HPXFFTTargets
      NAMESPACE HPXFFT::
      DESTINATION "${HPXFFT_INSTALL_CMAKEDIR}"
      COMPONENT Development)

    if(PROJECT_IS_TOP_LEVEL)
      include(CPack)
    endif()
  endif()
endif()
