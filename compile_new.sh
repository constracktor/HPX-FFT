spack load hpx
spack load fftw
spack load cmake
spack load gcc
################################################################################
# Compilation
################################################################################
BUILD_DIR=build
CMAKE_COMMAND=cmake
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
#$LDFLAGS=$LDFLAGS 
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release 

make VERBOSE=1 -j $(grep -c ^processor /proc/cpuinfo)
