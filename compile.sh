#module load gcc/14.2.0
spack load gcc@14.2.0
spack load hpx%gcc@14.2.0
spack load fftw%gcc@14.2.0
################################################################################
# Compilation
################################################################################
BUILD_DIR=build
INSTALL_DIR=$(pwd)/install
CMAKE_COMMAND=cmake
rm -rf $INSTALL_DIR
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
make -j
make install 
