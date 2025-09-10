#module load gcc/14.2.0
spack load gcc@14.2.0
spack load fftw%gcc@14.2.0
################################################################################
# Compilation
################################################################################
# FFTW paths
export FFTW_DIR="$(spack -i location fftw)/lib"
export PKG_CONFIG_PATH="$FFTW_DIR/pkgconfig":$PKG_CONFIG_PATH

cd examples/fftw
BUILD_DIR=build
CMAKE_COMMAND=cmake
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release
make -j

