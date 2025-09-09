module load gcc/14.2.0
spack load hpx%gcc@14.2.0
spack load fftw%gcc@14.2.0
################################################################################
# Compilation
################################################################################
cd examples/hpxfft
BUILD_DIR=build
CMAKE_COMMAND=cmake
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DHPXFFT_DIR=../../install/lib64/cmake/HPXFFT
make -j
