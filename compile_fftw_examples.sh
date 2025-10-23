#!/usr/bin/bash
#!/usr/bin/bash
################################################################################
# Config
################################################################################
if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	# Check if the spack environment exists
	SPACK_ENV=hpxfft_x86_epyc
	if spack env list | grep -q "${SPACK_ENV}"; then
	    echo "Found ${SPACK_ENV} environment, activating it."
	    module load gcc/14.2.0
	    spack env activate ${SPACK_ENV}
	fi
    elif [[ "$HOSTNAME" == "sven0"  ||  "$HOSTNAME" == "sven1" ]]; then
        echo "tbd."
    elif [[ $(uname -i) == "aarch64" ]]; then
        echo "tbd."
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
        # Check if the spack environment exists
        SPACK_ENV=hpxfft_x86_simcl
        if spack env list | grep -q "${SPACK_ENV}"; then
            echo "Found ${SPACK_ENV} environment, activating it."
            module load gcc/14.1.0
            spack env activate ${SPACK_ENV}
	    # FFTW paths
	    export FFTW_DIR="$(spack -i location fftw)/lib"
	    export PKG_CONFIG_PATH="$FFTW_DIR/pkgconfig":$PKG_CONFIG_PATH
        fi
    else
        echo "Hostname is $HOSTNAME — no action taken."
    fi
else
    echo "Spack command not found. Building example without Spack."
    # Assuming that Spack is not required on given system
fi
################################################################################
# Compilation
################################################################################
cd examples/fftw
BUILD_DIR=build
CMAKE_COMMAND=cmake
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release
make -j

mkdir -p $BUILD_DIR/result/runtimes
mkdir -p $BUILD_DIR/result/plans
