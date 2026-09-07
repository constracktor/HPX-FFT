#!/usr/bin/bash
#SBATCH --output=output/output_compile_hpxfft_examples_%j.log      # Standard output log
#SBATCH --error=output/error_compile_hpxfft_examples_%j.log        # Error log file
################################################################################
# Config
################################################################################
if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	module load gcc/14.2.0
	LIB_DIR=lib
	# Check if the spack environment exists
	SPACK_ENV=hpxfft_x86_epyc
	if spack env list | grep -q "${SPACK_ENV}"; then
	    echo "Found ${SPACK_ENV} environment, activating it."
	    spack env activate ${SPACK_ENV}
	fi
    elif [[ "$HOSTNAME" == "sven0"  ||  "$HOSTNAME" == "sven1" ]]; then
        echo "tbd."
    elif [[ $(uname -i) == "aarch64" ]]; then
	spack load gcc@14.2.0
	LIB_DIR=lib64
	# Check if the spack environment exists
	SPACK_ENV=hpxfft_arm_ookami
	if spack env list | grep -q "${SPACK_ENV}"; then
	    echo "Found ${SPACK_ENV} environment, activating it."
	    spack env activate ${SPACK_ENV}
	fi
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
	module load gcc/14.1.0
	LIB_DIR=lib
        # Check if the spack environment exists
        SPACK_ENV=hpxfft_x86_simcl
        if spack env list | grep -q "${SPACK_ENV}"; then
            echo "Found ${SPACK_ENV} environment, activating it."
            spack env activate ${SPACK_ENV}
        fi
    elif [[ "$HOSTNAME" == buran0[0-9] || "$HOSTNAME" == buran1[0-5] ]]; then
	module load gcc/14.2.0
	LIB_DIR=lib64
	# Check if the spack environment exists
	SPACK_ENV=hpxfft_x86_buran
	if spack env list | grep -q "${SPACK_ENV}"; then
	    echo "Found ${SPACK_ENV} environment, activating it."
	    spack env activate ${SPACK_ENV}
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
cd examples/hpxfft
BUILD_DIR=build
CMAKE_COMMAND=cmake
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
$CMAKE_COMMAND .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DHPXFFT_DIR=../../install/$LIB_DIR/cmake/HPXFFT
make -j
