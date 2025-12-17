#!/usr/bin/bash
################################################################################
# Config
################################################################################
if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	module load gcc/14.2.0
	# Check if the spack environment exists
	SPACK_ENV=hpxfft_x86_epyc
	if spack env list | grep -q "${SPACK_ENV}"; then
	    echo "Found ${SPACK_ENV} environment, activating it."
	    spack env activate ${SPACK_ENV}
	fi
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
	module load gcc/14.1.0
	# Check if the spack environment exists
	SPACK_ENV=hpxfft_x86_simcl
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
cd test
BUILD_DIR=build
CMAKE_COMMAND=cmake
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && 
cd $BUILD_DIR
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DHPXFFT_DIR=../install/lib/cmake/HPXFFT
make -j
