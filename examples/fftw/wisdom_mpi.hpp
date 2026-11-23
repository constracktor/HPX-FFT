#pragma once
// FFTW wisdom for the MPI examples: rank 0 reads the file and broadcasts it, all ranks gather it before rank 0 writes.
// Both functions are collective.
#include "fftw3-mpi.h"
#include "wisdom.hpp"
#include <iostream>
#include <string>

namespace fftw_wisdom
{
inline void import_mpi(const std::string &path, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0 && !import_file(path))
    {
        std::cout << "No FFTW wisdom imported from " << path << "\n";
    }
    fftw_mpi_broadcast_wisdom(comm);
}

inline void export_mpi(const std::string &path, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    fftw_mpi_gather_wisdom(comm);
    if (rank == 0)
    {
        export_file(path);
    }
}
}  // namespace fftw_wisdom
