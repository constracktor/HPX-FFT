#define FFTW
#define FFTW_FLAG_ESTIMATE
#include "omp.h"
#include "p3dfft.h"
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

typedef double real_t;
typedef std::chrono::duration<real_t> duration;
using namespace p3dfft;

int main(int argc, char *argv[])
{
    // nodes ranks    prog          threads N_X  N_Y  N_Z   ndim  header
    // srun   -N 2  -n 4 p3dfft_mpi_omp      4      8   14   20    2     0
    // mpirun       -n 4 p3dfft_mpi_omp      4      8   14   20    2     0
    ////////////////////////////////////////////////////////////////
    // Parameters
    int n_threads = std::stoi(argv[1]);
    const std::uint32_t dim_r_x = std::stoi(argv[2]);  // N_X
    const std::uint32_t dim_r_y = std::stoi(argv[3]);  // N_Y
    const std::uint32_t dim_r_z = std::stoi(argv[4]);  // N_Z
    const std::uint32_t dim_c_x = dim_r_x / 2 + 1;     // R2C halves the 1st (X) dimension
    int ndim = std::stoi(argv[5]);                     // 1 or 2 dimensional proc grid
    bool print_header = std::stoi(argv[6]);

    ////////////////////////////////////////////////////////////////
    // Time measurement
    auto t = std::chrono::steady_clock();
    std::map<std::string, real_t> runtimes;

    ////////////////////////////////////////////////////////////////
    // MPI setup
    int provided;
    int rank, n_ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    ////////////////////////////////////////////////////////////////
    // Threading setup (P3DFFT++ uses FFTW underneath)
    omp_set_num_threads(n_threads);

    ////////////////////////////////////////////////////////////////
    // P3DFFT++ setup
    setup();

    // Scope all P3DFFT objects (ProcGrid/DataGrid/transform3D) so their
    // destructors, which call MPI_Comm_free, run before MPI_Finalize().
    {
        // Forward (real-to-complex) transform type
        int type_ids_f[3] = { R2CFFT_D, CFFT_FORWARD_D, CFFT_FORWARD_D };
        trans_type3D type_rcc(type_ids_f);

        // Processor grid geometry (p1 x p2)
        int pdims[2] = { 0, 0 };
        if (ndim == 1)
        {
            pdims[0] = 1;
            pdims[1] = n_ranks;
        }
        else
        {
            MPI_Dims_create(n_ranks, 2, pdims);
            if (pdims[0] > pdims[1])
            {
                pdims[0] = pdims[1];
                pdims[1] = n_ranks / pdims[0];
            }
        }
        int p1 = pdims[0];
        int p2 = pdims[1];

        // Global grid dimensions (physical and Fourier space).
        // For R2C the conjugate-symmetry (1st) dimension is halved.
        int gdims[3] = { (int) dim_r_x, (int) dim_r_y, (int) dim_r_z };
        int gdims2[3] = { (int) dim_c_x, (int) dim_r_y, (int) dim_r_z };

        // Memory ordering: input X-pencil keeps natural order, output Z-pencil is rotated
        int mem_order[3] = { 0, 1, 2 };
        int mem_order2[3] = { 1, 2, 0 };

        // Data-to-processor mapping for each grid
        int dmap1[3] = { 0, 1, 2 };  // X-pencil (1st dimension local)
        int dmap2[3] = { 1, 2, 0 };  // Z-pencil (3rd dimension local)

        // Processor grid must be a 3-element array: X-pencil (1st dim local, 2nd/3rd split)
        int pgrid_dims[3] = { 1, p1, p2 };
        ProcGrid pgrid(pgrid_dims, comm);
        DataGrid grid1(gdims, -1, &pgrid, dmap1, mem_order);
        DataGrid grid2(gdims2, 0, &pgrid, dmap2, mem_order2);

        // Local sizes in storage order
        int sdims1[3], glob_start1[3];
        int sdims2[3], glob_start2[3];
        for (int i = 0; i < 3; ++i)
        {
            sdims1[mem_order[i]] = grid1.Ldims[i];
            glob_start1[mem_order[i]] = grid1.GlobStart[i];
            sdims2[mem_order2[i]] = grid2.Ldims[i];
            glob_start2[mem_order2[i]] = grid2.GlobStart[i];
        }
        long int size_in = (long int) sdims1[0] * sdims1[1] * sdims1[2];
        long int size_out = (long int) sdims2[0] * sdims2[1] * sdims2[2];

        std::vector<double> input(size_in);
        std::vector<complex_double> output(size_out);

        ////////////////////////////////////////////////////////////////
        // Plan creation
        MPI_Barrier(comm);
        auto start_plan_r2c = t.now();
        transform3D<double, complex_double> trans_f(grid1, grid2, &type_rcc);
        MPI_Barrier(comm);
        auto stop_plan_r2c = t.now();
        runtimes["plan_p3dfft_r2c"] = duration(stop_plan_r2c - start_plan_r2c).count();

        ////////////////////////////////////////////////////////////////
        // Initialization (row-wise from 0 in X, contiguous)
        for (int k = 0; k < sdims1[2]; ++k)
        {
            for (int j = 0; j < sdims1[1]; ++j)
            {
                for (int i = 0; i < sdims1[0]; ++i)
                {
                    input[(long int) k * sdims1[1] * sdims1[0] + (long int) j * sdims1[0] + i] = i;
                }
            }
        }

        ////////////////////////////////////////////////////////////////
        // Computation
        MPI_Barrier(comm);
        auto start_r2c = t.now();
        trans_f.exec(input.data(), output.data(), false);
        MPI_Barrier(comm);
        auto stop_r2c = t.now();
        runtimes["p3dfft_r2c"] = duration(stop_r2c - start_r2c).count();
        runtimes["init_p3dfft_r2c"] = duration(start_r2c - stop_plan_r2c).count();

        ////////////////////////////////////////////////////////////////
        // Postprocessing
        if (rank == 0)
        {
            std::cout << "P3DFFT++ 3D R2C with MPI + OpenMP:" << "\n MPI ranks      = " << n_ranks
                      << "\n OpenMP threads = " << n_threads << "\n proc grid      = " << p1 << " x " << p2
                      << "\n plan_r2c       = " << runtimes["plan_p3dfft_r2c"] << "\n init_r2c       = "
                      << runtimes["init_p3dfft_r2c"] << "\n p3dfft_3d_r2c  = " << runtimes["p3dfft_r2c"] << std::endl;

            std::ofstream runtime_file;
            runtime_file.open("runtimes/runtimes_p3dfft_mpi_omp_3d.txt", std::ios_base::app);
            if (print_header)
            {
                runtime_file << "n_ranks;n_threads;n_x;n_y;n_z;ndim;planning;initialization,p3dfft_3d_r2c;\n";
            }
            runtime_file << n_ranks << ";" << n_threads << ";" << dim_r_x << ";" << dim_r_y << ";" << dim_r_z << ";"
                         << ndim << ";" << runtimes["plan_p3dfft_r2c"] << ";" << runtimes["init_p3dfft_r2c"] << ";"
                         << runtimes["p3dfft_r2c"] << ";\n";
            runtime_file.close();
        }
    }  // end P3DFFT object scope

    ////////////////////////////////////////////////////////////////
    // Cleanup
    cleanup();
    MPI_Finalize();

    return 0;
}
