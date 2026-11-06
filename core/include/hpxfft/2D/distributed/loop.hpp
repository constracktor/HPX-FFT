#pragma once
#ifndef hpxfft_distributed_loop_H_INCLUDED
#define hpxfft_distributed_loop_H_INCLUDED

#include "../../util/adapter_fftw.hpp"
#include "../../util/vector_2d.hpp"  // for hpxfft::util::vector_2d
#include <hpx/future.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::fft2D::distributed
{
using vector_2d = hpxfft::util::vector_2d<real>;

struct loop
{
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<std::vector<real>> vector_comm;

  public:
    loop() = default;

    void initialize(vector_2d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG);

    vector_2d fft_2d_r2c();

    // Skip the final transpose: the result stays in the layout (y_local, x)
    // with y distributed over the localities, like FFTW_MPI_TRANSPOSED_OUT.
    // Saves the second communication phase.
    void set_transposed_output(bool transposed_out) { transposed_out_ = transposed_out; }

    // Split every exchange into this many chunks of rows. Each chunk is sent as
    // soon as it is computed, so communication overlaps with the remaining
    // computation. 1 keeps the phase-by-phase algorithm. 0 (default) chooses
    // automatically: 8 chunks for all_to_all if the local data is large
    // (chunking only pays off then), otherwise 1.
    void set_num_chunks(std::size_t num_chunks) { num_chunks_ = num_chunks; }

    real get_measurement(std::string name);

    ~loop() { hpxfft::util::fftw_adapter::cleanup(); }

  private:
    vector_2d fft_2d_r2c_pipelined(std::size_t num_chunks);
    std::size_t effective_num_chunks() const;

    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i);
    void fft_1d_c2c_inplace(const std::size_t i);

    // split data for communication
    void split_vec(const std::size_t i);
    void split_trans_vec(const std::size_t i);

    // scatter communication
    void communicate_scatter_vec(const std::size_t i);
    void communicate_scatter_trans_vec(const std::size_t i);

    // all to all communication
    void communicate_all_to_all_vec();
    void communicate_all_to_all_trans_vec();

    // blocked transpose after communication of the tile-th block of rows of the output (see loop.cpp)
    void transpose_y_to_x(const std::size_t tile, const std::size_t i);
    void transpose_x_to_y(const std::size_t tile, const std::size_t i);

  private:
    // parameters
    std::size_t n_x_local_, n_y_local_;
    std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
    std::size_t dim_c_y_part_, dim_c_x_part_;
    // 1D adapters
    hpxfft::util::fftw_adapter::r2c_1d fft_r2c_adapter_;
    hpxfft::util::fftw_adapter::c2c_1d fft_c2c_adapter_;
    // value vectors
    vector_2d values_vec_;
    vector_2d trans_values_vec_;
    // future vectors
    std::vector<hpx::future<std::vector<real>>> communication_futures_;
    // time measurement
    hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
    std::map<std::string, real> measurements_;
    // communication vectors
    vector_comm values_prep_;
    vector_comm trans_values_prep_;
    vector_comm communication_vec_;
    // locality information
    std::size_t this_locality_, num_localities_;
    bool transposed_out_ = false;
    std::size_t num_chunks_ = 0;
    // communicators
    std::string COMM_FLAG_;
    std::vector<std::string> basename_storage_;
    std::vector<const char *> basenames_;
    std::vector<hpx::collectives::communicator> communicators_;
};
}  // namespace hpxfft::fft2D::distributed
#endif  // hpxfft_distributed_loop_H_INCLUDED
