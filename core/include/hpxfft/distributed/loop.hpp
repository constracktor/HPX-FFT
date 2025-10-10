#pragma once
#ifndef hpxfft_distributed_loop_H_INCLUDED
#define hpxfft_distributed_loop_H_INCLUDED

#include "../util/vector_2d.hpp"  // for hpxfft::util::vector_2d
#include <fftw3.h>                // for fftw_plan, fftw_destroy_plan, fftw_cleanup, FFTW_FLAGS
#include <hpx/future.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::distributed
{
using vector_2d = hpxfft::util::vector_2d<real>;

struct loop
{
    typedef fftw_plan fft_backend_plan;
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<std::vector<real>> vector_comm;

  public:
    loop() = default;

    void initialize(vector_2d values_vec, const std::string COMM_FLAG, const unsigned PLAN_FLAG);

    vector_2d fft_2d_r2c();

    real get_measurement(std::string name);

    ~loop()
    {
        fftw_destroy_plan(plan_1d_r2c_);
        fftw_destroy_plan(plan_1d_c2c_);
        fftw_cleanup();
    }

  private:
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

    // transpose after communication
    void transpose_y_to_x(const std::size_t k, const std::size_t i);
    void transpose_x_to_y(const std::size_t j, const std::size_t i);

  private:
    // parameters
    std::size_t n_x_local_, n_y_local_;
    std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
    std::size_t dim_c_y_part_, dim_c_x_part_;
    // FFTW plans
    unsigned PLAN_FLAG_;
    fft_backend_plan plan_1d_r2c_;
    fft_backend_plan plan_1d_c2c_;
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
    // communicators
    std::string COMM_FLAG_;
    std::vector<const char *> basenames_;
    std::vector<hpx::collectives::communicator> communicators_;
};
}  // namespace hpxfft::distributed
#endif  // hpxfft_distributed_loop_H_INCLUDED
