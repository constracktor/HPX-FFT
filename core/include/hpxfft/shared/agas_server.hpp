#pragma once
#ifndef hpxfft_shared_agas_server_H_INCLUDED
#define hpxfft_shared_agas_server_H_INCLUDED

#include "../util/adapter_fftw.hpp"
#include "../util/vector_2d.hpp"  // for hpxfft::util::vector_2d
#include <hpx/future.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::shared
{
using vector_2d = hpxfft::util::vector_2d<real>;

struct agas_server : hpx::components::component_base<agas_server>
{
    typedef std::vector<hpx::future<void>> vector_future;

  public:
    agas_server() = default;

    void initialize(vector_2d values_vec, const std::string PLAN_FLAG);

    vector_2d fft_2d_r2c();

  private:
    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, fft_1d_r2c_inplace, fft_1d_r2c_inplace_action)
    void fft_1d_c2c_inplace(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, fft_1d_c2c_inplace, fft_1d_c2c_inplace_action)

    // transpose
    void transpose_shared_y_to_x(const std::size_t index);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, transpose_shared_y_to_x, transpose_shared_y_to_x_action)
    void transpose_shared_x_to_y(const std::size_t index_trans);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, transpose_shared_x_to_y, transpose_shared_x_to_y_action)

  private:
    // parameters
    std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
    // FFTW plans
    hpxfft::util::fftw_plan_flag PLAN_FLAG_;
    // IMPORTANT: declare r2c adapter before c2c so r2c destructor is called after c2c
    hpxfft::util::fftw_adapter_r2c fftw_r2c_adapter_;
    hpxfft::util::fftw_adapter_c2c fftw_c2c_adapter_;
    // value vectors
    vector_2d values_vec_;
    vector_2d trans_values_vec_;
    // future vectors
    vector_future r2c_futures_;
    vector_future trans_y_to_x_futures_;
    vector_future c2c_futures_;
    vector_future trans_x_to_y_futures_;
};

}  // namespace hpxfft::shared

HPX_DEFINE_COMPONENT_ACTION(hpxfft::shared::agas_server, fft_2d_r2c, fft_2d_r2c_action)

HPX_DEFINE_COMPONENT_ACTION(hpxfft::shared::agas_server, initialize, initialize_action)

#endif  // hpxfft_shared_agas_server_H_INCLUDED
