#pragma once
#ifndef hpxfft_shared_opt_H_INCLUDED
#define hpxfft_shared_opt_H_INCLUDED

#include "../util/adapter_fftw.hpp"
#include "../util/vector_2d.hpp"  // for hpxfft::util::vector_2d
#include <hpx/future.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::shared
{
using vector_2d = hpxfft::util::vector_2d<real>;

struct opt
{
    typedef std::vector<hpx::future<void>> vector_future;

  public:
    opt() = default;

    void initialize(vector_2d values_vec, const std::string PLAN_FLAG);

    vector_2d fft_2d_r2c();

    real get_measurement(std::string name);

    ~opt() { hpxfft::util::fftw_adapter::cleanup(); }

  private:
    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i);
    void fft_1d_c2c_inplace(const std::size_t i);

    // transpose
    void transpose_shared_y_to_x(const std::size_t index);
    void transpose_shared_x_to_y(const std::size_t index_trans);

    // static wrappers
    static void fft_1d_r2c_inplace_wrapper(opt *th, const std::size_t i);
    static void fft_1d_c2c_inplace_wrapper(opt *th, const std::size_t i);
    static void transpose_shared_y_to_x_wrapper(opt *th, const std::size_t index);
    static void transpose_shared_x_to_y_wrapper(opt *th, const std::size_t index_trans);

  private:
    // parameters
    std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
    // 1D adapters
    hpxfft::util::fftw_adapter::r2c_1d fft_r2c_adapter_;
    hpxfft::util::fftw_adapter::c2c_1d fft_c2c_adapter_;
    // value vectors
    vector_2d values_vec_;
    vector_2d trans_values_vec_;
    // time measurement
    hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
    std::map<std::string, real> measurements_;
    // future vectors
    vector_future r2c_futures_;
    vector_future trans_y_to_x_futures_;
    vector_future c2c_futures_;
    vector_future trans_x_to_y_futures_;
};
}  // namespace hpxfft::shared
#endif  // hpxfft_shared_opt_H_INCLUDED
