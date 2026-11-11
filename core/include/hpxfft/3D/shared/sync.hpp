#pragma once
#ifndef hpxfft_shared_sync_3D_H_INCLUDED
#define hpxfft_shared_sync_3D_H_INCLUDED

#include "../../util/vector_3d.hpp"  // for hpxfft::util::vector_3d
#include "shared_base.hpp"
#include <hpx/future.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::fft3D::shared
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct sync : public base
{
    typedef std::vector<hpx::future<void>> vector_future;

  public:
    sync() = default;

    void initialize(vector_3d values_vec, const std::string PLAN_FLAG);

    vector_3d fft_3d_r2c();

    void write_plans_to_file(std::string file_path);

  private:
    // static wrappers
    static void fft_1d_r2c_inplace_wrapper(sync *th, const std::size_t i, const std::size_t j);
    static void fft_1d_c2c_y_inplace_wrapper(sync *th, const std::size_t i, const std::size_t j);
    static void fft_1d_c2c_x_inplace_wrapper(sync *th, const std::size_t i, const std::size_t j);
    static void permute_shared_x_z_y_wrapper(sync *th, const std::size_t slice_x);
    static void permute_shared_z_y_x_wrapper(sync *th, const std::size_t slice_y);
    static void permute_shared_z_x_y_wrapper(sync *th, const std::size_t slice_x);

    // future vectors
    vector_future fft_z_r2c_futures_;
    vector_future permute_first_futures_;
    vector_future fft_y_c2c_futures_;
    vector_future permute_second_futures_;
    vector_future fft_x_c2c_futures_;
    vector_future permute_third_futures_;
};
}  // namespace hpxfft::fft3D::shared
#endif  // hpxfft_shared_sync_3D_H_INCLUDED
