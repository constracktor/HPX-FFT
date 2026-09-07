#pragma once
#ifndef hpxfft_shared_loop_3D_H_INCLUDED
#define hpxfft_shared_loop_3D_H_INCLUDED

#include "../../util/vector_3d.hpp"  // for hpxfft::util::vector_3d
#include "shared_base.hpp"
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::fft3D::shared
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct loop : public base
{
  public:
    loop() = default;

    void initialize(vector_3d values_vec, const std::string PLAN_FLAG);

    vector_3d fft_3d_r2c_par();

    vector_3d fft_3d_r2c_seq();

    void write_plans_to_file(std::string file_path);
};
}  // namespace hpxfft::fft3D::shared
#endif  // hpxfft_shared_loop_3D_H_INCLUDED
