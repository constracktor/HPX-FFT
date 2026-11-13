#pragma once
#ifndef hpxfft_shared_agas_H_INCLUDED
#define hpxfft_shared_agas_H_INCLUDED
#include "agas_server.hpp"
#include <hpx/hpx.hpp>

namespace hpxfft::fft2D::shared
{  ///////////////////////////////////////////////////////////////////////////////

// This is a client side member function
struct agas : ::hpx::components::client_base<agas, agas_server>
{
    typedef ::hpx::components::client_base<agas, agas_server> base_type;

    explicit agas() :
        base_type(hpx::new_<agas_server>(hpx::find_here()))
    { }

    hpx::future<vector_2d> fft_2d_r2c() { return ::hpx::async(agas_server::fft_2d_r2c_action(), get_id()); }

    // ny: see agas_server::initialize (0: derive an even ny from the row length)
    hpx::future<void> initialize(vector_2d values_vec, const std::string PLAN_FLAG, const std::size_t ny = 0)
    {
        return ::hpx::async(agas_server::initialize_action(), get_id(), std::move(values_vec), PLAN_FLAG, ny);
    }

    ~agas() = default;
};
}  // namespace hpxfft::fft2D::shared
#endif  // hpxfft_shared_agas_H_INCLUDED
