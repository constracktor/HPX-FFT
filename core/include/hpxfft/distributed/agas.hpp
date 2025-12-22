#pragma once
#ifndef hpxfft_distributed_agas_H_INCLUDED
#define hpxfft_distributed_agas_H_INCLUDED

#include "agas_server.hpp"
#include <hpx/hpx.hpp>

namespace hpxfft::distributed
{
///////////////////////////////////////////////////////////////////////////////
// This is a client side member function. It can now be implemented as the
// agas_server has been defined.
struct agas : hpx::components::client_base<agas, agas_server>
{
    typedef hpx::components::client_base<agas, agas_server> base_type;

    explicit agas() :
        base_type(hpx::new_<agas_server>(hpx::find_here()))
    { }

    hpx::future<vector_2d> fft_2d_r2c() { return hpx::async(fft_2d_r2c_action(), get_id()); }

    hpx::future<void> initialize(vector_2d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG)
    {
        return hpx::async(initialize_action(), get_id(), std::move(values_vec), COMM_FLAG, PLAN_FLAG);
    }

    ~agas() = default;
};
}  // namespace hpxfft::distributed
#endif  // hpxfft_distributed_agas_H_INCLUDED
