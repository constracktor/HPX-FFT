#pragma once
#ifndef hpxfft_distributed_agas_server_H_INCLUDED
#define hpxfft_distributed_agas_server_H_INCLUDED

#include "../../util/adapter_fftw.hpp"
#include "../../util/vector_2d.hpp"  // for hpxfft::util::vector_2d
#include <hpx/future.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::fft2D::distributed
{
using vector_2d = hpxfft::util::vector_2d<real>;

struct agas_server : hpx::components::component_base<agas_server>
{
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<std::vector<real>> vector_comm;

  public:
    agas_server() = default;

    void initialize(vector_2d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG);

    // One transform per initialize (see the 2D loop): a second call without a new initialize throws
    // std::logic_error.
    vector_2d fft_2d_r2c();

    // actions of the public member functions (registered as hpxfft_fft2D_distributed_agas_*)
    HPX_DEFINE_COMPONENT_ACTION(agas_server, initialize, initialize_action)
    HPX_DEFINE_COMPONENT_ACTION(agas_server, fft_2d_r2c, fft_2d_r2c_action)

  private:
    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, fft_1d_r2c_inplace, fft_1d_r2c_inplace_action)
    void fft_1d_c2c_inplace(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, fft_1d_c2c_inplace, fft_1d_c2c_inplace_action)

    // split data for communication
    void split_vec(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, split_vec, split_vec_action)

    void split_trans_vec(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, split_trans_vec, split_trans_vec_action)

    // scatter communication
    void communicate_scatter_vec(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, communicate_scatter_vec, communicate_scatter_vec_action)

    void communicate_scatter_trans_vec(const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, communicate_scatter_trans_vec, communicate_scatter_trans_vec_action)

    // all to all communication
    void communicate_all_to_all_vec();
    HPX_DEFINE_COMPONENT_ACTION(agas_server, communicate_all_to_all_vec, communicate_all_to_all_vec_action)

    void communicate_all_to_all_trans_vec();
    HPX_DEFINE_COMPONENT_ACTION(agas_server, communicate_all_to_all_trans_vec, communicate_all_to_all_trans_vec_action)

    // transpose after communication
    void transpose_y_to_x(const std::size_t k, const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, transpose_y_to_x, transpose_y_to_x_action)

    void transpose_x_to_y(const std::size_t k, const std::size_t i);
    HPX_DEFINE_COMPONENT_ACTION(agas_server, transpose_x_to_y, transpose_x_to_y_action)

  private:
    // parameters
    std::size_t n_x_local_, n_y_local_;
    std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
    std::size_t dim_c_y_part_, dim_c_x_part_;
    // 1D adapters
    // must precede the plans: destroyed after them
    hpxfft::util::fftw_adapter::cleanup_guard fftw_cleanup_guard_;
    hpxfft::util::fftw_adapter::r2c_1d fft_r2c_adapter_;
    hpxfft::util::fftw_adapter::c2c_1d fft_c2c_adapter_;
    // value vectors
    vector_2d values_vec_;
    vector_2d trans_values_vec_;
    // future vectors
    vector_future r2c_futures_;
    vector_future split_vec_futures_;
    std::vector<hpx::shared_future<void>> communication_futures_;
    std::vector<vector_future> trans_y_to_x_futures_;
    vector_future c2c_futures_;
    vector_future split_trans_vec_futures_;
    std::vector<vector_future> trans_x_to_y_futures_;
    // communication vectors
    vector_comm values_prep_;
    vector_comm trans_values_prep_;
    vector_comm communication_vec_;
    // locality information
    std::size_t this_locality_, num_localities_;
    // set by a successful initialize, cleared by the transform (initialize + one transform per object)
    bool ready_ = false;
    // communicators
    std::string COMM_FLAG_;
    // owns the basename strings passed (as const char *) to create_communicator
    std::vector<std::string> basename_storage_;
    std::vector<hpx::collectives::communicator> communicators_;
};
}  // namespace hpxfft::fft2D::distributed

HPX_REGISTER_ACTION_DECLARATION(hpxfft::fft2D::distributed::agas_server::initialize_action,
                                hpxfft_fft2D_distributed_agas_initialize_action)
HPX_REGISTER_ACTION_DECLARATION(hpxfft::fft2D::distributed::agas_server::fft_2d_r2c_action,
                                hpxfft_fft2D_distributed_agas_fft_2d_r2c_action)

#endif  // hpxfft_distributed_agas_server_H_INCLUDED
