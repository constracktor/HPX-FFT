#pragma once
#ifndef hpxfft_distributed_agas_server_H_INCLUDED
#define hpxfft_distributed_agas_server_H_INCLUDED

#include <hpx/future.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp> // for hpx::chrono::high_resolution_timer
#include <fftw3.h> // for fftw_plan, fftw_destroy_plan, fftw_cleanup, FFTW_FLAGS
#include "../util/vector_2d.hpp" // for hpxfft::util::vector_2d

typedef double real;

namespace hpxfft::distributed
{
    using vector_2d = hpxfft::util::vector_2d<real>;

    struct agas_server: hpx::components::component_base<agas_server>
{
    typedef fftw_plan fft_backend_plan;
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<std::vector<real>> vector_comm;

    public:
        agas_server() = default;

        void initialize(vector_2d values_vec, 
                        const std::string COMM_FLAG,
                        const unsigned PLAN_FLAG);

        vector_2d fft_2d_r2c();


        virtual ~agas_server()
        {
            fftw_destroy_plan(plan_1d_r2c_);
            fftw_destroy_plan(plan_1d_c2c_);
            fftw_cleanup();
        }

    private:
        // FFT backend
        void fft_1d_r2c_inplace(const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    fft_1d_r2c_inplace, 
                                    fft_1d_r2c_inplace_action)
        void fft_1d_c2c_inplace(const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    fft_1d_c2c_inplace,
                                    fft_1d_c2c_inplace_action)
        
        // split data for communication
        void split_vec(const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    split_vec,
                                    split_vec_action)

        void split_trans_vec(const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server,
                                    split_trans_vec,
                                    split_trans_vec_action)

        // scatter communication
        void communicate_scatter_vec(const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    communicate_scatter_vec, 
                                    communicate_scatter_vec_action)

        void communicate_scatter_trans_vec(const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    communicate_scatter_trans_vec, 
                                    communicate_scatter_trans_vec_action)

        // all to all communication
        void communicate_all_to_all_vec();
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    communicate_all_to_all_vec, 
                                    communicate_all_to_all_vec_action)

        void communicate_all_to_all_trans_vec();
        HPX_DEFINE_COMPONENT_ACTION(agas_server,
                                    communicate_all_to_all_trans_vec, 
                                    communicate_all_to_all_trans_vec_action)

        // transpose after communication
        void transpose_y_to_x(const std::size_t k, const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server, 
                                    transpose_y_to_x,
                                    transpose_y_to_x_action)

        void transpose_x_to_y(const std::size_t k, const std::size_t i);
        HPX_DEFINE_COMPONENT_ACTION(agas_server,
                                    transpose_x_to_y,
                                    transpose_x_to_y_action)

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
        // communicators
        std::string COMM_FLAG_;
        std::vector<const char*> basenames_;
        std::vector<hpx::collectives::communicator> communicators_;
};
}

HPX_DEFINE_COMPONENT_ACTION(hpxfft::distributed::agas_server, 
                            initialize, 
                            initialize_action)

HPX_DEFINE_COMPONENT_ACTION(hpxfft::distributed::agas_server, 
                            fft_2d_r2c,
                            fft_2d_r2c_action)

#endif // hpxfft_distributed_agas_server_H_INCLUDED
