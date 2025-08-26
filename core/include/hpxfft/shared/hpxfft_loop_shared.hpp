#ifndef hpxfft_loop_shared_H_INCLUDED
#define hpxfft_loop_shared_H_INCLUDED
#include <hpx/config.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/timing/high_resolution_timer.hpp>

#include <string>
#include <map>
#include <fftw3.h>

#include "../util/vector_2d.hpp"

typedef double real;

// FFT calculation struct
struct hpxfft
{
    typedef fftw_plan fft_backend_plan;

    public:
        hpxfft() = default;

        void initialize(vector_2d<real> values_vec, 
                        const unsigned PLAN_FLAG);

        vector_2d<real> fft_2d_r2c_par();

        vector_2d<real> fft_2d_r2c_seq();

        real get_measurement(std::string name);

        void write_plans_to_file(FILE* file_name);

        ~hpxfft()
        {
            fftw_destroy_plan(plan_1d_r2c_);
            fftw_destroy_plan(plan_1d_c2c_);
            fftw_cleanup();
        }

    private:
        // FFT backend
        void fft_1d_r2c_inplace(const std::size_t i);
        void fft_1d_c2c_inplace(const std::size_t i);

        // transpose
        // transpose with write running index
        void transpose_shared_y_to_x(const std::size_t index);
        //void transpose_shared_y_to_x(const std::size_t index_trans);
        // transpose with read running index
        //void transpose_shared_x_to_y(const std::size_t index);
        void transpose_shared_x_to_y(const std::size_t index_trans);    

    private:
        // parameters
        std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
        // FFTW plans
        unsigned PLAN_FLAG_;
        fft_backend_plan plan_1d_r2c_;
        fft_backend_plan plan_1d_c2c_;
        // value vectors
        vector_2d<real> values_vec_;
        vector_2d<real> trans_values_vec_;
        // time measurement
        hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
        std::map<std::string, real> measurements_;
};
#endif // hpxfft_loop_shared_H_INCLUDED
