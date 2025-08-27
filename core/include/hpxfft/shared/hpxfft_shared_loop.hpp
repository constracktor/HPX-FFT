#ifndef hpxfft_shared_loop_H_INCLUDED
#define hpxfft_shared_loop_H_INCLUDED

#include <hpx/timing/high_resolution_timer.hpp> // for hpx::chrono::high_resolution_timer
#include <fftw3.h> // for fftw_plan, fftw_destroy_plan, fftw_cleanup, FFTW_FLAGS
#include "../util/vector_2d.hpp" // for hpxfft::util::vector_2d

typedef double real;

namespace hpxfft::shared
{
    using vector_2d = hpxfft::util::vector_2d<real>;

    struct loop
    {
        typedef fftw_plan fft_backend_plan;
 
        public:
            loop() = default;

            void initialize(vector_2d values_vec, 
                            const unsigned PLAN_FLAG);

            vector_2d fft_2d_r2c_par();

            vector_2d fft_2d_r2c_seq();

            real get_measurement(std::string name);

            void write_plans_to_file(std::string file_path);

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
            vector_2d values_vec_;
            vector_2d trans_values_vec_;
            // time measurement
            hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
            std::map<std::string, real> measurements_;
    };
}
#endif // hpxfft_shared_loop_H_INCLUDED
