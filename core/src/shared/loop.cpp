#include "../../include/hpxfft/shared/loop.hpp"

#include <hpx/parallel/algorithms/for_loop.hpp>

// FFT backend
void hpxfft::shared::loop::fft_1d_r2c_inplace(const std::size_t i)
{
    fft_r2c_adapter_.execute(values_vec_.row(i), reinterpret_cast<fftw_complex *>(values_vec_.row(i)));
}

void hpxfft::shared::loop::fft_1d_c2c_inplace(const std::size_t i)
{
    fft_c2c_adapter_.execute(reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)),
                             reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)));
}

// transpose with write running index
void hpxfft::shared::loop::transpose_shared_y_to_x(const std::size_t index)
{
    for (std::size_t index_trans = 0; index_trans < dim_c_x_; ++index_trans)
    {
        trans_values_vec_(index, 2 * index_trans) = values_vec_(index_trans, 2 * index);
        trans_values_vec_(index, 2 * index_trans + 1) = values_vec_(index_trans, 2 * index + 1);
    }
}

// void hpxfft::shared::loop::transpose_shared_x_to_y(const std::size_t index)
// {
//     for( std::size_t index_trans = 0; index_trans < dim_c_y_; ++index_trans)
//     {
//         values_vec_(index, 2 * index_trans) = trans_values_vec_(index_trans, 2 * index);
//         values_vec_(index, 2 * index_trans + 1) = trans_values_vec_(index_trans, 2 * index + 1);
//     }
// }

// transpose with read running index
// void hpxfft::shared::loop::transpose_shared_y_to_x(const std::size_t index_trans)
// {
//     for( std::size_t index = 0; index < dim_c_y_; ++index)
//     {
//         trans_values_vec_(index, 2 * index_trans) = values_vec_(index_trans, 2 * index);
//         trans_values_vec_(index, 2 * index_trans + 1) = values_vec_(index_trans, 2 * index + 1);
//     }
// }

void hpxfft::shared::loop::transpose_shared_x_to_y(const std::size_t index_trans)
{
    for (std::size_t index = 0; index < dim_c_x_; ++index)
    {
        values_vec_(index, 2 * index_trans) = trans_values_vec_(index_trans, 2 * index);
        values_vec_(index, 2 * index_trans + 1) = trans_values_vec_(index_trans, 2 * index + 1);
    }
}

// 2D FFT algorithm
hpxfft::shared::vector_2d hpxfft::shared::loop::fft_2d_r2c_par()
{
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_x_,
        [&](auto i)
        {
            // 1d FFT r2c in y-direction
            fft_1d_r2c_inplace(i);
        });
    auto start_first_trans = t_.now();
    // hpx::experimental::for_loop(hpx::execution::par, 0, dim_c_x_, [&](auto i) for other transpose
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_y_,
        [&](auto i)
        {
            // transpose from y-direction to x-direction
            transpose_shared_y_to_x(i);
        });
    // second dimension
    auto start_second_fft = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_y_,
        [&](auto i)
        {
            // 1D FFT c2c in x-direction
            fft_1d_c2c_inplace(i);
        });
    auto start_second_trans = t_.now();
    // hpx::experimental::for_loop(hpx::execution::par, 0, dim_c_x_, [&](auto i) for other transpose
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_y_,
        [&](auto i)
        {
            // transpose from x-direction to y-direction
            transpose_shared_x_to_y(i);
        });
    auto stop_total = t_.now();
    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_trans - start_total;
    measurements_["first_trans"] = start_second_fft - start_first_trans;
    measurements_["second_fftw"] = start_second_trans - start_second_fft;
    measurements_["second_trans"] = stop_total - start_second_trans;

    return std::move(values_vec_);
}

hpxfft::shared::vector_2d hpxfft::shared::loop::fft_2d_r2c_seq()
{
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        // 1d FFT r2c in y-direction
        fft_1d_r2c_inplace(i);
    }
    auto start_first_trans = t_.now();
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        // transpose from y-direction to x-direction
        transpose_shared_y_to_x(i);
    }
    // second dimension
    auto start_second_fft = t_.now();
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // 1d FFT c2c in x-direction
        fft_1d_c2c_inplace(i);
    }
    auto start_second_trans = t_.now();
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // transpose from x-direction to y-direction
        transpose_shared_x_to_y(i);
    }
    ////////////////////////////////////////////////////////////////
    // additional runtimes
    auto stop_total = t_.now();
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_trans - start_total;
    measurements_["first_trans"] = start_second_fft - start_first_trans;
    measurements_["second_fftw"] = start_second_trans - start_second_fft;
    measurements_["second_trans"] = stop_total - start_second_trans;

    ///////////////////////////////////////////////////////////////7
    return std::move(values_vec_);
}

// initialization
void hpxfft::shared::loop::initialize(vector_2d values_vec, const std::string PLAN_FLAG)
{
    // move data into own data structure
    values_vec_ = std::move(values_vec);
    // parameters
    dim_c_x_ = values_vec_.n_row();
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = 2 * dim_c_y_ - 2;
    // resize transposed data structure
    trans_values_vec_ = std::move(vector_2d(dim_c_y_, 2 * dim_c_x_));
    // create FFTW plans
    auto start_plan = t_.now();
    // r2c in y-direction
    fft_r2c_adapter_ = hpxfft::util::fftw_adapter::r2c_1d();
    fft_r2c_adapter_.plan(
        dim_r_y_, PLAN_FLAG, trans_values_vec_.row(0), reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)));
    // c2c in x-direction
    fft_c2c_adapter_ = hpxfft::util::fftw_adapter::c2c_1d();
    fft_c2c_adapter_.plan(dim_c_x_,
                          PLAN_FLAG,
                          reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
                          reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
                          hpxfft::util::fftw_adapter::direction::forward);
    auto stop_plan = t_.now();
    measurements_["plan"] = stop_plan - start_plan;
    // compute overall plan flops
    double add_r2c, mul_r2c, fma_r2c;
    fft_r2c_adapter_.flops(&add_r2c, &mul_r2c, &fma_r2c);
    double add_c2c, mul_c2c, fma_c2c;
    fft_c2c_adapter_.flops(&add_c2c, &mul_c2c, &fma_c2c);
    measurements_["plan_flops"] = dim_r_y_ * (add_r2c + mul_r2c + fma_r2c) + dim_c_x_ * (add_c2c + mul_c2c + fma_c2c);
}

// helpers
real hpxfft::shared::loop::get_measurement(std::string name) { return measurements_[name]; }

void hpxfft::shared::loop::write_plans_to_file(std::string file_path)
{
    // Open file
    FILE *file_name = fopen(file_path.c_str(), "a");
    if (!file_name)
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    // Write first plan
    fprintf(file_name, "FFTW r2c 1D plan:\n");
    fft_r2c_adapter_.print_plan(file_name);
    fprintf(file_name, "\n");
    // Write second plan
    fprintf(file_name, "FFTW c2c 1D plan:\n");
    fft_c2c_adapter_.print_plan(file_name);
    fprintf(file_name, "\n\n");
    // Close file
    fclose(file_name);
}
