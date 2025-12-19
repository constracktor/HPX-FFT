#include "../../include/hpxfft/shared/sync.hpp"

// FFT backend
void hpxfft::shared::sync::fft_1d_r2c_inplace(const std::size_t i)
{
    fftw_r2c_adapter_.execute_r2c(values_vec_.row(i), reinterpret_cast<fftw_complex *>(values_vec_.row(i)));
}

void hpxfft::shared::sync::fft_1d_c2c_inplace(const std::size_t i)
{
    fftw_c2c_adapter_.execute_c2c(reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)),
                                  reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)));
}

// transpose with write running index
void hpxfft::shared::sync::transpose_shared_y_to_x(const std::size_t index)
{
    for (std::size_t index_trans = 0; index_trans < dim_c_x_; ++index_trans)
    {
        trans_values_vec_(index, 2 * index_trans) = values_vec_(index_trans, 2 * index);
        trans_values_vec_(index, 2 * index_trans + 1) = values_vec_(index_trans, 2 * index + 1);
    }
}

// transpose with read running index
void hpxfft::shared::sync::transpose_shared_x_to_y(const std::size_t index_trans)
{
    for (std::size_t index = 0; index < dim_c_x_; ++index)
    {
        values_vec_(index, 2 * index_trans) = trans_values_vec_(index_trans, 2 * index);
        values_vec_(index, 2 * index_trans + 1) = trans_values_vec_(index_trans, 2 * index + 1);
    }
}

// wrappers
void hpxfft::shared::sync::fft_1d_r2c_inplace_wrapper(sync *th, const std::size_t i) { th->fft_1d_r2c_inplace(i); }

void hpxfft::shared::sync::fft_1d_c2c_inplace_wrapper(sync *th, const std::size_t i) { th->fft_1d_c2c_inplace(i); }

void hpxfft::shared::sync::transpose_shared_y_to_x_wrapper(sync *th, const std::size_t index)
{
    th->transpose_shared_y_to_x(index);
}

void hpxfft::shared::sync::transpose_shared_x_to_y_wrapper(sync *th, const std::size_t index_trans)
{
    th->transpose_shared_x_to_y(index_trans);
}

// 2D FFT algorithm
hpxfft::shared::vector_2d hpxfft::shared::sync::fft_2d_r2c()
{
    auto start_total = t_.now();
    // first dimension
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        // 1d FFT r2c in y-direction
        r2c_futures_[i] = hpx::async(&fft_1d_r2c_inplace_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(r2c_futures_);
    auto start_first_trans = t_.now();
    // for(std::size_t i = 0; i < dim_c_x_; ++i) for other transpose
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // transpose from y-direction to x-direction
        trans_y_to_x_futures_[i] = hpx::async(&hpxfft::shared::sync::transpose_shared_y_to_x_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(trans_y_to_x_futures_);
    // second dimension
    auto start_second_fft = t_.now();
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // 1D FFT in x-direction
        c2c_futures_[i] = hpx::async(&hpxfft::shared::sync::fft_1d_c2c_inplace_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(c2c_futures_);
    auto start_second_trans = t_.now();
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        trans_x_to_y_futures_[i] = hpx::async(&hpxfft::shared::sync::transpose_shared_x_to_y_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(trans_x_to_y_futures_);
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

// initialization
void hpxfft::shared::sync::initialize(hpxfft::shared::vector_2d values_vec, const std::string PLAN_FLAG)
{
    // move data into own data structure
    values_vec_ = std::move(values_vec);
    // parameters
    dim_c_x_ = values_vec_.n_row();
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = 2 * dim_c_y_ - 2;
    // resize transposed data structure
    trans_values_vec_ = std::move(hpxfft::shared::vector_2d(dim_c_y_, 2 * dim_c_x_));
    // create FFTW plans
    PLAN_FLAG_ = hpxfft::util::string_to_fftw_plan_flag(PLAN_FLAG);
    // r2c in y-direction
    fftw_r2c_adapter_ = hpxfft::util::fftw_adapter_r2c();
    fftw_r2c_adapter_.initialize(
        dim_r_y_, PLAN_FLAG_, trans_values_vec_.row(0), reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)));
    // c2c in x-direction
    fftw_c2c_adapter_ = hpxfft::util::fftw_adapter_c2c();
    fftw_c2c_adapter_.initialize(
        dim_c_x_,
        PLAN_FLAG_,
        reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
        reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
        hpxfft::util::fftw_direction::forward);
    // resize futures
    r2c_futures_.resize(dim_c_x_);
    trans_y_to_x_futures_.resize(dim_c_x_);
    c2c_futures_.resize(dim_c_y_);
    trans_x_to_y_futures_.resize(dim_c_y_);
}

// helpers
real hpxfft::shared::sync::get_measurement(std::string name) { return measurements_[name]; }
