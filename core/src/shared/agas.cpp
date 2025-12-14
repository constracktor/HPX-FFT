#include "../../include/hpxfft/shared/agas.hpp"

#include <hpx/modules/components.hpp>

// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<hpxfft::shared::agas_server> agas_server_type;
HPX_REGISTER_COMPONENT(agas_server_type, agas_server);

// HPX_REGISTER_ACTION() exposes the component member function
HPX_REGISTER_ACTION(fft_2d_r2c_action);
HPX_REGISTER_ACTION(initialize_action);

// FFT backend
void hpxfft::shared::agas_server::fft_1d_r2c_inplace(const std::size_t i)
{
    fftw_execute_dft_r2c(plan_1d_r2c_, values_vec_.row(i), reinterpret_cast<fftw_complex *>(values_vec_.row(i)));
}

void hpxfft::shared::agas_server::fft_1d_c2c_inplace(const std::size_t i)
{
    fftw_execute_dft(plan_1d_c2c_,
                     reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)),
                     reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)));
}

// transpose with write running index
void hpxfft::shared::agas_server::transpose_shared_y_to_x(const std::size_t index)
{
    for (std::size_t index_trans = 0; index_trans < dim_c_x_; ++index_trans)
    {
        trans_values_vec_(index, 2 * index_trans) = values_vec_(index_trans, 2 * index);
        trans_values_vec_(index, 2 * index_trans + 1) = values_vec_(index_trans, 2 * index + 1);
    }
}

// transpose with read running index
void hpxfft::shared::agas_server::transpose_shared_x_to_y(const std::size_t index_trans)
{
    for (std::size_t index = 0; index < dim_c_x_; ++index)
    {
        values_vec_(index, 2 * index_trans) = trans_values_vec_(index_trans, 2 * index);
        values_vec_(index, 2 * index_trans + 1) = trans_values_vec_(index_trans, 2 * index + 1);
    }
}

// 2D FFT algorithm
hpxfft::shared::vector_2d hpxfft::shared::agas_server::fft_2d_r2c()
{
    // first dimension
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        // 1d FFT r2c in y-direction
        r2c_futures_[i] = hpx::async(fft_1d_r2c_inplace_action(), get_id(), i);
    }
    // global synchronization
    hpx::shared_future<vector_future> all_r2c_futures = hpx::when_all(r2c_futures_);
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // transpose from y-direction to x-direction
        trans_y_to_x_futures_[i] = all_r2c_futures.then(
            [=, this](hpx::shared_future<vector_future> r)
            {
                r.get();
                return hpx::async(transpose_shared_y_to_x_action(), get_id(), i);
            });
        // second dimension
        // 1D FFT in x-direction
        c2c_futures_[i] = trans_y_to_x_futures_[i].then(
            [=, this](hpx::future<void> r)
            {
                r.get();
                return hpx::async(fft_1d_c2c_inplace_action(), get_id(), i);
            });
        // transpose from x-direction to y-direction
        trans_x_to_y_futures_[i] = c2c_futures_[i].then(
            [=, this](hpx::future<void> r)
            {
                r.get();
                return hpx::async(transpose_shared_x_to_y_action(), get_id(), i);
            });
    }
    // global synchronization
    hpx::wait_all(trans_x_to_y_futures_);

    return std::move(values_vec_);
}

// initialization
void hpxfft::shared::agas_server::initialize(hpxfft::shared::vector_2d values_vec, const unsigned PLAN_FLAG)
{
    // move data into own data structure
    values_vec_ = std::move(values_vec);
    // parameters
    dim_c_x_ = values_vec_.n_row();
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = 2 * dim_c_y_ - 2;
    // resize transposed data structure
    trans_values_vec_ = std::move(hpxfft::shared::vector_2d(dim_c_y_, 2 * dim_c_x_));
    // create fftw plans
    PLAN_FLAG_ = PLAN_FLAG;
    // r2c in y-direction
    plan_1d_r2c_ = fftw_plan_dft_r2c_1d(
        dim_r_y_, trans_values_vec_.row(0), reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)), PLAN_FLAG);
    // c2c in x-direction
    plan_1d_c2c_ = fftw_plan_dft_1d(
        dim_c_x_,
        reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
        reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
        FFTW_FORWARD,
        PLAN_FLAG);
    // resize futures
    r2c_futures_.resize(dim_c_x_);
    trans_y_to_x_futures_.resize(dim_c_x_);
    c2c_futures_.resize(dim_c_y_);
    trans_x_to_y_futures_.resize(dim_c_y_);
}
