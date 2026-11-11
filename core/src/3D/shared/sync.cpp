#include "../../../include/hpxfft/3D/shared/sync.hpp"

void hpxfft::fft3D::shared::sync::initialize(vector_3d values_vec, const std::string PLAN_FLAG)
{
    values_vec_ = std::move(values_vec);
    dim_c_x_ = values_vec_.n_x();
    dim_c_y_ = values_vec_.n_y();
    dim_c_z_ = values_vec_.n_z() / 2;
    dim_r_z_ = 2 * dim_c_z_ - 2;
    // resize transposed data structure
    permuted_vec_ = vector_3d(dim_c_x_, dim_c_z_, 2 * dim_c_y_);
    auto start_plan = t_.now();
    // initialize FFTW adapters
    fftw_r2c_adapter_dir_z_ = hpxfft::util::fftw_adapter::r2c_1d();
    fftw_r2c_adapter_dir_z_.plan(
        dim_r_z_, PLAN_FLAG, permuted_vec_.slice_yz(0), reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)));
    fftw_c2c_adapter_dir_y_ = hpxfft::util::fftw_adapter::c2c_1d();
    fftw_c2c_adapter_dir_y_.plan(
        dim_c_y_,
        PLAN_FLAG,
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        hpxfft::util::fftw_adapter::direction::forward);
    fftw_c2c_adapter_dir_x_ = hpxfft::util::fftw_adapter::c2c_1d();
    fftw_c2c_adapter_dir_x_.plan(
        dim_c_x_,
        PLAN_FLAG,
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        hpxfft::util::fftw_adapter::direction::forward);
    auto stop_plan = t_.now();
    measurements_["plan"] = stop_plan - start_plan;
    // compute overall plan flops
    double add_z, mul_z, fma_z;
    fftw_r2c_adapter_dir_z_.flops(&add_z, &mul_z, &fma_z);
    double add_y, mul_y, fma_y;
    fftw_c2c_adapter_dir_y_.flops(&add_y, &mul_y, &fma_y);
    double add_x, mul_x, fma_x;
    fftw_c2c_adapter_dir_x_.flops(&add_x, &mul_x, &fma_x);
    measurements_["plan_flops"] =
        dim_r_z_ * (add_z + mul_z + fma_z) + dim_c_y_ * (add_y + mul_y + fma_y) + dim_c_x_ * (add_x + mul_x + fma_x);
    // resize futures
    fft_z_r2c_futures_.resize(dim_c_x_ * dim_c_y_);
    permute_first_futures_.resize(dim_c_x_);
    fft_y_c2c_futures_.resize(dim_c_x_ * dim_c_z_);
    permute_second_futures_.resize(dim_c_z_);
    fft_x_c2c_futures_.resize(dim_c_y_ * dim_c_z_);
    permute_third_futures_.resize(dim_c_y_);
}

// wrapper for fft_1d_r2c_inplace to use with hpx::async
void hpxfft::fft3D::shared::sync::fft_1d_r2c_inplace_wrapper(sync *th, const std::size_t i, const std::size_t j)
{
    th->fft_1d_r2c_inplace(i, j);
}

void hpxfft::fft3D::shared::sync::fft_1d_c2c_y_inplace_wrapper(sync *th, const std::size_t i, const std::size_t j)
{
    th->fft_1d_c2c_y_inplace(i, j);
}

void hpxfft::fft3D::shared::sync::fft_1d_c2c_x_inplace_wrapper(sync *th, const std::size_t i, const std::size_t j)
{
    th->fft_1d_c2c_x_inplace(i, j);
}

void hpxfft::fft3D::shared::sync::permute_shared_x_z_y_wrapper(sync *th, const std::size_t slice_x)
{
    th->permute_shared_x_z_y(slice_x);
}

void hpxfft::fft3D::shared::sync::permute_shared_z_y_x_wrapper(sync *th, const std::size_t slice_y)
{
    th->permute_shared_z_y_x(slice_y);
}

void hpxfft::fft3D::shared::sync::permute_shared_z_x_y_wrapper(sync *th, const std::size_t slice_x)
{
    th->permute_shared_z_x_y(slice_x);
}

hpxfft::fft3D::shared::vector_3d hpxfft::fft3D::shared::sync::fft_3d_r2c()
{
    auto start_total = t_.now();

    /////////////////////////////////////////////////////////////////
    // First dimension (Z)
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        for (std::size_t j = 0; j < dim_c_y_; ++j)
        {
            fft_z_r2c_futures_[i * dim_c_y_ + j] = hpx::async(&fft_1d_r2c_inplace_wrapper, this, i, j);
        }
    }
    hpx::wait_all(fft_z_r2c_futures_);
    auto start_first_trans = t_.now();

    /////////////////////////////////////////////////////////////////
    // Permute (X, Y, Z) -> (X, Z, Y)
    for (std::size_t slice_x = 0; slice_x < dim_c_x_; ++slice_x)
    {
        permute_first_futures_[slice_x] = hpx::async(&permute_shared_x_z_y_wrapper, this, slice_x);
    }
    hpx::wait_all(permute_first_futures_);
    auto start_second_dim = t_.now();

    /////////////////////////////////////////////////////////////////
    // Second dimension (Y)
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        for (std::size_t j = 0; j < dim_c_z_; ++j)
        {
            fft_y_c2c_futures_[i * dim_c_z_ + j] = hpx::async(&fft_1d_c2c_y_inplace_wrapper, this, i, j);
        }
    }
    hpx::wait_all(fft_y_c2c_futures_);
    values_vec_.rearrange(dim_c_y_, dim_c_z_, 2 * dim_c_x_);
    auto start_second_trans = t_.now();

    /////////////////////////////////////////////////////////////////
    // Permute (X, Z, Y) -> (Y, Z, X)
    for (std::size_t slice_y = 0; slice_y < dim_c_z_; ++slice_y)
    {
        permute_second_futures_[slice_y] = hpx::async(&permute_shared_z_y_x_wrapper, this, slice_y);
    }
    hpx::wait_all(permute_second_futures_);
    auto start_third_dim = t_.now();

    /////////////////////////////////////////////////////////////////
    // Third dimension (X)
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        for (std::size_t j = 0; j < dim_c_z_; ++j)
        {
            fft_x_c2c_futures_[i * dim_c_z_ + j] = hpx::async(&fft_1d_c2c_x_inplace_wrapper, this, i, j);
        }
    }
    hpx::wait_all(fft_x_c2c_futures_);
    permuted_vec_.rearrange(dim_c_x_, dim_c_y_, 2 * dim_c_z_);
    auto start_third_trans = t_.now();

    /////////////////////////////////////////////////////////////////
    // Permute (Y, Z, X) -> (X, Y, Z)
    for (std::size_t slice_x = 0; slice_x < dim_c_y_; ++slice_x)
    {
        permute_third_futures_[slice_x] = hpx::async(&permute_shared_z_x_y_wrapper, this, slice_x);
    }
    hpx::wait_all(permute_third_futures_);
    auto stop_total = t_.now();
    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_trans - start_total;
    measurements_["first_permute"] = start_second_dim - start_first_trans;
    measurements_["second_fftw"] = start_second_trans - start_second_dim;
    measurements_["second_permute"] = start_third_dim - start_second_trans;
    measurements_["third_fftw"] = start_third_trans - start_third_dim;
    measurements_["third_permute"] = stop_total - start_third_trans;
    ////////////////////////////////////////////////////////////////
    return std::move(permuted_vec_);
}

void hpxfft::fft3D::shared::sync::write_plans_to_file(std::string file_path)
{
    // Open file
    FILE *file_name = fopen(file_path.c_str(), "a");
    if (!file_name)
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    // Write first plan
    fprintf(file_name, "FFTW r2c 1D plan:\n");
    fftw_r2c_adapter_dir_z_.print_plan(file_name);
    fprintf(file_name, "\n");
    // Write second plan
    fprintf(file_name, "FFTW c2c 1D plan direction y:\n");
    fftw_c2c_adapter_dir_y_.print_plan(file_name);
    fprintf(file_name, "\n");
    // Write third plan
    fprintf(file_name, "FFTW c2c 1D plan direction x:\n");
    fftw_c2c_adapter_dir_x_.print_plan(file_name);
    fprintf(file_name, "\n\n");
    // Close file
    fclose(file_name);
}
