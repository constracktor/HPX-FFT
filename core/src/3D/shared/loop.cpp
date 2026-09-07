#include "../../../include/hpxfft/3D/shared/loop.hpp"

void hpxfft::fft3D::shared::loop::initialize(vector_3d values_vec, const std::string PLAN_FLAG)
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
}

hpxfft::fft3D::shared::vector_3d hpxfft::fft3D::shared::loop::fft_3d_r2c_par()
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
            for (std::size_t j = 0; j < dim_c_y_; ++j)
            {
                // 1D FFT r2c in z-direction
                fft_1d_r2c_inplace(i, j);
            }
        });
    auto start_first_permute = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_x_,
        [&](auto i)
        {
            // permute from x-y-z to x-z-y
            permute_shared_x_z_y(i);
        });
    // second dimension

    auto start_second_fft = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_x_,
        [&](auto i)
        {
            for (std::size_t j = 0; j < dim_c_z_; ++j)
            {
                // 1D FFT c2c in y-direction
                fft_1d_c2c_y_inplace(i, j);
            }
        });
    auto start_second_permute = t_.now();
    values_vec_.rearrange(dim_c_y_, dim_c_z_, 2 * dim_c_x_);
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_z_,
        [&](auto i)
        {
            // permute from x-z-y to y-z-x
            permute_shared_z_y_x(i);
        });

    // third dimension
    auto start_third_fft = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_y_,
        [&](auto i)
        {
            for (std::size_t j = 0; j < dim_c_z_; ++j)
            {
                // 1D FFT c2c in x-direction
                fft_1d_c2c_x_inplace(i, j);
            }
        });
    auto start_third_permute = t_.now();
    permuted_vec_.rearrange(dim_c_x_, dim_c_y_, 2 * dim_c_z_);
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        dim_c_y_,
        [&](auto i)
        {
            // permute from y-z-x to x-y-z
            permute_shared_z_x_y(i);
        });
    auto stop_total = t_.now();
    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_permute - start_total;
    measurements_["first_permute"] = start_second_fft - start_first_permute;
    measurements_["second_fftw"] = start_second_permute - start_second_fft;
    measurements_["second_permute"] = start_third_fft - start_second_permute;
    measurements_["third_fftw"] = start_third_permute - start_third_fft;
    measurements_["third_permute"] = stop_total - start_third_permute;
    ///////////////////////////////////////////////////////////////
    return std::move(permuted_vec_);
}

hpxfft::fft3D::shared::vector_3d hpxfft::fft3D::shared::loop::fft_3d_r2c_seq()
{
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        for (std::size_t j = 0; j < dim_c_y_; ++j)
        {
            // 1D FFT r2c in z-direction
            fft_1d_r2c_inplace(i, j);
        }
    }
    auto start_first_permute = t_.now();
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        // permute from x-y-z to x-z-y
        permute_shared_x_z_y(i);
    }
    // second dimension
    auto start_second_fft = t_.now();
    for (std::size_t i = 0; i < dim_c_x_; ++i)
    {
        for (std::size_t j = 0; j < dim_c_z_; ++j)
        {
            // 1D FFT c2c in y-direction
            fft_1d_c2c_y_inplace(i, j);
        }
    }
    auto start_second_permute = t_.now();
    values_vec_.rearrange(dim_c_y_, dim_c_z_, 2 * dim_c_x_);
    for (std::size_t i = 0; i < dim_c_z_; ++i)
    {
        // permute from x-z-y to y-z-x
        permute_shared_z_y_x(i);
    }
    // third dimension
    auto start_third_fft = t_.now();
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        for (std::size_t j = 0; j < dim_c_z_; ++j)
        {
            // 1D FFT c2c in x-direction
            fft_1d_c2c_x_inplace(i, j);
        }
    }
    auto start_third_permute = t_.now();
    permuted_vec_.rearrange(dim_c_x_, dim_c_y_, 2 * dim_c_z_);
    for (std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // permute from y-z-x to x-y-z
        permute_shared_z_x_y(i);
    }
    auto stop_total = t_.now();
    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_permute - start_total;
    measurements_["first_permute"] = start_second_fft - start_first_permute;
    measurements_["second_fftw"] = start_second_permute - start_second_fft;
    measurements_["second_permute"] = start_third_fft - start_second_permute;
    measurements_["third_fftw"] = start_third_permute - start_third_fft;
    measurements_["third_permute"] = stop_total - start_third_permute;
    ///////////////////////////////////////////////////////////////
    return std::move(permuted_vec_);
}

void hpxfft::fft3D::shared::loop::write_plans_to_file(std::string file_path)
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
