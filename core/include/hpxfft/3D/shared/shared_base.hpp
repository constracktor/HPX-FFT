#pragma once
#ifndef hpxfft_shared_base_3D_H_INCLUDED
#define hpxfft_shared_base_3D_H_INCLUDED

#include "../../util/adapter_fftw.hpp"
#include "../../util/vector_3d.hpp"              // for hpxfft::util::vector_3d
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::fft3D::shared
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct base
{
  public:
    base() = default;

    real get_measurement(std::string name);

    ~base() { hpxfft::util::fftw_adapter::cleanup(); }

  protected:
    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i, const std::size_t j);
    void fft_1d_c2c_y_inplace(const std::size_t i, const std::size_t j);
    void fft_1d_c2c_x_inplace(const std::size_t i, const std::size_t j);

    // permute
    void permute_shared_x_z_y(const std::size_t slice_x);
    void permute_shared_z_y_x(const std::size_t slice_y);
    void permute_shared_z_x_y(const std::size_t slice_z);

  protected:
    // prarameters
    std::size_t dim_r_z_, dim_c_z_, dim_c_y_, dim_c_x_;
    // IMPORTANT: declare r2c adapter before c2c so r2c destructor is called after c2c
    hpxfft::util::fftw_adapter::r2c_1d fftw_r2c_adapter_dir_z_;
    hpxfft::util::fftw_adapter::c2c_1d fftw_c2c_adapter_dir_y_;
    hpxfft::util::fftw_adapter::c2c_1d fftw_c2c_adapter_dir_x_;
    // value vectors
    vector_3d values_vec_;
    vector_3d permuted_vec_;
    // time measurement
    hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
    std::map<std::string, real> measurements_;
};

inline real hpxfft::fft3D::shared::base::get_measurement(std::string name) { return measurements_[name]; }

inline void hpxfft::fft3D::shared::base::fft_1d_r2c_inplace(const std::size_t i, const std::size_t j)
{
    fftw_r2c_adapter_dir_z_.execute(
        values_vec_.vector_z(i, j), reinterpret_cast<fftw_complex *>(values_vec_.vector_z(i, j)));
}

inline void hpxfft::fft3D::shared::base::fft_1d_c2c_y_inplace(const std::size_t i, const std::size_t j)
{
    fftw_c2c_adapter_dir_y_.execute(reinterpret_cast<fftw_complex *>(permuted_vec_.vector_z(i, j)),
                                    reinterpret_cast<fftw_complex *>(permuted_vec_.vector_z(i, j)));
}

inline void hpxfft::fft3D::shared::base::fft_1d_c2c_x_inplace(const std::size_t i, const std::size_t j)
{
    fftw_c2c_adapter_dir_x_.execute(reinterpret_cast<fftw_complex *>(values_vec_.vector_z(i, j)),
                                    reinterpret_cast<fftw_complex *>(values_vec_.vector_z(i, j)));
}

inline void hpxfft::fft3D::shared::base::permute_shared_x_z_y(const std::size_t slice_x)
{
    const std::size_t n_x = values_vec_.n_x();
    const std::size_t n_y = values_vec_.n_y();
    const std::size_t n_z = values_vec_.n_z();
    const std::size_t n_z_c = n_z / 2;

    for (std::size_t index_y = 0; index_y < n_y; ++index_y)
    {
        for (std::size_t index_z = 0; index_z < n_z_c; ++index_z)
        {
            permuted_vec_(slice_x, index_z, 2 * index_y) = values_vec_(slice_x, index_y, 2 * index_z);
            permuted_vec_(slice_x, index_z, 2 * index_y + 1) = values_vec_(slice_x, index_y, 2 * index_z + 1);
        }
    }
}

inline void hpxfft::fft3D::shared::base::permute_shared_z_y_x(const std::size_t slice_y)
{
    const std::size_t n_x = permuted_vec_.n_x();
    const std::size_t n_y = permuted_vec_.n_y();
    const std::size_t n_z = permuted_vec_.n_z();
    const std::size_t n_z_c = n_z / 2;

    for (std::size_t index_x = 0; index_x < n_x; ++index_x)
    {
        for (std::size_t index_z = 0; index_z < n_z_c; ++index_z)
        {
            values_vec_(index_z, slice_y, 2 * index_x) = permuted_vec_(index_x, slice_y, 2 * index_z);
            values_vec_(index_z, slice_y, 2 * index_x + 1) = permuted_vec_(index_x, slice_y, 2 * index_z + 1);
        }
    }
}

inline void hpxfft::fft3D::shared::base::permute_shared_z_x_y(const std::size_t slice_x)
{
    const std::size_t n_x = values_vec_.n_x();
    const std::size_t n_y = values_vec_.n_y();
    const std::size_t n_z = values_vec_.n_z();
    const std::size_t n_z_c = n_z / 2;

    for (std::size_t index_y = 0; index_y < n_y; ++index_y)
    {
        for (std::size_t index_z = 0; index_z < n_z_c; ++index_z)
        {
            permuted_vec_(index_z, slice_x, 2 * index_y) = values_vec_(slice_x, index_y, 2 * index_z);
            permuted_vec_(index_z, slice_x, 2 * index_y + 1) = values_vec_(slice_x, index_y, 2 * index_z + 1);
        }
    }
}
}  // namespace hpxfft::fft3D::shared
#endif  // hpxfft_shared_3D_H_INCLUDED
