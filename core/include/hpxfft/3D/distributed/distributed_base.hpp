#pragma once
#ifndef hpxfft_distributed_base_3D_H_INCLUDED
#define hpxfft_distributed_base_3D_H_INCLUDED

#include "../../util/adapter_fftw.hpp"
#include "../../util/vector_3d.hpp"  // for hpxfft::util::vector_3d
#include <hpx/future.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer

typedef double real;

namespace hpxfft::fft3D::distributed
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct base
{
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<vector_3d> vector_comm;

  public:
    base() = default;

    virtual void initialize(vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG) = 0;
    virtual vector_3d fft_3d_r2c() = 0;

    real get_measurement(std::string name);

    ~base() { hpxfft::util::fftw_adapter::cleanup(); }

  protected:
    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i, const std::size_t j);
    void fft_1d_c2c_y_inplace(const std::size_t i, const std::size_t j);
    void fft_1d_c2c_x_inplace(const std::size_t i, const std::size_t j);

    // premute (only local data)
    virtual void permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i) = 0;
    virtual void permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i) = 0;
    virtual void permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i) = 0;

  protected:
    // prarameters
    std::size_t n_x_local_, n_y_local_, n_z_local_;
    std::size_t dim_r_z_, dim_c_z_, dim_c_y_, dim_c_x_;
    std::size_t dim_c_z_part_, dim_c_y_part_, dim_c_x_part_;

    hpxfft::util::fftw_adapter::r2c_1d fftw_r2c_adapter_dir_z_;
    hpxfft::util::fftw_adapter::c2c_1d fftw_c2c_adapter_dir_y_;
    hpxfft::util::fftw_adapter::c2c_1d fftw_c2c_adapter_dir_x_;
    // value vectors
    vector_3d values_vec_;
    vector_3d permuted_vec_;
    // time measurement
    hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
    std::map<std::string, real> measurements_;
    // communication vectors
    vector_comm values_prep_;
    vector_comm permuted_values_prep_;
    vector_comm communication_vec_;
    std::vector<hpx::future<vector_3d>> communication_futures_;
    // locality information
    std::size_t this_locality_, num_localities_;
    // communicators
    std::string COMM_FLAG_;
    std::vector<std::string> basename_storage_;
    std::vector<const char *> basenames_;
    std::vector<hpx::collectives::communicator> communicators_;
};

inline real hpxfft::fft3D::distributed::base::get_measurement(std::string name) { return measurements_[name]; }

inline void hpxfft::fft3D::distributed::base::fft_1d_r2c_inplace(const std::size_t i, const std::size_t j)
{
    fftw_r2c_adapter_dir_z_.execute(
        values_vec_.vector_z(i, j), reinterpret_cast<fftw_complex *>(values_vec_.vector_z(i, j)));
}

inline void hpxfft::fft3D::distributed::base::fft_1d_c2c_y_inplace(const std::size_t i, const std::size_t j)
{
    fftw_c2c_adapter_dir_y_.execute(reinterpret_cast<fftw_complex *>(permuted_vec_.vector_z(i, j)),
                                    reinterpret_cast<fftw_complex *>(permuted_vec_.vector_z(i, j)));
}

inline void hpxfft::fft3D::distributed::base::fft_1d_c2c_x_inplace(const std::size_t i, const std::size_t j)
{
    fftw_c2c_adapter_dir_x_.execute(reinterpret_cast<fftw_complex *>(values_vec_.vector_z(i, j)),
                                    reinterpret_cast<fftw_complex *>(values_vec_.vector_z(i, j)));
}
}  // namespace hpxfft::fft3D::distributed
#endif  // hpxfft_distributed_base_3D_H_INCLUDED
