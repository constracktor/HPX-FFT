#pragma once
#ifndef hpxfft_distributed_base_3D_H_INCLUDED
#define hpxfft_distributed_base_3D_H_INCLUDED

#include "../../util/adapter_fftw.hpp"
#include "../../util/blocks.hpp"
#include "../../util/check_shape.hpp"
#include "../../util/comm_flag.hpp"
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

    // nz: number of real values per row in z; 0 derives it from the row length as 2 * (n_z / 2 - 1)
    virtual void initialize(
        vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG, const std::size_t nz = 0) = 0;
    // One transform per initialize: the result is moved out of the object (with the communication buffers), so
    // calling it again without a new initialize throws std::logic_error.
    virtual vector_3d fft_3d_r2c() = 0;

    real get_measurement(std::string name);

    // Skip the final exchange(s) and keep the result in the transposed layout (like FFTW_MPI_TRANSPOSED_OUT).
    void set_transposed_output(bool transposed_out) { transposed_out_ = transposed_out; }

    // Parameters of the hierarchical communicators used by the run flag "scatter_hierarchical" (reference): number of
    // children per tree node, and the number of localities (in a group) below which they collapse to flat
    // communicators.
    void set_hierarchical_parameters(std::size_t arity, std::size_t flat_fallback_threshold)
    {
        hierarchical_arity_ = arity;
        hierarchical_flat_threshold_ = flat_fallback_threshold;
    }

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
    // initialize + one transform per object: the transform moves the data and the communication buffers out of the
    // object, so a second call without a new initialize (or a call without any) throws std::logic_error
    void begin_transform(const char *name)
    {
        if (!ready_)
        {
            throw std::logic_error(std::string(name)
                                   + " needs a preceding initialize: every initialize allows one transform, which "
                                     "moves the data out of the object");
        }
        ready_ = false;
    }

    // Throw std::invalid_argument on every locality if the run flag is not supported (call before any collective).
    static void check_comm_flag(const std::string &flag, const char *name)
    {
        hpxfft::util::check_comm_flag(
            flag, { "scatter", "scatter_async", "scatter_hierarchical", "all_to_all", "all_to_all_routed" }, name);
    }

    // Map the run flag onto COMM_FLAG_ ("scatter" or "all_to_all") and the variant flags below. num_sites is the
    // number of localities taking part in one exchange.
    void parse_comm_flag(const std::string &flag, std::size_t num_sites)
    {
        // a new initialize must not keep the variant of the previous one
        scatter_sync_ = true;
        routed_ = false;
        hierarchical_scatter_ = false;
        COMM_FLAG_ = flag;
        if (flag == "scatter_async")
        {
            scatter_sync_ = false;
            COMM_FLAG_ = "scatter";
        }
        else if (flag == "scatter_hierarchical")
        {
            hierarchical_scatter_ = true;
            COMM_FLAG_ = "scatter";
        }
        else if (flag == "all_to_all_routed")
        {
            routed_ = true;
            COMM_FLAG_ = "all_to_all";
        }
        // the direct (pairwise) exchange needs at least three sites; with fewer the routed all_to_all is used
        if (COMM_FLAG_ == "all_to_all" && num_sites < 3)
        {
            routed_ = true;
        }
    }

    // prarameters
    std::size_t n_x_local_, n_y_local_, n_z_local_;
    std::size_t dim_r_z_, dim_c_z_, dim_c_y_, dim_c_x_;
    std::size_t dim_c_z_part_, dim_c_y_part_, dim_c_x_part_;

    // must precede the plans: destroyed after them
    hpxfft::util::fftw_adapter::cleanup_guard fftw_cleanup_guard_;
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
    std::vector<hpx::collectives::communicator> communicators_;
    // run variants
    bool scatter_sync_ = true;
    // set by a successful initialize, cleared by the transform
    bool ready_ = false;
    bool transposed_out_ = false;
    // all_to_all (run flag all_to_all) is a direct exchange between the localities; the run flag all_to_all_routed
    // selects the reference implementation that routes the data through a root site
    bool routed_ = false;
    // scatter through one hierarchical communicator per root (run flag scatter_hierarchical); needs an HPX whose
    // hierarchical communicators support a root site other than 0
    bool hierarchical_scatter_ = false;
    std::size_t hierarchical_arity_ = 2;
    std::size_t hierarchical_flat_threshold_ = 16;
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
