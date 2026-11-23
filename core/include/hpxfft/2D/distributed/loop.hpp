#pragma once
#ifndef hpxfft_distributed_loop_H_INCLUDED
#define hpxfft_distributed_loop_H_INCLUDED

#include "../../util/adapter_fftw.hpp"
#include "../../util/blocks.hpp"
#include "../../util/check_shape.hpp"
#include "../../util/measurement.hpp"  // for hpxfft::util::get_measurement
#include "../../util/real.hpp"
#include "../../util/transform_guard.hpp"
#include "../../util/vector_2d.hpp"  // for hpxfft::util::vector_2d
#include <hpx/future.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp>  // for hpx::chrono::high_resolution_timer
#include <optional>

namespace hpxfft::fft2D::distributed
{
using vector_2d = hpxfft::util::vector_2d<real>;

struct loop
{
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<std::vector<real>> vector_comm;

  public:
    loop() = default;

    // ny: number of real values per row; 0 derives it from the row length as 2 * (n_col / 2 - 1)
    void initialize(
        vector_2d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG, const std::size_t ny = 0);

    // One transform per initialize: the result is moved out of the object (with the communication buffers), so
    // calling it again without a new initialize throws std::logic_error.
    vector_2d fft_2d_r2c();

    // Skip the final transpose: the result stays in the layout (y_local, x)
    // with y distributed over the localities, like FFTW_MPI_TRANSPOSED_OUT.
    // Saves the second communication phase.
    void set_transposed_output(bool transposed_out) { transposed_out_ = transposed_out; }

    // Parameters of the hierarchical communicators used by the run flag
    // "scatter_hierarchical" (reference): number of children per tree node, and
    // the number of localities below which they collapse to flat communicators.
    void set_hierarchical_parameters(std::size_t arity, std::size_t flat_fallback_threshold)
    {
        hierarchical_arity_ = arity;
        hierarchical_flat_threshold_ = flat_fallback_threshold;
    }

    real get_measurement(const std::string &name) const;

    // y frequencies (complex numbers) held by this locality in the transposed output, and the first one; valid after a
    // successful initialize (also after the transform), std::logic_error before
    std::size_t local_y_count() const
    {
        hpxfft::util::require_initialized(initialized_, "local_y_count of the 2D loop");
        return n_y_local_;
    }

    std::size_t local_y_offset() const
    {
        hpxfft::util::require_initialized(initialized_, "local_y_offset of the 2D loop");
        return y_offsets_[this_locality_];
    }

  private:
    // FFT backend
    void fft_1d_r2c_inplace(const std::size_t i);
    void fft_1d_c2c_inplace(const std::size_t i);

    // split data for communication
    void split_vec(const std::size_t i);
    void split_trans_vec(const std::size_t i);

    // scatter communication
    void communicate_scatter_vec(const std::size_t i);
    void communicate_scatter_trans_vec(const std::size_t i);

    // all to all communication
    void communicate_all_to_all_vec();
    void communicate_all_to_all_trans_vec();

    // blocked transpose after communication of the tile-th block of rows of the output (see loop.cpp)
    void transpose_y_to_x(const std::size_t tile, const std::size_t i);
    void transpose_x_to_y(const std::size_t tile, const std::size_t i);

  private:
    // parameters
    std::size_t n_x_local_, n_y_local_;
    std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
    std::size_t dim_c_y_part_, dim_c_x_part_;
    // block offsets of y: locality i holds the y frequencies [y_offsets_[i], y_offsets_[i + 1]) after the first
    // exchange
    std::vector<std::size_t> y_offsets_;
    // 1D adapters
    // must precede the plans: destroyed after them
    hpxfft::util::fftw_adapter::cleanup_guard fftw_cleanup_guard_;
    hpxfft::util::fftw_adapter::r2c_1d fft_r2c_adapter_;
    hpxfft::util::fftw_adapter::c2c_1d fft_c2c_adapter_;
    // value vectors
    vector_2d values_vec_;
    vector_2d trans_values_vec_;
    // future vectors
    std::vector<hpx::future<std::vector<real>>> communication_futures_;
    // time measurement
    hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
    std::map<std::string, real> measurements_;
    // communication vectors
    vector_comm values_prep_;
    vector_comm trans_values_prep_;
    vector_comm communication_vec_;
    // locality information
    std::size_t this_locality_, num_localities_;
    bool transposed_out_ = false;
    // set by a successful initialize, cleared by the transform (initialize + one transform per object)
    bool ready_ = false;
    // set by a successful initialize, cleared at the start of initialize (not by the transform): the getters of the
    // local output block are valid
    bool initialized_ = false;
    // all_to_all (run flag all_to_all) is a direct exchange between the localities; the run flag all_to_all_routed
    // selects the reference implementation that routes the data through a root site
    bool routed_ = false;
    bool hierarchical_scatter_ = false;
    std::size_t hierarchical_arity_ = 2;
    std::size_t hierarchical_flat_threshold_ = 16;
    std::vector<std::optional<hpx::collectives::hierarchical_communicator>> hierarchical_scatter_communicators_;
    hpx::future<std::vector<std::vector<real>>>
    all_to_all_exchange(std::vector<std::vector<real>> &&prep, std::size_t generation);
    hpx::future<std::vector<real>>
    scatter_exchange(std::size_t root, std::vector<std::vector<real>> &&prep, std::size_t generation);
    // communicators
    std::string COMM_FLAG_;
    std::vector<std::string> basename_storage_;
    std::vector<hpx::collectives::communicator> communicators_;
};
}  // namespace hpxfft::fft2D::distributed
#endif  // hpxfft_distributed_loop_H_INCLUDED
