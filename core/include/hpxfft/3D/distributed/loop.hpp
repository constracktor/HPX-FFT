#pragma once
#ifndef hpxfft_distributed_loop_3D_H_INCLUDED
#define hpxfft_distributed_loop_3D_H_INCLUDED

#include "../../util/print_vector_3d.hpp"  // for hpxfft::util::print_vector_3d
#include "../../util/vector_3d.hpp"        // for hpxfft::util::vector_3d
#include "distributed_base.hpp"
#include <cmath>
#include <hpx/hpx_init.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <optional>

typedef double real;

namespace hpxfft::fft3D::distributed::loop
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct slab : public base
{
  public:
    slab() = default;

    void initialize(vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG) override;

    vector_3d fft_3d_r2c() override;

    // Skip the final global transpose: the result stays in the layout
    // (y_local, z, x) with y distributed over the localities, like
    // FFTW_MPI_TRANSPOSED_OUT. Saves the second communication phase.
    void set_transposed_output(bool transposed_out) { transposed_out_ = transposed_out; }

    // Split every exchange into this many chunks along the local x (first
    // exchange) and local y (second exchange). Each chunk is sent as soon as it
    // is computed, so communication overlaps with the remaining computation.
    // 1 keeps the phase-by-phase algorithm. 0 (default) chooses automatically:
    // 8 chunks for all_to_all if the local data is large (chunking only pays off then), otherwise 1
    // (overlap only pays off for the all_to_all collective).
    void set_num_chunks(std::size_t num_chunks) { num_chunks_ = num_chunks; }

    // Parameters of the hierarchical communicator used by the run flag
    // "all_to_all_hierarchical": number of children per tree node, and the
    // number of localities below which it collapses to a flat communicator.
    void set_hierarchical_parameters(std::size_t arity, std::size_t flat_fallback_threshold)
    {
        hierarchical_arity_ = arity;
        hierarchical_flat_threshold_ = flat_fallback_threshold;
    }

  private:
    vector_3d fft_3d_r2c_pipelined(std::size_t num_chunks);
    std::size_t effective_num_chunks() const;

    // permute
    void permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i) override;
    void permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i) override;
    void permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i) override;

    // split data for communication
    void split_vec(const std::size_t i, const std::size_t j);
    void split_permuted_vec(const std::size_t i, const std::size_t j);

    // scatter communication
    void communicate_scatter_vec(const std::size_t i);
    void communicate_scatter_permuted_vec(const std::size_t i);

    // all to all communication
    void communicate_all_to_all_vec();
    void communicate_all_to_all_permuted_vec();

    bool scatter_sync_ = true;
    bool transposed_out_ = false;
    std::size_t num_chunks_ = 0;
    // hierarchical communicator for all_to_all
    bool hierarchical_ = false;
    std::size_t hierarchical_arity_ = 2;
    std::size_t hierarchical_flat_threshold_ = 16;
    std::optional<hpx::collectives::hierarchical_communicator> hierarchical_communicator_;
    hpx::future<std::vector<vector_3d>> all_to_all_exchange(std::vector<vector_3d> &&prep, std::size_t generation);
};

struct pencil : public base
{
  public:
    pencil() = default;

    void initialize(vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG) override;

    vector_3d fft_3d_r2c() override;

    // Skip the two final exchanges: the result stays in the layout
    // (y_local, z_local, x), y distributed over the x-direction of the locality
    // grid and z over its y-direction. Saves half of the communication.
    void set_transposed_output(bool transposed_out) { transposed_out_ = transposed_out; }

  private:
    // permute (only local data)
    void permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i) override;
    void permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i) override;
    void permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i) override;

    // split data for communication
    void split_vec(const std::size_t i, const std::size_t j);
    void split_permuted_vec(const std::size_t i, const std::size_t j);

    // scatter communication
    void communicate_scatter_vec_x(const std::size_t i);
    void communicate_scatter_vec_y(const std::size_t i);
    void communicate_scatter_permuted_vec_x(const std::size_t i);
    void communicate_scatter_permuted_vec_y(const std::size_t i);

    // all to all communication
    void communicate_all_to_all_vec_x();
    void communicate_all_to_all_vec_y();
    void communicate_all_to_all_permuted_vec_x();
    void communicate_all_to_all_permuted_vec_y();

    // locality information
    std::size_t localities_per_dir_;
    std::size_t pos_dir_x_;
    std::size_t pos_dir_y_;
    std::size_t generation_counterx_;
    std::size_t generation_countery_;

    vector_comm values_prep2_;
    vector_comm permuted_values_prep2_;

    bool scatter_sync_ = true;
    bool transposed_out_ = false;
};

}  // namespace hpxfft::fft3D::distributed::loop
#endif  // hpxfft_distributed_loop_3D_H_INCLUDED
