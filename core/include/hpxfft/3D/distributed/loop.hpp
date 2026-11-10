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

    void initialize(vector_3d values_vec,
                    const std::string COMM_FLAG,
                    const std::string PLAN_FLAG,
                    const std::size_t nz = 0) override;

    vector_3d fft_3d_r2c() override;

  private:
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

    hpx::future<std::vector<vector_3d>> all_to_all_exchange(std::vector<vector_3d> &&prep, std::size_t generation);
    // scatter through one hierarchical communicator per root (communicator i is rooted at locality i); needs an HPX
    // whose hierarchical communicators support a root site other than 0
    std::vector<std::optional<hpx::collectives::hierarchical_communicator>> hierarchical_scatter_communicators_;
    hpx::future<vector_3d> scatter_exchange(std::size_t root, std::vector<vector_3d> &&prep, std::size_t generation);
};

struct pencil : public base
{
  public:
    pencil() = default;

    void initialize(vector_3d values_vec,
                    const std::string COMM_FLAG,
                    const std::string PLAN_FLAG,
                    const std::size_t nz = 0) override;

    vector_3d fft_3d_r2c() override;

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


    // all_to_all of the group (0: x direction, 1: y direction)
    hpx::future<std::vector<vector_3d>> all_to_all_exchange(
        std::size_t group, std::vector<vector_3d> &&prep, std::size_t generation);
    // scatter through one hierarchical communicator per root (slot 2 i: x direction, 2 i + 1: y direction); needs an
    // HPX whose hierarchical communicators support a root site other than 0
    std::vector<std::optional<hpx::collectives::hierarchical_communicator>> hierarchical_scatter_communicators_;
    hpx::future<vector_3d> scatter_exchange(std::size_t slot, std::size_t root, std::size_t site,
                                            std::vector<vector_3d> &&prep, std::size_t generation);
};

}  // namespace hpxfft::fft3D::distributed::loop
#endif  // hpxfft_distributed_loop_3D_H_INCLUDED
