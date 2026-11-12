#include "../../../include/hpxfft/2D/distributed/loop.hpp"

#include "../../../include/hpxfft/util/comm_basename.hpp"
#include "../../../include/hpxfft/util/comm_flag.hpp"
#include "../../../include/hpxfft/util/real_length.hpp"
#include "../../../include/hpxfft/util/transform_guard.hpp"
#include <algorithm>
#include <hpx/hpx_init.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>

// FFT backend
void hpxfft::fft2D::distributed::loop::fft_1d_r2c_inplace(const std::size_t i)
{
    fft_r2c_adapter_.execute(values_vec_.row(i), reinterpret_cast<fftw_complex *>(values_vec_.row(i)));
}

void hpxfft::fft2D::distributed::loop::fft_1d_c2c_inplace(const std::size_t i)
{
    fft_c2c_adapter_.execute(reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)),
                             reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)));
}

// split data for communication
void hpxfft::fft2D::distributed::loop::split_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j)
    {  // std::move same performance
        const std::size_t part = 2 * (y_offsets_[j + 1] - y_offsets_[j]);
        std::copy(values_vec_.row(i) + 2 * y_offsets_[j],
                  values_vec_.row(i) + 2 * y_offsets_[j] + part,
                  values_prep_[j].begin() + i * part);
    }
}

void hpxfft::fft2D::distributed::loop::split_trans_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j)
    {  // std::move same performance
        std::copy(trans_values_vec_.row(i) + j * dim_c_x_part_,
                  trans_values_vec_.row(i) + (j + 1) * dim_c_x_part_,
                  trans_values_prep_[j].begin() + i * dim_c_x_part_);
    }
}

// scatter communication (flat or hierarchical communicators): the locality `root` sends one block to every locality
hpx::future<std::vector<real>> hpxfft::fft2D::distributed::loop::scatter_exchange(
    const std::size_t root, std::vector<std::vector<real>> &&prep, const std::size_t generation)
{
    if (hierarchical_scatter_)
    {
        auto &communicator = *hierarchical_scatter_communicators_[root];
        if (this_locality_ != root)
        {
            return hpx::collectives::scatter_from<std::vector<real>>(
                communicator,
                hpx::collectives::this_site_arg(this_locality_),
                hpx::collectives::generation_arg(generation));
        }
        return hpx::collectives::scatter_to(
            communicator,
            std::move(prep),
            hpx::collectives::this_site_arg(this_locality_),
            hpx::collectives::generation_arg(generation));
    }
    if (this_locality_ != root)
    {
        // receive from other locality
        return hpx::collectives::scatter_from<std::vector<real>>(
            communicators_[root], hpx::collectives::generation_arg(generation));
    }
    // send from this locality
    return hpx::collectives::scatter_to(
        communicators_[root], std::move(prep), hpx::collectives::generation_arg(generation));
}

void hpxfft::fft2D::distributed::loop::communicate_scatter_vec(const std::size_t i)
{
    communication_futures_[i] = scatter_exchange(i, std::move(values_prep_), 1);
}

void hpxfft::fft2D::distributed::loop::communicate_scatter_trans_vec(const std::size_t i)
{
    communication_futures_[i] = scatter_exchange(i, std::move(trans_values_prep_), 2);
}

// all to all communication: direct exchange between the localities, or (reference) routed through a root site
hpx::future<std::vector<std::vector<real>>> hpxfft::fft2D::distributed::loop::all_to_all_exchange(
    std::vector<std::vector<real>> &&prep, const std::size_t generation)
{
    if (routed_)
    {
        return hpx::collectives::all_to_all(
            communicators_[0], std::move(prep), hpx::collectives::generation_arg(generation));
    }
    // all_to_all by name with a threshold of 0 selects the direct exchange also for rows of dynamic size; it needs an
    // explicit generation and at least three sites (otherwise HPX routes the data)
    return hpx::collectives::all_to_all(
        basename_storage_[0].c_str(),
        std::move(prep),
        hpx::collectives::num_sites_arg(num_localities_),
        hpx::collectives::this_site_arg(this_locality_),
        hpx::collectives::generation_arg(generation),
        hpx::collectives::root_site_arg(0),
        hpx::collectives::pairwise_threshold_arg(0));
}

void hpxfft::fft2D::distributed::loop::communicate_all_to_all_vec()
{
    communication_vec_ = all_to_all_exchange(std::move(values_prep_), 1).get();
}

void hpxfft::fft2D::distributed::loop::communicate_all_to_all_trans_vec()
{
    communication_vec_ = all_to_all_exchange(std::move(trans_values_prep_), 2).get();
}

namespace
{
// Tile edge (in complex numbers) of the blocked transposes below: a tile of 16 x 16 complex numbers stays in L1 cache,
// so neither the strided reads nor the strided writes of a transpose thrash the cache.
constexpr std::size_t TILE = 16;
}  // namespace

// transpose after communication: processes the TILE rows k = tile * TILE ... of the output
void hpxfft::fft2D::distributed::loop::transpose_y_to_x(const std::size_t tile, const std::size_t i)
{
    const std::size_t factor_in = dim_c_y_part_;
    const std::size_t dim_input = communication_vec_[i].size() / factor_in;
    // the rows (x) of locality i form the block starting at x = i * dim_input
    const std::size_t x_offset = i * dim_input;
    const real *src = communication_vec_[i].data();
    const std::size_t k_begin = tile * TILE;
    const std::size_t k_end = std::min(k_begin + TILE, n_y_local_);

    for (std::size_t j_begin = 0; j_begin < dim_input; j_begin += TILE)
    {
        const std::size_t j_end = std::min(j_begin + TILE, dim_input);
        for (std::size_t k = k_begin; k < k_end; ++k)
        {
            real *dst = trans_values_vec_.row(k);
            for (std::size_t j = j_begin; j < j_end; ++j)
            {
                const std::size_t index_in = factor_in * j + 2 * k;
                const std::size_t index_out = 2 * (x_offset + j);
                dst[index_out] = src[index_in];
                dst[index_out + 1] = src[index_in + 1];
            }
        }
    }
}

// processes the TILE rows j = tile * TILE ... of the input blocks
void hpxfft::fft2D::distributed::loop::transpose_x_to_y(const std::size_t tile, const std::size_t i)
{
    const std::size_t factor_in = dim_c_x_part_;
    // the rows (y) of locality i form the block starting at y_offsets_[i]; the rows (x) of the received block are the
    // local rows of this locality
    const std::size_t dim_input = n_x_local_;
    const std::size_t y_offset = y_offsets_[i];
    const std::size_t n_y_source = y_offsets_[i + 1] - y_offsets_[i];
    const real *src = communication_vec_[i].data();
    const std::size_t j_begin = tile * TILE;
    const std::size_t j_end = std::min(j_begin + TILE, n_y_source);

    for (std::size_t k_begin = 0; k_begin < dim_input; k_begin += TILE)
    {
        const std::size_t k_end = std::min(k_begin + TILE, dim_input);
        for (std::size_t k = k_begin; k < k_end; ++k)
        {
            real *dst = values_vec_.row(k);
            for (std::size_t j = j_begin; j < j_end; ++j)
            {
                const std::size_t index_in = factor_in * j + 2 * k;
                const std::size_t index_out = 2 * (y_offset + j);
                dst[index_out] = src[index_in];
                dst[index_out + 1] = src[index_in + 1];
            }
        }
    }
}

// 2D FFT algorithm
hpxfft::fft2D::distributed::vector_2d hpxfft::fft2D::distributed::loop::fft_2d_r2c()
{
    // initialize + one transform per object: the transform moves the data and buffers out of the object
    hpxfft::util::begin_transform(ready_, "fft_2d_r2c of the 2D loop");
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // 1d FFT r2c in y-direction
            fft_1d_r2c_inplace(i);
        });
    auto start_first_split = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_vec(i);
        });
    // communication for FFT in second dimension
    auto start_first_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_vec(i);
        }
        // global sychronization
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            communication_vec_[i] = communication_futures_[i].get();
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_vec();
    }
    else
    {
        throw std::logic_error("Communication scheme not specified during initialization");
    }
    auto start_first_trans = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        num_localities_,
        [&](auto i)
        {
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                (n_y_local_ + TILE - 1) / TILE,
                [&](auto k)
                {
                    // blocked transpose from y-direction to x-direction
                    transpose_y_to_x(k, i);
                });
        });
    // second dimension
    auto start_second_fft = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_y_local_,
        [&](auto i)
        {
            // 1D FFT c2c in x-direction
            fft_1d_c2c_inplace(i);
        });
    auto start_second_split = t_.now();
    if (transposed_out_)
    {
        // result stays in (y_local, x) layout: no final transpose
        auto stop_total = t_.now();
        measurements_["total"] = stop_total - start_total;
        measurements_["first_fftw"] = start_first_split - start_total;
        measurements_["first_split"] = start_first_comm - start_first_split;
        measurements_["first_comm"] = start_first_trans - start_first_comm;
        measurements_["first_trans"] = start_second_fft - start_first_trans;
        measurements_["second_fftw"] = start_second_split - start_second_fft;
        measurements_["second_split"] = 0.0;
        measurements_["second_comm"] = 0.0;
        measurements_["second_trans"] = 0.0;
        return std::move(trans_values_vec_);
    }
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_y_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_trans_vec(i);
        });
    // communication to get original data layout
    auto start_second_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_trans_vec(i);
        }
        // global synchronization
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            communication_vec_[i] = communication_futures_[i].get();
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_trans_vec();
    }
    auto start_second_trans = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        num_localities_,
        [&](auto i)
        {
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                (y_offsets_[i + 1] - y_offsets_[i] + TILE - 1) / TILE,
                [&](auto j)
                {
                    // blocked transpose from x-direction to y-direction
                    transpose_x_to_y(j, i);
                });
        });
    auto stop_total = t_.now();

    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_split - start_total;
    measurements_["first_split"] = start_first_comm - start_first_split;
    measurements_["first_comm"] = start_first_trans - start_first_comm;
    measurements_["first_trans"] = start_second_fft - start_first_trans;
    measurements_["second_fftw"] = start_second_split - start_second_fft;
    measurements_["second_split"] = start_second_comm - start_second_split;
    measurements_["second_comm"] = start_second_trans - start_second_comm;
    measurements_["second_trans"] = stop_total - start_second_trans;

    ////////////////////////////////////////////////////////////////
    return std::move(values_vec_);
}

// initialization
void hpxfft::fft2D::distributed::loop::initialize(hpxfft::fft2D::distributed::vector_2d values_vec,
                                                  const std::string COMM_FLAG,
                                                  const std::string PLAN_FLAG,
                                                  const std::size_t ny)
{
    // a failed initialize leaves the object without a transform to run
    ready_ = false;
    // reject an unsupported run flag before any collective operation (the flag is the same on all localities)
    hpxfft::util::check_comm_flag(
        COMM_FLAG, { "scatter", "scatter_hierarchical", "all_to_all", "all_to_all_routed" }, "the 2D loop");
    // unique communicator names of this object (taken at the same point on every locality)
    const std::string basename_prefix = hpxfft::util::unique_basename_prefix("hpxfft/2d/");
    // move data into own structure
    values_vec_ = std::move(values_vec);
    // the global size is derived from the local block: all blocks must have the same shape
    hpxfft::util::check_equal_local_shape({ values_vec_.n_row(), values_vec_.n_col() });
    // locality information
    this_locality_ = hpx::get_locality_id();
    num_localities_ = hpx::get_num_localities(hpx::launch::sync);
    // parameters
    n_x_local_ = values_vec_.n_row();
    dim_c_x_ = n_x_local_ * num_localities_;
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = hpxfft::util::real_length(dim_c_y_, ny, "ny");
    // y is split into blocks of complex numbers (dim_c_y = ny / 2 + 1 of them): dim_c_y / L each, the first
    // dim_c_y % L localities hold one more
    if (dim_c_y_ < num_localities_)
    {
        throw std::invalid_argument(
            "Number of complex values in y (ny / 2 + 1) is smaller than the number of localities");
    }
    y_offsets_ = hpxfft::util::block_offsets(dim_c_y_, num_localities_);
    n_y_local_ = y_offsets_[this_locality_ + 1] - y_offsets_[this_locality_];
    // length of a row of the block of y received from every other locality, in doubles
    dim_c_y_part_ = 2 * n_y_local_;
    dim_c_x_part_ = 2 * dim_c_x_ / num_localities_;
    // resize other data structures
    trans_values_vec_ = std::move(hpxfft::fft2D::distributed::vector_2d(n_y_local_, 2 * dim_c_x_));
    values_prep_.resize(num_localities_);
    trans_values_prep_.resize(num_localities_);
    for (std::size_t i = 0; i < num_localities_; ++i)
    {
        values_prep_[i].resize(n_x_local_ * 2 * (y_offsets_[i + 1] - y_offsets_[i]));
        trans_values_prep_[i].resize(n_y_local_ * dim_c_x_part_);
    }
    // create FFTW plans
    // r2c in y-direction
    fft_r2c_adapter_ = hpxfft::util::fftw_adapter::r2c_1d();
    fft_r2c_adapter_.plan(
        dim_r_y_, PLAN_FLAG, trans_values_vec_.row(0), reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)));
    // c2c in x-direction
    fft_c2c_adapter_ = hpxfft::util::fftw_adapter::c2c_1d();
    fft_c2c_adapter_.plan(dim_c_x_,
                          PLAN_FLAG,
                          reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
                          reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
                          hpxfft::util::fftw_adapter::direction::forward);
    // communication specific initialization (a new initialize must not keep the variant of the previous one)
    routed_ = false;
    hierarchical_scatter_ = false;
    COMM_FLAG_ = COMM_FLAG;
    if (COMM_FLAG == "scatter_hierarchical")
    {
        // scatter through hierarchical (tree) communicators, one per root
        hierarchical_scatter_ = true;
        COMM_FLAG_ = "scatter";
    }
    else if (COMM_FLAG == "all_to_all_routed")
    {
        // reference all_to_all: the data is routed through a root site (single flat communicator)
        routed_ = true;
        COMM_FLAG_ = "all_to_all";
    }
    // the pairwise exchange needs at least three sites; with fewer the routed all_to_all is used
    if (COMM_FLAG_ == "all_to_all" && num_localities_ < 3)
    {
        routed_ = true;
    }
    if (COMM_FLAG_ == "scatter")
    {
        communication_vec_.resize(num_localities_);
        communication_futures_.resize(num_localities_);
        // setup communicators
        basename_storage_.resize(num_localities_);
        if (hierarchical_scatter_)
        {
            // communicator i is rooted at locality i; below the flat fallback threshold each of them is a single
            // flat communicator
            hierarchical_scatter_communicators_.resize(num_localities_);
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                basename_storage_[i] = basename_prefix + std::to_string(i);
                hierarchical_scatter_communicators_[i].emplace(hpx::collectives::create_hierarchical_communicator(
                    basename_storage_[i].c_str(),
                    hpx::collectives::num_sites_arg(num_localities_),
                    hpx::collectives::this_site_arg(this_locality_),
                    hpx::collectives::arity_arg(hierarchical_arity_),
                    hpx::collectives::generation_arg(),
                    hpx::collectives::root_site_arg(i),
                    hpx::collectives::flat_fallback_threshold_arg(hierarchical_flat_threshold_)));
            }
            // allow one transform
            ready_ = true;
            return;
        }
        communicators_.resize(num_localities_);
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            basename_storage_[i] = basename_prefix + std::to_string(i);
            // Host the communicator on the locality that scatters through it
            // (root_site = i). With the default root site (0) every scatter is
            // staged on locality 0 first: the data crosses the network twice
            // and locality 0 becomes a hub for all of them.
            communicators_[i] = std::move(hpx::collectives::create_communicator(
                basename_storage_[i].c_str(),
                hpx::collectives::num_sites_arg(num_localities_),
                hpx::collectives::this_site_arg(this_locality_),
                hpx::collectives::generation_arg(),
                hpx::collectives::root_site_arg(i)));
        }
        // The communicators are futures: on HPX master, scatter_to/scatter_from read num_sites from the
        // communicator's info, which is only attached once the communicator has been created/found. Make sure all
        // communicators are ready before they are used (otherwise num_sites is unset and scatter_to fails with
        // "the number of values to scatter must be equal to the number of participating sites").
        for (auto &communicator : communicators_)
        {
            communicator.wait();
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        communication_vec_.resize(1);
        // setup communicators
        basename_storage_.resize(1);
        communicators_.resize(1);
        basename_storage_[0] = basename_prefix + "0";
        if (routed_)
        {
            communicators_[0] = std::move(hpx::collectives::create_communicator(
                basename_storage_[0].c_str(),
                hpx::collectives::num_sites_arg(num_localities_),
                hpx::collectives::this_site_arg(this_locality_)));
        }
    }
    else
    {
        // unreachable: the flag was checked at the start of initialize
        throw std::logic_error("Unsupported communication flag " + COMM_FLAG);
    }
    // allow one transform
    ready_ = true;
}

// helpers
real hpxfft::fft2D::distributed::loop::get_measurement(std::string name) { return measurements_[name]; }
