#include <algorithm>
#include "../../../include/hpxfft/3D/distributed/loop.hpp"

void hpxfft::fft3D::distributed::loop::slab::initialize(
    vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG)
{
    this_locality_ = hpx::get_locality_id();
    // move data into own structure
    values_vec_ = std::move(values_vec);
    // locality information
    num_localities_ = hpx::get_num_localities(hpx::launch::sync);
    // parameters
    n_x_local_ = values_vec_.n_x();
    n_y_local_ = values_vec_.n_y() / num_localities_;
    n_z_local_ = values_vec_.n_z() / (2 * num_localities_);
    dim_c_y_ = values_vec_.n_y();
    dim_c_z_ = values_vec_.n_z() / 2;
    dim_r_z_ = 2 * dim_c_z_ - 2;
    dim_c_x_ = n_x_local_ * num_localities_;
    dim_c_z_part_ = 2 * dim_c_z_ / num_localities_;
    dim_c_y_part_ = 2 * dim_c_y_ / num_localities_;
    dim_c_x_part_ = 2 * dim_c_x_ / num_localities_;

    // z is never distributed in the slab decomposition, only y has to be divisible
    if (values_vec_.n_y() % num_localities_ != 0)
    {
        throw std::invalid_argument("Dimension y of the 3D Matrix is not divisable by number of localities");
    }

    // resize other data structures
    permuted_vec_ = std::move(vector_3d(n_x_local_, dim_c_z_, 2 * dim_c_y_));
    values_prep_.resize(num_localities_);
    permuted_values_prep_.resize(num_localities_);
    for (std::size_t i = 0; i < num_localities_; ++i)
    {
        permuted_values_prep_[i] = vector_3d(n_x_local_, dim_c_z_, 2 * n_y_local_, vector_3d::uninitialized);
        values_prep_[i] = vector_3d(n_y_local_, dim_c_z_, 2 * n_x_local_, vector_3d::uninitialized);
    }
    // create FFTW plans
    // r2c in z-direction
    fftw_r2c_adapter_dir_z_ = hpxfft::util::fftw_adapter::r2c_1d();
    fftw_r2c_adapter_dir_z_.plan(
        dim_r_z_, PLAN_FLAG, permuted_vec_.slice_yz(0), reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)));
    // c2c in y-direction
    fftw_c2c_adapter_dir_y_ = hpxfft::util::fftw_adapter::c2c_1d();
    fftw_c2c_adapter_dir_y_.plan(
        dim_c_y_,
        PLAN_FLAG,
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        hpxfft::util::fftw_adapter::direction::forward);
    // c2c in x-direction
    fftw_c2c_adapter_dir_x_ = hpxfft::util::fftw_adapter::c2c_1d();
    fftw_c2c_adapter_dir_x_.plan(
        dim_c_x_,
        PLAN_FLAG,
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        reinterpret_cast<fftw_complex *>(permuted_vec_.slice_yz(0)),
        hpxfft::util::fftw_adapter::direction::forward);
    // communication specific initialization
    parse_comm_flag(COMM_FLAG, num_localities_);
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
                basename_storage_[i] = std::to_string(i);
                hierarchical_scatter_communicators_[i].emplace(hpx::collectives::create_hierarchical_communicator(
                    basename_storage_[i].c_str(),
                    hpx::collectives::num_sites_arg(num_localities_),
                    hpx::collectives::this_site_arg(this_locality_),
                    hpx::collectives::arity_arg(hierarchical_arity_),
                    hpx::collectives::generation_arg(),
                    hpx::collectives::root_site_arg(i),
                    hpx::collectives::flat_fallback_threshold_arg(hierarchical_flat_threshold_)));
            }
            return;
        }
        communicators_.resize(num_localities_);
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            basename_storage_[i] = std::to_string(i);
            // Host the communicator on the locality that scatters through it
            // (root_site = i); with the default root site (0) every scatter
            // is staged on locality 0 first.
            communicators_[i] = std::move(hpx::collectives::create_communicator(
                basename_storage_[i].c_str(),
                hpx::collectives::num_sites_arg(num_localities_),
                hpx::collectives::this_site_arg(this_locality_),
                hpx::collectives::generation_arg(),
                hpx::collectives::root_site_arg(i)));
        }
        // The communicators are futures: scatter_to/scatter_from read num_sites from the communicator's info, which
        // is only attached once it has been created/found. Wait for all of them before use.
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
        basename_storage_[0] = std::to_string(0);
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
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
}

// scatter communitcation (flat or hierarchical communicators): the locality `root` sends one block to every locality
hpx::future<hpxfft::fft3D::distributed::vector_3d> hpxfft::fft3D::distributed::loop::slab::scatter_exchange(
    const std::size_t root, std::vector<vector_3d> &&prep, std::size_t generation)
{
    if (hierarchical_scatter_)
    {
        auto &communicator = *hierarchical_scatter_communicators_[root];
        if (this_locality_ != root)
        {
            // receive from the root
            return hpx::collectives::scatter_from<vector_3d>(
                communicator,
                hpx::collectives::this_site_arg(this_locality_),
                hpx::collectives::generation_arg(generation));
        }
        // send from this locality (the root of the communicator)
        return hpx::collectives::scatter_to(communicator,
                                            std::move(prep),
                                            hpx::collectives::this_site_arg(this_locality_),
                                            hpx::collectives::generation_arg(generation));
    }
    if (this_locality_ != root)
    {
        // receive from other locality
        return hpx::collectives::scatter_from<vector_3d>(
            communicators_[root], hpx::collectives::generation_arg(generation));
    }
    // send from this locality
    return hpx::collectives::scatter_to(
        communicators_[root], std::move(prep), hpx::collectives::generation_arg(generation));
}

void hpxfft::fft3D::distributed::loop::slab::communicate_scatter_vec(const std::size_t i)
{
    communication_futures_[i] = scatter_exchange(i, std::move(values_prep_), 2);
}

void hpxfft::fft3D::distributed::loop::slab::communicate_scatter_permuted_vec(const std::size_t i)
{
    communication_futures_[i] = scatter_exchange(i, std::move(permuted_values_prep_), 1);
}

// all to all communication: direct exchange between the localities, or (reference) routed through a root site
hpx::future<std::vector<hpxfft::fft3D::distributed::vector_3d>>
hpxfft::fft3D::distributed::loop::slab::all_to_all_exchange(std::vector<vector_3d> &&prep, std::size_t generation)
{
    if (routed_)
    {
        return hpx::collectives::all_to_all(
            communicators_[0], std::move(prep), hpx::collectives::generation_arg(generation));
    }
    // all_to_all by name with a threshold of 0 selects the direct exchange also for rows of dynamic size; it needs an
    // explicit generation and at least three sites (otherwise HPX routes the data)
    return hpx::collectives::all_to_all(basename_storage_[0].c_str(),
                                        std::move(prep),
                                        hpx::collectives::num_sites_arg(num_localities_),
                                        hpx::collectives::this_site_arg(this_locality_),
                                        hpx::collectives::generation_arg(generation),
                                        hpx::collectives::root_site_arg(0),
                                        hpx::collectives::pairwise_threshold_arg(0));
}

void hpxfft::fft3D::distributed::loop::slab::communicate_all_to_all_vec()
{
    communication_vec_ = all_to_all_exchange(std::move(values_prep_), 2).get();
}

void hpxfft::fft3D::distributed::loop::slab::communicate_all_to_all_permuted_vec()
{
    communication_vec_ = all_to_all_exchange(std::move(permuted_values_prep_), 1).get();
}

namespace
{
// Tile edge (in complex numbers) of the blocked transposes below: a tile of 16 x 16 complex numbers stays in L1 cache,
// so neither the strided reads nor the strided writes of a transpose thrash the cache.
constexpr std::size_t TILE = 16;
}  // namespace

// permute data for FFT in y-direction (only local data)
void hpxfft::fft3D::distributed::loop::slab::permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i)
{
    const std::size_t n_y = values_vec_.n_y();
    const std::size_t n_z = values_vec_.n_z();
    const std::size_t n_z_c = n_z / 2;
    const double *src = values_vec_.slice_yz(slice_x);
    double *dst = permuted_vec_.slice_yz(slice_x);
    const std::size_t dst_row = permuted_vec_.n_z();

    // blocked transpose of the (y, z) matrix of complex numbers
    for (std::size_t yb = 0; yb < n_y; yb += TILE)
    {
        const std::size_t ye = std::min(yb + TILE, n_y);
        for (std::size_t zb = 0; zb < n_z_c; zb += TILE)
        {
            const std::size_t ze = std::min(zb + TILE, n_z_c);
            for (std::size_t y = yb; y < ye; ++y)
            {
                for (std::size_t z = zb; z < ze; ++z)
                {
                    dst[z * dst_row + 2 * y] = src[y * n_z + 2 * z];
                    dst[z * dst_row + 2 * y + 1] = src[y * n_z + 2 * z + 1];
                }
            }
        }
    }
}

// permute data after communication
void hpxfft::fft3D::distributed::loop::slab::permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i)
{
    const std::size_t part = values_vec_.n_z() / num_localities_;
    const std::size_t offset = part * i;
    const std::size_t n_0 = values_vec_.n_x();  // local y
    const std::size_t n_1 = part / 2;           // x of the source locality
    const vector_3d &comm = communication_vec_[i];
    const double *src = comm.data();
    const std::size_t src_k = comm.n_z();
    const std::size_t src_x = comm.n_y() * src_k;
    double *dst = values_vec_.data();
    const std::size_t dst_k = values_vec_.n_z();
    const std::size_t dst_y = values_vec_.n_y() * dst_k;

    for (std::size_t zb = 0; zb < n_0; zb += TILE)
    {
        const std::size_t ze = std::min(zb + TILE, n_0);
        for (std::size_t xb = 0; xb < n_1; xb += TILE)
        {
            const std::size_t xe = std::min(xb + TILE, n_1);
            for (std::size_t z = zb; z < ze; ++z)
            {
                for (std::size_t x = xb; x < xe; ++x)
                {
                    dst[z * dst_y + slice_y * dst_k + offset + 2 * x] = src[x * src_x + slice_y * src_k + 2 * z];
                    dst[z * dst_y + slice_y * dst_k + offset + 2 * x + 1] = src[x * src_x + slice_y * src_k + 2 * z + 1];
                }
            }
        }
    }
}

void hpxfft::fft3D::distributed::loop::slab::permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i)
{
    const std::size_t part = permuted_vec_.n_y() / num_localities_;
    const std::size_t offset = part * i;
    const std::size_t n_0 = permuted_vec_.n_x();      // x
    const std::size_t n_1 = permuted_vec_.n_z() / 2;  // z (complex)
    const vector_3d &comm = communication_vec_[i];
    const double *src = comm.data();
    const std::size_t src_x = comm.n_y() * comm.n_z();
    const std::size_t src_y = comm.n_z();
    double *dst = permuted_vec_.data();
    const std::size_t dst_x = permuted_vec_.n_y() * permuted_vec_.n_z();
    const std::size_t dst_y = permuted_vec_.n_z();

    for (std::size_t zb = 0; zb < n_0; zb += TILE)
    {
        const std::size_t ze = std::min(zb + TILE, n_0);
        for (std::size_t yb = 0; yb < n_1; yb += TILE)
        {
            const std::size_t ye = std::min(yb + TILE, n_1);
            for (std::size_t z = zb; z < ze; ++z)
            {
                for (std::size_t y = yb; y < ye; ++y)
                {
                    dst[z * dst_x + (offset + slice_x) * dst_y + 2 * y] = src[slice_x * src_x + y * src_y + 2 * z];
                    dst[z * dst_x + (offset + slice_x) * dst_y + 2 * y + 1] =
                        src[slice_x * src_x + y * src_y + 2 * z + 1];
                }
            }
        }
    }
}

void hpxfft::fft3D::distributed::loop::slab::split_vec(const std::size_t x, const std::size_t dummy)
{
    const std::size_t part = values_vec_.n_z() / num_localities_;
    for (std::size_t j = 0; j < num_localities_; j++)
    {
        for (std::size_t y = 0; y < values_prep_[j].n_y(); ++y)
        {
            // contiguous run in z: plain copy
            std::copy_n(&values_vec_(x, y, part * j), part, &values_prep_[j](x, y, 0));
        }
    }
}

void hpxfft::fft3D::distributed::loop::slab::split_permuted_vec(const std::size_t x, const std::size_t dummy)
{
    const std::size_t part = permuted_vec_.n_z() / num_localities_;
    for (std::size_t j = 0; j < num_localities_; j++)
    {
        for (std::size_t y = 0; y < permuted_vec_.n_y(); ++y)
        {
            std::copy_n(&permuted_vec_(x, y, part * j), part, &permuted_values_prep_[j](x, y, 0));
        }
    }
}

// 3D FFT algorithm
hpxfft::fft3D::distributed::vector_3d hpxfft::fft3D::distributed::loop::slab::fft_3d_r2c()
{
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    // first loop over x
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // second loop over y
            for (std::size_t j = 0; j < dim_c_y_; j++)
            {
                // fft for z direction
                fft_1d_r2c_inplace(i, j);
            }
        });
    // first permute and second fft can all happen locally
    auto start_first_permute = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // permute second and third dimension x-y-z -> x-z-y
            permute_distributed_x_z_y(i, 0);
        });
    // dimesions are now x z y
    auto start_second_fft = t_.now();
    // first loop over x
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // second loop over z
            for (std::size_t j = 0; j < dim_c_z_; j++)
            {
                // fft for y direction
                fft_1d_c2c_y_inplace(i, j);
            }
        });
    auto start_first_split = t_.now();
    // splitting along the third dimesion (y)
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_permuted_vec(i, 0);
        });
    // communication for FFT in third dimension
    auto start_first_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_permuted_vec(i);
        }
        // global sychronization
        if (scatter_sync_)
        {
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_permuted_vec();
    }
    else
    {
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
    auto start_second_permute = t_.now();
    values_vec_.rearrange(n_y_local_, dim_c_z_, 2 * dim_c_x_);
    // for every locality
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        num_localities_,
        [&](auto i)
        {
            if (!scatter_sync_)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
            // for every "slice" in the second dimension (z)
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                dim_c_z_,
                [&](auto k)
                {
                    // permute first and third dimension x-z-y -> y-z-x
                    permute_distributed_z_y_x(k, i);
                });
        });
    // third fft
    auto start_third_fft = t_.now();
    // for every (local) y
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_y_local_,
        [&](auto i)
        {
            // for every z
            for (std::size_t j = 0; j < dim_c_z_; ++j)
            {
                // 1D FFT c2c in x-direction
                fft_1d_c2c_x_inplace(i, j);
            }
        });
    auto start_second_split = t_.now();
    if (transposed_out_)
    {
        // result stays in y-z-x layout (y distributed): no final transpose
        auto stop_total = t_.now();
        measurements_["total"] = stop_total - start_total;
        measurements_["first_fftw"] = start_first_permute - start_total;
        measurements_["first_permute"] = start_second_fft - start_first_permute;
        measurements_["second_fftw"] = start_first_split - start_second_fft;
        measurements_["first_split"] = start_first_comm - start_first_split;
        measurements_["first_comm"] = start_second_permute - start_first_comm;
        measurements_["second_permute"] = start_third_fft - start_second_permute;
        measurements_["third_fftw"] = start_second_split - start_third_fft;
        measurements_["second_split"] = 0.0;
        measurements_["second_comm"] = 0.0;
        measurements_["third_permute"] = 0.0;
        return std::move(values_vec_);
    }
    // splitting matrix for final communication
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_y_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_vec(i, 0);
        });
    // communication to get original data layout
    auto start_second_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_vec(i);
        }
        // global synchronization
        if (scatter_sync_)
        {
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_vec();
    }
    permuted_vec_.rearrange(n_x_local_, dim_c_y_, 2 * dim_c_z_);
    auto start_third_permute = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        num_localities_,
        [&](auto i)
        {
            if (!scatter_sync_)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                n_y_local_,
                [&](auto j)
                {
                    // permute whole matrix y-z-x -> x-y-z
                    permute_distributed_z_x_y(j, i);
                });
        });
    auto stop_total = t_.now();

    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_permute - start_total;
    measurements_["first_permute"] = start_second_fft - start_first_permute;
    measurements_["second_fftw"] = start_first_split - start_second_fft;
    measurements_["first_split"] = start_first_comm - start_first_split;
    measurements_["first_comm"] = start_second_permute - start_first_comm;
    measurements_["second_permute"] = start_third_fft - start_second_permute;
    measurements_["third_fftw"] = start_second_split - start_third_fft;
    measurements_["second_split"] = start_second_comm - start_second_split;
    measurements_["second_comm"] = start_third_permute - start_second_comm;
    measurements_["third_permute"] = stop_total - start_third_permute;

    ////////////////////////////////////////////////////////////////
    return std::move(permuted_vec_);
}
