#include "../../../include/hpxfft/3D/distributed/loop.hpp"
#include "../../../include/hpxfft/util/comm_basename.hpp"
#include "../../../include/hpxfft/util/real_length.hpp"
#include <algorithm>
#include <hpx/runtime_distributed.hpp>  // for hpx::get_locality_id, hpx::get_num_localities

void hpxfft::fft3D::distributed::loop::pencil::initialize(
    vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG, const std::size_t nz)
{
    // reject an unsupported run flag before any collective operation
    // a failed initialize leaves the object without a transform to run
    ready_ = false;
    check_comm_flag(COMM_FLAG, "the pencil decomposition");
    // unique communicator names of this object (taken at the same point on every locality)
    const std::string basename_prefix = hpxfft::util::unique_basename_prefix("hpxfft/pencil/");
    // move data into own structure
    values_vec_ = std::move(values_vec);
    // the global size is derived from the local block: all blocks must have the same shape
    hpxfft::util::check_equal_local_shape({ values_vec_.n_x(), values_vec_.n_y(), values_vec_.n_z() });
    // locality information
    this_locality_ = hpx::get_locality_id();
    num_localities_ = hpx::get_num_localities(hpx::launch::sync);
    localities_per_dir_ = std::sqrt(num_localities_);
    // parameters
    n_x_local_ = values_vec_.n_x();
    n_y_local_ = values_vec_.n_y();
    dim_c_x_ = n_x_local_ * localities_per_dir_;
    dim_c_y_ = n_y_local_ * localities_per_dir_;
    dim_c_z_ = values_vec_.n_z() / 2;
    dim_r_z_ = hpxfft::util::real_length(dim_c_z_, nz, "nz");
    dim_c_z_part_ = 2 * dim_c_z_ / localities_per_dir_;
    dim_c_y_part_ = 2 * dim_c_y_ / localities_per_dir_;
    dim_c_x_part_ = 2 * dim_c_x_ / localities_per_dir_;
    if (localities_per_dir_ * localities_per_dir_ != num_localities_)
    {
        throw std::invalid_argument("Number of localities must be a square number using pencil decomposition");
    }
    // z is split into blocks of complex numbers (dim_c_z = nz / 2 + 1 of them): dim_c_z / P each, the first
    // dim_c_z % P blocks hold one more
    if (dim_c_z_ < localities_per_dir_)
    {
        throw std::invalid_argument("Number of complex values in z (nz / 2 + 1) is smaller than the number of "
                                    "localities per direction");
    }
    pos_dir_x_ = this_locality_ / localities_per_dir_;
    pos_dir_y_ = this_locality_ % localities_per_dir_;
    z_offsets_ = hpxfft::util::block_offsets(dim_c_z_, localities_per_dir_);
    n_z_local_ = z_offsets_[pos_dir_y_ + 1] - z_offsets_[pos_dir_y_];
    // resize other data structures
    permuted_vec_ = std::move(vector_3d(n_x_local_, n_z_local_, 2 * dim_c_y_));
    values_prep_.resize(localities_per_dir_);
    permuted_values_prep_.resize(localities_per_dir_);
    values_prep2_.resize(localities_per_dir_);
    permuted_values_prep2_.resize(localities_per_dir_);
    for (std::size_t i = 0; i < localities_per_dir_; ++i)
    {
        permuted_values_prep_[i] = vector_3d(n_x_local_, n_z_local_, 2 * n_y_local_);
        values_prep_[i] = vector_3d(n_x_local_, n_y_local_, 2 * (z_offsets_[i + 1] - z_offsets_[i]));
        permuted_values_prep2_[i] = vector_3d(n_x_local_, n_z_local_, 2 * n_y_local_);
        values_prep2_[i] = vector_3d(n_y_local_, n_z_local_, 2 * n_x_local_);
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
    parse_comm_flag(COMM_FLAG, localities_per_dir_);
    // first generation of the exchanges (set before the communicator setup, which may return early)
    generation_counterx_ = 1;
    generation_countery_ = 1;
    if (COMM_FLAG_ == "scatter")
    {
        communication_vec_.resize(localities_per_dir_);
        communication_futures_.resize(localities_per_dir_);
        // setup communicators
        basename_storage_.resize(2 * localities_per_dir_);
        communicators_.resize(2 * localities_per_dir_);
        if (hierarchical_scatter_)
        {
            // communicator i of a group is rooted at its site i; below the flat fallback threshold each of them is a
            // single flat communicator
            hierarchical_scatter_communicators_.resize(2 * localities_per_dir_);
            for (std::size_t i = 0; i < localities_per_dir_; ++i)
            {
                basename_storage_[2 * i] =
                    basename_prefix + "x/" + std::to_string(pos_dir_y_) + "/" + std::to_string(i);
                basename_storage_[2 * i + 1] =
                    basename_prefix + "y/" + std::to_string(pos_dir_x_) + "/" + std::to_string(i);
                hierarchical_scatter_communicators_[2 * i].emplace(hpx::collectives::create_hierarchical_communicator(
                    basename_storage_[2 * i].c_str(),
                    hpx::collectives::num_sites_arg(localities_per_dir_),
                    hpx::collectives::this_site_arg(pos_dir_x_),
                    hpx::collectives::arity_arg(hierarchical_arity_),
                    hpx::collectives::generation_arg(),
                    hpx::collectives::root_site_arg(i),
                    hpx::collectives::flat_fallback_threshold_arg(hierarchical_flat_threshold_)));
                hierarchical_scatter_communicators_[2 * i + 1].emplace(
                    hpx::collectives::create_hierarchical_communicator(
                        basename_storage_[2 * i + 1].c_str(),
                        hpx::collectives::num_sites_arg(localities_per_dir_),
                        hpx::collectives::this_site_arg(pos_dir_y_),
                        hpx::collectives::arity_arg(hierarchical_arity_),
                        hpx::collectives::generation_arg(),
                        hpx::collectives::root_site_arg(i),
                        hpx::collectives::flat_fallback_threshold_arg(hierarchical_flat_threshold_)));
            }
            // allow one transform
            ready_ = true;
            return;
        }
        for (std::size_t i = 0; i < localities_per_dir_; ++i)
        {
            basename_storage_[2 * i] = basename_prefix + "x/" + std::to_string(pos_dir_y_) + "/" + std::to_string(i);
            basename_storage_[2 * i + 1] =
                basename_prefix + "y/" + std::to_string(pos_dir_x_) + "/" + std::to_string(i);
            // Host communicator i on the site that scatters through it
            // (root_site = i); with the default root site (0) every scatter
            // is staged on site 0 of the group first.
            communicators_[2 * i] = std::move(hpx::collectives::create_communicator(
                basename_storage_[2 * i].c_str(),
                hpx::collectives::num_sites_arg(localities_per_dir_),
                hpx::collectives::this_site_arg(pos_dir_x_),
                hpx::collectives::generation_arg(),
                hpx::collectives::root_site_arg(i)));
            communicators_[2 * i + 1] = std::move(hpx::collectives::create_communicator(
                basename_storage_[2 * i + 1].c_str(),
                hpx::collectives::num_sites_arg(localities_per_dir_),
                hpx::collectives::this_site_arg(pos_dir_y_),
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
        communication_vec_.resize(localities_per_dir_);
        // setup communicators
        basename_storage_.resize(2);
        communicators_.resize(2);
        basename_storage_[0] = basename_prefix + "x/" + std::to_string(pos_dir_y_);
        basename_storage_[1] = basename_prefix + "y/" + std::to_string(pos_dir_x_);
        if (routed_)
        {
            communicators_[0] = std::move(hpx::collectives::create_communicator(
                basename_storage_[0].c_str(),
                hpx::collectives::num_sites_arg(localities_per_dir_),
                hpx::collectives::this_site_arg(pos_dir_x_)));
            communicators_[1] = std::move(hpx::collectives::create_communicator(
                basename_storage_[1].c_str(),
                hpx::collectives::num_sites_arg(localities_per_dir_),
                hpx::collectives::this_site_arg(pos_dir_y_)));
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

// scatter communitcation
void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_vec_x(const std::size_t i)
{
    communication_futures_[i] = scatter_exchange(2 * i, i, pos_dir_x_, std::move(values_prep2_), generation_counterx_);
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_vec_y(const std::size_t i)
{
    communication_futures_[i] =
        scatter_exchange(2 * i + 1, i, pos_dir_y_, std::move(values_prep_), generation_countery_);
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_permuted_vec_x(const std::size_t i)
{
    communication_futures_[i] =
        scatter_exchange(2 * i, i, pos_dir_x_, std::move(permuted_values_prep_), generation_counterx_);
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_permuted_vec_y(const std::size_t i)
{
    communication_futures_[i] =
        scatter_exchange(2 * i + 1, i, pos_dir_y_, std::move(permuted_values_prep2_), generation_countery_);
}

// scatter communication (flat or hierarchical communicators): the locality `root` of a group sends one block to every
// locality of the group; `slot` selects the communicator (2 root: x direction, 2 root + 1: y direction)
hpx::future<hpxfft::fft3D::distributed::vector_3d> hpxfft::fft3D::distributed::loop::pencil::scatter_exchange(
    const std::size_t slot,
    const std::size_t root,
    const std::size_t site,
    std::vector<vector_3d> &&prep,
    const std::size_t generation)
{
    if (hierarchical_scatter_)
    {
        auto &communicator = *hierarchical_scatter_communicators_[slot];
        if (site != root)
        {
            return hpx::collectives::scatter_from<vector_3d>(
                communicator, hpx::collectives::this_site_arg(site), hpx::collectives::generation_arg(generation));
        }
        return hpx::collectives::scatter_to(
            communicator,
            std::move(prep),
            hpx::collectives::this_site_arg(site),
            hpx::collectives::generation_arg(generation));
    }
    if (site != root)
    {
        // receive from other locality
        return hpx::collectives::scatter_from<vector_3d>(
            communicators_[slot], hpx::collectives::this_site_arg(site), hpx::collectives::generation_arg(generation));
    }
    // send from this locality
    return hpx::collectives::scatter_to(
        communicators_[slot],
        std::move(prep),
        hpx::collectives::this_site_arg(site),
        hpx::collectives::generation_arg(generation));
}

// all to all communication of a group (0: x direction, 1: y direction): direct exchange between the localities of the
// group, or (reference) routed through a root site
hpx::future<std::vector<hpxfft::fft3D::distributed::vector_3d>>
hpxfft::fft3D::distributed::loop::pencil::all_to_all_exchange(
    const std::size_t group, std::vector<vector_3d> &&prep, const std::size_t generation)
{
    const std::size_t site = (group == 0) ? pos_dir_x_ : pos_dir_y_;
    if (routed_)
    {
        return hpx::collectives::all_to_all(
            communicators_[group],
            std::move(prep),
            hpx::collectives::this_site_arg(site),
            hpx::collectives::generation_arg(generation));
    }
    // all_to_all by name with a threshold of 0 selects the direct exchange also for rows of dynamic size; it needs an
    // explicit generation and at least three sites in the group (otherwise HPX routes the data)
    return hpx::collectives::all_to_all(
        basename_storage_[group].c_str(),
        std::move(prep),
        hpx::collectives::num_sites_arg(localities_per_dir_),
        hpx::collectives::this_site_arg(site),
        hpx::collectives::generation_arg(generation),
        hpx::collectives::root_site_arg(0),
        hpx::collectives::pairwise_threshold_arg(0));
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_vec_x()
{
    communication_vec_ = all_to_all_exchange(0, std::move(values_prep2_), generation_counterx_).get();
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_vec_y()
{
    communication_vec_ = all_to_all_exchange(1, std::move(values_prep_), generation_countery_).get();
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_permuted_vec_x()
{
    communication_vec_ = all_to_all_exchange(0, std::move(permuted_values_prep_), generation_counterx_).get();
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_permuted_vec_y()
{
    communication_vec_ = all_to_all_exchange(1, std::move(permuted_values_prep2_), generation_countery_).get();
}

namespace
{
// Tile edge (in complex numbers) of the blocked transposes below: a tile of 16 x 16 complex numbers stays in L1 cache,
// so neither the strided reads nor the strided writes of a transpose thrash the cache.
constexpr std::size_t TILE = 16;

using vector_3d = hpxfft::fft3D::distributed::vector_3d;

// dst(slice, z, offset + 2 y) = src(slice, y, 2 z) for y < n_1, z < dst.n_y() (complex numbers)
void transpose_slice_yz(
    vector_3d &dst, const vector_3d &src, const std::size_t slice, const std::size_t offset, const std::size_t n_1)
{
    const std::size_t n_0 = dst.n_y();
    const double *s = src.data();
    const std::size_t src_x = src.n_y() * src.n_z();
    const std::size_t src_y = src.n_z();
    double *d = dst.data();
    const std::size_t dst_x = dst.n_y() * dst.n_z();
    const std::size_t dst_y = dst.n_z();

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
                    d[slice * dst_x + z * dst_y + offset + 2 * y] = s[slice * src_x + y * src_y + 2 * z];
                    d[slice * dst_x + z * dst_y + offset + 2 * y + 1] = s[slice * src_x + y * src_y + 2 * z + 1];
                }
            }
        }
    }
}

// dst(z, slice, offset + 2 x) = src(x, slice, 2 z) for x < n_1, z < dst.n_x() (complex numbers)
void transpose_slice_xz(
    vector_3d &dst, const vector_3d &src, const std::size_t slice, const std::size_t offset, const std::size_t n_1)
{
    const std::size_t n_0 = dst.n_x();
    const double *s = src.data();
    const std::size_t src_x = src.n_y() * src.n_z();
    const std::size_t src_y = src.n_z();
    double *d = dst.data();
    const std::size_t dst_x = dst.n_y() * dst.n_z();
    const std::size_t dst_y = dst.n_z();

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
                    d[z * dst_x + slice * dst_y + offset + 2 * x] = s[x * src_x + slice * src_y + 2 * z];
                    d[z * dst_x + slice * dst_y + offset + 2 * x + 1] = s[x * src_x + slice * src_y + 2 * z + 1];
                }
            }
        }
    }
}
}  // namespace

// permute data for FFT in y-direction
void hpxfft::fft3D::distributed::loop::pencil::permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i)
{
    if (generation_countery_ == 2)
    {
        const std::size_t part = permuted_vec_.n_z() / localities_per_dir_;
        transpose_slice_yz(permuted_vec_, communication_vec_[i], slice_x, part * i, part / 2);
    }
    else
    {
        // the z block of locality i of the group
        transpose_slice_yz(
            values_vec_, communication_vec_[i], slice_x, 2 * z_offsets_[i], z_offsets_[i + 1] - z_offsets_[i]);
    }
}

// permute data after communication
void hpxfft::fft3D::distributed::loop::pencil::permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i)
{
    if (generation_counterx_ == 2)
    {
        const std::size_t part = values_vec_.n_z() / localities_per_dir_;
        transpose_slice_xz(values_vec_, communication_vec_[i], slice_y, part * i, part / 2);
    }
    else
    {
        const std::size_t part = permuted_vec_.n_z() / localities_per_dir_;
        transpose_slice_xz(permuted_vec_, communication_vec_[i], slice_y, part * i, part / 2);
    }
}

void hpxfft::fft3D::distributed::loop::pencil::permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i)
{
    // Currently not used due to communication setup
    /*
    const std::size_t part = permuted_vec_.n_y()/localities_per_dir_;
    const std::size_t offset = part * i;
    for(std::size_t index_z = 0; index_z < permuted_vec_.n_x(); index_z++)
    {
        for(std::size_t index_y = 0; index_y < permuted_vec_.n_z()/2; index_y++)
        {
            permuted_vec_(index_z,offset + slice_x, 2 * index_y) = communication_vec_[i](slice_x, index_y, 2 * index_z);
            permuted_vec_(index_z,offset + slice_x, 2 * index_y + 1) = communication_vec_[i](slice_x, index_y, 2 *
    index_z + 1);
        }
    }
    */
}

void hpxfft::fft3D::distributed::loop::pencil::split_vec(const std::size_t x, const std::size_t dummy)
{
    if (dummy == 1)
    {
        for (std::size_t j = 0; j < localities_per_dir_; j++)
        {
            const std::size_t part = 2 * (z_offsets_[j + 1] - z_offsets_[j]);
            for (std::size_t y = 0; y < values_prep_[j].n_y(); ++y)
            {
                // contiguous run in z: plain copy
                std::copy_n(&values_vec_(x, y, 2 * z_offsets_[j]), part, &values_prep_[j](x, y, 0));
            }
        }
    }
    else if (dummy == 2)
    {
        std::size_t part = values_vec_.n_z() / localities_per_dir_;
        for (std::size_t j = 0; j < localities_per_dir_; j++)
        {
            for (std::size_t y = 0; y < values_prep2_[j].n_y(); ++y)
            {
                // contiguous run in z: plain copy
                std::copy_n(&values_vec_(x, y, part * j), part, &values_prep2_[j](x, y, 0));
            }
        }
    }
}

void hpxfft::fft3D::distributed::loop::pencil::split_permuted_vec(const std::size_t x, const std::size_t dummy)
{
    if (dummy == 1)
    {
        std::size_t part = permuted_vec_.n_z() / localities_per_dir_;
        for (std::size_t j = 0; j < localities_per_dir_; j++)
        {
            for (std::size_t y = 0; y < permuted_vec_.n_y(); ++y)
            {
                // contiguous run in z: plain copy
                std::copy_n(&permuted_vec_(x, y, part * j), part, &permuted_values_prep_[j](x, y, 0));
            }
        }
    }
    else if (dummy == 2)
    {
        std::size_t part = permuted_vec_.n_z() / localities_per_dir_;
        for (std::size_t j = 0; j < localities_per_dir_; j++)
        {
            for (std::size_t y = 0; y < permuted_vec_.n_y(); ++y)
            {
                // contiguous run in z: plain copy
                std::copy_n(&permuted_vec_(x, y, part * j), part, &permuted_values_prep2_[j](x, y, 0));
            }
        }
    }
}

// 3D FFT algorithm
hpxfft::fft3D::distributed::vector_3d hpxfft::fft3D::distributed::loop::pencil::fft_3d_r2c()
{
    begin_transform("fft_3d_r2c");
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
            for (std::size_t j = 0; j < n_y_local_; j++)
            {
                // fft for z direction
                fft_1d_r2c_inplace(i, j);
            }
        });
    // first split
    auto start_first_split = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        values_vec_.n_x(),
        [&](auto i)
        {
            // rearrange for communication step
            split_vec(i, 1);
        });
    // communication to get original data layout
    auto start_first_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < localities_per_dir_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_vec_y(i);
        }
        // global synchronization
        if (scatter_sync_)
        {
            for (std::size_t i = 0; i < localities_per_dir_; ++i)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_vec_y();
    }
    else
    {
        throw std::logic_error("Communication scheme not specified during initialization");
    }
    generation_countery_++;
    auto start_first_permute = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        localities_per_dir_,
        [&](auto i)
        {
            if (!scatter_sync_)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
            // for every "slice" in the first dimension (x)
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                values_vec_.n_x(),
                [&](auto k)
                {
                    // permute first and third dimension x-y-z -> x-z-y
                    permute_distributed_x_z_y(k, i);
                });
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
            for (std::size_t j = 0; j < n_z_local_; j++)
            {
                // fft for y direction
                fft_1d_c2c_y_inplace(i, j);
            }
        });
    auto start_second_split = t_.now();
    // splitting along the third dimesion (y)
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_permuted_vec(i, 1);
        });
    // communication for FFT in third dimension
    auto start_second_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < localities_per_dir_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_permuted_vec_x(i);
        }
        // global sychronization
        if (scatter_sync_)
        {
            for (std::size_t i = 0; i < localities_per_dir_; ++i)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_permuted_vec_x();
    }
    generation_counterx_++;
    auto start_second_permute = t_.now();
    // reuse the buffer when the size is the same, otherwise allocate
    if (n_y_local_ * n_z_local_ * 2 * dim_c_x_ == values_vec_.n_x() * values_vec_.n_y() * values_vec_.n_z())
    {
        values_vec_.rearrange(n_y_local_, n_z_local_, 2 * dim_c_x_);
    }
    else
    {
        values_vec_ = vector_3d(n_y_local_, n_z_local_, 2 * dim_c_x_, vector_3d::uninitialized);
    }
    // for every locality
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        localities_per_dir_,
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
                values_vec_.n_y(),
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
            for (std::size_t j = 0; j < n_z_local_; ++j)
            {
                // 1D FFT c2c in x-direction
                fft_1d_c2c_x_inplace(i, j);
            }
        });
    auto start_third_split = t_.now();
    if (transposed_out_)
    {
        // result stays in y-z-x layout: no final exchanges
        auto stop_total = t_.now();
        measurements_["total"] = stop_total - start_total;
        measurements_["first_fftw"] = start_first_split - start_total;
        measurements_["first_split"] = start_first_comm - start_first_split;
        measurements_["first_comm"] = start_first_permute - start_first_comm;
        measurements_["first_permute"] = start_second_fft - start_first_permute;
        measurements_["second_fftw"] = start_second_split - start_second_fft;
        measurements_["second_split"] = start_second_comm - start_second_split;
        measurements_["second_comm"] = start_second_permute - start_second_comm;
        measurements_["second_permute"] = start_third_fft - start_second_permute;
        measurements_["third_fftw"] = start_third_split - start_third_fft;
        measurements_["third_split"] = 0.0;
        measurements_["third_comm"] = 0.0;
        measurements_["third_permute"] = 0.0;
        measurements_["fourth_split"] = 0.0;
        measurements_["fourth_comm"] = 0.0;
        measurements_["fourth_permute"] = 0.0;
        return std::move(values_vec_);
    }
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        values_vec_.n_x(),
        [&](auto i)
        {
            // rearrange for communication step
            split_vec(i, 2);
        });
    // communication to get original data layout
    auto start_third_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < localities_per_dir_; ++i)
        {
            // scatter operation from all localities in x direction
            communicate_scatter_vec_x(i);
        }
        // global synchronization
        if (scatter_sync_)
        {
            for (std::size_t i = 0; i < localities_per_dir_; ++i)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_vec_x();
    }
    generation_counterx_++;
    auto start_third_permute = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        localities_per_dir_,
        [&](auto i)
        {
            if (!scatter_sync_)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                permuted_vec_.n_y(),
                [&](auto j)
                {
                    // permute whole matrix y-z-x -> x-z-y
                    permute_distributed_z_y_x(j, i);
                });
        });
    auto start_fourth_split = t_.now();
    // splitting along the third dimesion (y)
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        permuted_vec_.n_x(),
        [&](auto i)
        {
            // rearrange for communication step
            split_permuted_vec(i, 2);
        });
    // communication for FFT in third dimension
    auto start_fourth_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < localities_per_dir_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_permuted_vec_y(i);
        }
        // global sychronization
        if (scatter_sync_)
        {
            for (std::size_t i = 0; i < localities_per_dir_; ++i)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_permuted_vec_y();
    }
    generation_countery_++;
    auto start_fourth_permute = t_.now();
    if (n_x_local_ * n_y_local_ * 2 * dim_c_z_ == values_vec_.n_x() * values_vec_.n_y() * values_vec_.n_z())
    {
        values_vec_.rearrange(n_x_local_, n_y_local_, 2 * dim_c_z_);
    }
    else
    {
        values_vec_ = vector_3d(n_x_local_, n_y_local_, 2 * dim_c_z_, vector_3d::uninitialized);
    }
    // for every locality
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        localities_per_dir_,
        [&](auto i)
        {
            if (!scatter_sync_)
            {
                communication_vec_[i] = communication_futures_[i].get();
            }
            // for every "slice" in the first dimension (x)
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                values_vec_.n_x(),
                [&](auto k)
                {
                    // permute first and third dimension x-z-y -> x-y-z
                    permute_distributed_x_z_y(k, i);
                });
        });
    auto stop_total = t_.now();

    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_split - start_total;
    measurements_["first_split"] = start_first_comm - start_first_split;
    measurements_["first_comm"] = start_first_permute - start_first_comm;
    measurements_["first_permute"] = start_second_fft - start_first_permute;
    measurements_["second_fftw"] = start_second_split - start_second_fft;
    measurements_["second_split"] = start_second_comm - start_second_split;
    measurements_["second_comm"] = start_second_permute - start_second_comm;
    measurements_["second_permute"] = start_third_fft - start_second_permute;
    measurements_["third_fftw"] = start_third_split - start_third_fft;
    measurements_["third_split"] = start_third_comm - start_third_split;
    measurements_["third_comm"] = start_third_permute - start_third_comm;
    measurements_["third_permute"] = start_fourth_split - start_third_permute;
    measurements_["fourth_split"] = start_fourth_comm - start_fourth_split;
    measurements_["fourth_comm"] = start_fourth_permute - start_fourth_comm;
    measurements_["fourth_permute"] = stop_total - start_fourth_permute;
    ////////////////////////////////////////////////////////////////

    return std::move(values_vec_);
}
