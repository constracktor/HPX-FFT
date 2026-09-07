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

    if (values_vec_.n_y() % num_localities_ != 0 || values_vec_.n_z() % num_localities_ != 0)
    {
        throw std::invalid_argument("Dimensions of the 3D Matrix are not divisable by number of localities");
    }

    // resize other data structures
    permuted_vec_ = std::move(vector_3d(n_x_local_, dim_c_z_, 2 * dim_c_y_));
    values_prep_.resize(num_localities_);
    permuted_values_prep_.resize(num_localities_);
    for (std::size_t i = 0; i < num_localities_; ++i)
    {
        permuted_values_prep_[i] = vector_3d(n_x_local_, dim_c_z_, 2 * n_y_local_);
        values_prep_[i] = vector_3d(n_y_local_, dim_c_z_, 2 * n_x_local_);
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
    if (COMM_FLAG == "scatter_async")
    {
        scatter_sync_ = false;
        COMM_FLAG_ = "scatter";
    }
    else
    {
        COMM_FLAG_ = COMM_FLAG;
    }
    if (COMM_FLAG_ == "scatter")
    {
        communication_vec_.resize(num_localities_);
        communication_futures_.resize(num_localities_);
        // setup communicators
        basename_storage_.resize(num_localities_);
        basenames_.resize(num_localities_);
        communicators_.resize(num_localities_);
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            basename_storage_[i] = std::to_string(i);
            basenames_[i] = basename_storage_[i].c_str();
            communicators_[i] = std::move(hpx::collectives::create_communicator(
                basenames_[i],
                hpx::collectives::num_sites_arg(num_localities_),
                hpx::collectives::this_site_arg(this_locality_)));
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        communication_vec_.resize(1);
        // setup communicators
        basename_storage_.resize(1);
        basenames_.resize(1);
        communicators_.resize(1);
        basename_storage_[0] = std::to_string(0);
        basenames_[0] = basename_storage_[0].c_str();
        communicators_[0] = std::move(hpx::collectives::create_communicator(
            basenames_[0],
            hpx::collectives::num_sites_arg(num_localities_),
            hpx::collectives::this_site_arg(this_locality_)));
    }
    else
    {
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
}

// scatter communitcation
void hpxfft::fft3D::distributed::loop::slab::communicate_scatter_vec(const std::size_t i)
{
    if (this_locality_ != i)
    {
        // receive from other locality
        communication_futures_[i] =
            hpx::collectives::scatter_from<vector_3d>(communicators_[i], hpx::collectives::generation_arg(2));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[i], std::move(values_prep_), hpx::collectives::generation_arg(2));
    }
}

void hpxfft::fft3D::distributed::loop::slab::communicate_scatter_permuted_vec(const std::size_t i)
{
    if (this_locality_ != i)
    {
        // receive from other locality
        communication_futures_[i] =
            hpx::collectives::scatter_from<vector_3d>(communicators_[i], hpx::collectives::generation_arg(1));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[i], std::move(permuted_values_prep_), hpx::collectives::generation_arg(1));
    }
}

// all to all communication
void hpxfft::fft3D::distributed::loop::slab::communicate_all_to_all_vec()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[0], std::move(values_prep_), hpx::collectives::generation_arg(2))
            .get();
}

void hpxfft::fft3D::distributed::loop::slab::communicate_all_to_all_permuted_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(
                             communicators_[0], std::move(permuted_values_prep_), hpx::collectives::generation_arg(1))
                             .get();
}

// permute data for FFT in y-direction (only local data)
void hpxfft::fft3D::distributed::loop::slab::permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i)
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

// permute data after communication
void hpxfft::fft3D::distributed::loop::slab::permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i)
{
    const std::size_t part = values_vec_.n_z() / num_localities_;
    const std::size_t offset = part * i;
    for (std::size_t index_z = 0; index_z < values_vec_.n_x(); index_z++)
    {
        for (std::size_t index_x = 0; index_x < part / 2; index_x++)
        {
            values_vec_(index_z, slice_y, offset + 2 * index_x) = communication_vec_[i](index_x, slice_y, 2 * index_z);
            values_vec_(index_z, slice_y, offset + 2 * index_x + 1) =
                communication_vec_[i](index_x, slice_y, 2 * index_z + 1);
        }
    }
}

void hpxfft::fft3D::distributed::loop::slab::permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i)
{
    const std::size_t part = permuted_vec_.n_y() / num_localities_;
    const std::size_t offset = part * i;
    for (std::size_t index_z = 0; index_z < permuted_vec_.n_x(); index_z++)
    {
        for (std::size_t index_y = 0; index_y < permuted_vec_.n_z() / 2; index_y++)
        {
            permuted_vec_(index_z, offset + slice_x, 2 * index_y) =
                communication_vec_[i](slice_x, index_y, 2 * index_z);
            permuted_vec_(index_z, offset + slice_x, 2 * index_y + 1) =
                communication_vec_[i](slice_x, index_y, 2 * index_z + 1);
        }
    }
}

void hpxfft::fft3D::distributed::loop::slab::split_vec(const std::size_t x, const std::size_t dummy)
{
    std::size_t part = values_vec_.n_z() / num_localities_;
    for (std::size_t j = 0; j < num_localities_; j++)
    {
        for (std::size_t y = 0; y < values_prep_[j].n_y(); ++y)
        {
            for (std::size_t z = 0; z < part; z++)
            {
                values_prep_[j](x, y, z) = values_vec_(x, y, z + part * j);
            }
        }
    }
}

void hpxfft::fft3D::distributed::loop::slab::split_permuted_vec(const std::size_t x, const std::size_t dummy)
{
    std::size_t part = permuted_vec_.n_z() / num_localities_;
    for (std::size_t j = 0; j < num_localities_; j++)
    {
        for (std::size_t y = 0; y < permuted_vec_.n_y(); ++y)
        {
            for (std::size_t z = 0; z < part; z++)
            {
                permuted_values_prep_[j](x, y, z) = permuted_vec_(x, y, z + part * j);
            }
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
