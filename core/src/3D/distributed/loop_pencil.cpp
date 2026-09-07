#include "../../../include/hpxfft/3D/distributed/loop.hpp"

void hpxfft::fft3D::distributed::loop::pencil::initialize(
    vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG)
{
    // move data into own structure
    values_vec_ = std::move(values_vec);
    // locality information
    this_locality_ = hpx::get_locality_id();
    num_localities_ = hpx::get_num_localities(hpx::launch::sync);
    localities_per_dir_ = std::sqrt(num_localities_);
    // parameters
    n_x_local_ = values_vec_.n_x();
    n_y_local_ = values_vec_.n_y();
    n_z_local_ = values_vec_.n_z() / (2 * localities_per_dir_);
    dim_c_x_ = n_x_local_ * localities_per_dir_;
    dim_c_y_ = n_y_local_ * localities_per_dir_;
    dim_c_z_ = values_vec_.n_z() / 2;
    dim_r_z_ = 2 * dim_c_z_ - 2;
    dim_c_z_part_ = 2 * dim_c_z_ / localities_per_dir_;
    dim_c_y_part_ = 2 * dim_c_y_ / localities_per_dir_;
    dim_c_x_part_ = 2 * dim_c_x_ / localities_per_dir_;
    if (localities_per_dir_ * localities_per_dir_ != num_localities_)
    {
        throw std::invalid_argument("Number of localities must be a square number using pencil decomposition");
    }
    if (values_vec_.n_z() % localities_per_dir_ != 0)
    {
        throw std::invalid_argument("Dimensions of 3D Matrix are not divisable by number of localities");
    }
    pos_dir_x_ = this_locality_ / localities_per_dir_;
    pos_dir_y_ = this_locality_ % localities_per_dir_;
    // resize other data structures
    permuted_vec_ = std::move(vector_3d(n_x_local_, n_z_local_, 2 * dim_c_y_));
    values_prep_.resize(localities_per_dir_);
    permuted_values_prep_.resize(localities_per_dir_);
    values_prep2_.resize(localities_per_dir_);
    permuted_values_prep2_.resize(localities_per_dir_);
    for (std::size_t i = 0; i < localities_per_dir_; ++i)
    {
        permuted_values_prep_[i] = vector_3d(n_x_local_, n_z_local_, 2 * n_y_local_);
        values_prep_[i] = vector_3d(n_x_local_, n_y_local_, 2 * n_z_local_);
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
        communication_vec_.resize(localities_per_dir_);
        communication_futures_.resize(localities_per_dir_);
        // setup communicators
        basename_storage_.resize(2 * localities_per_dir_);
        communicators_.resize(2 * localities_per_dir_);
        for (std::size_t i = 0; i < localities_per_dir_; ++i)
        {
            basename_storage_[2 * i] = "hpxfft/pencil/x/" + std::to_string(pos_dir_y_) + "/" + std::to_string(i);
            basename_storage_[2 * i + 1] = "hpxfft/pencil/y/" + std::to_string(pos_dir_x_) + "/" + std::to_string(i);
            communicators_[2 * i] = std::move(hpx::collectives::create_communicator(
                basename_storage_[2 * i].c_str(),
                hpx::collectives::num_sites_arg(localities_per_dir_),
                hpx::collectives::this_site_arg(pos_dir_x_)));
            communicators_[2 * i + 1] = std::move(hpx::collectives::create_communicator(
                basename_storage_[2 * i + 1].c_str(),
                hpx::collectives::num_sites_arg(localities_per_dir_),
                hpx::collectives::this_site_arg(pos_dir_y_)));
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        communication_vec_.resize(localities_per_dir_);
        // setup communicators
        basename_storage_.resize(2);
        communicators_.resize(2);
        basename_storage_[0] = "hpxfft/pencil/x/" + std::to_string(pos_dir_y_);
        basename_storage_[1] = "hpxfft/pencil/y/" + std::to_string(pos_dir_x_);
        communicators_[0] = std::move(hpx::collectives::create_communicator(
            basename_storage_[0].c_str(),
            hpx::collectives::num_sites_arg(localities_per_dir_),
            hpx::collectives::this_site_arg(pos_dir_x_)));
        communicators_[1] = std::move(hpx::collectives::create_communicator(
            basename_storage_[1].c_str(),
            hpx::collectives::num_sites_arg(localities_per_dir_),
            hpx::collectives::this_site_arg(pos_dir_y_)));
    }
    else
    {
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
    // Global synchronization to ensure all communicators are fully initialized before use
    // hpx::distributed::barrier("initialize Barrier").wait();
    generation_counterx_ = 1;
    generation_countery_ = 1;
}

// scatter communitcation
void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_vec_x(const std::size_t i)
{
    if (pos_dir_x_ != i)
    {
        // receive from other locality
        communication_futures_[i] = hpx::collectives::scatter_from<vector_3d>(
            communicators_[2 * i],
            hpx::collectives::this_site_arg(pos_dir_x_),
            hpx::collectives::generation_arg(generation_counterx_));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[2 * i],
            std::move(values_prep2_),
            hpx::collectives::this_site_arg(pos_dir_x_),
            hpx::collectives::generation_arg(generation_counterx_));
    }
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_vec_y(const std::size_t i)
{
    if (pos_dir_y_ != i)
    {
        // receive from other locality
        communication_futures_[i] = hpx::collectives::scatter_from<vector_3d>(
            communicators_[2 * i + 1],
            hpx::collectives::this_site_arg(pos_dir_y_),
            hpx::collectives::generation_arg(generation_countery_));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[2 * i + 1],
            std::move(values_prep_),
            hpx::collectives::this_site_arg(pos_dir_y_),
            hpx::collectives::generation_arg(generation_countery_));
    }
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_permuted_vec_x(const std::size_t i)
{
    if (pos_dir_x_ != i)
    {
        // receive from other locality
        communication_futures_[i] = hpx::collectives::scatter_from<vector_3d>(
            communicators_[2 * i],
            hpx::collectives::this_site_arg(pos_dir_x_),
            hpx::collectives::generation_arg(generation_counterx_));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[2 * i],
            std::move(permuted_values_prep_),
            hpx::collectives::this_site_arg(pos_dir_x_),
            hpx::collectives::generation_arg(generation_counterx_));
    }
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_scatter_permuted_vec_y(const std::size_t i)
{
    if (pos_dir_y_ != i)
    {
        // receive from other locality
        communication_futures_[i] = hpx::collectives::scatter_from<vector_3d>(
            communicators_[2 * i + 1],
            hpx::collectives::this_site_arg(pos_dir_y_),
            hpx::collectives::generation_arg(generation_countery_));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[2 * i + 1],
            std::move(permuted_values_prep2_),
            hpx::collectives::this_site_arg(pos_dir_y_),
            hpx::collectives::generation_arg(generation_countery_));
    }
}

// all to all communication
void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_vec_x()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[0],
                                     std::move(values_prep2_),
                                     hpx::collectives::this_site_arg(pos_dir_x_),
                                     hpx::collectives::generation_arg(generation_counterx_))
            .get();
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_vec_y()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[1],
                                     std::move(values_prep_),
                                     hpx::collectives::this_site_arg(pos_dir_y_),
                                     hpx::collectives::generation_arg(generation_countery_))
            .get();
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_permuted_vec_x()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[0],
                                     std::move(permuted_values_prep_),
                                     hpx::collectives::this_site_arg(pos_dir_x_),
                                     hpx::collectives::generation_arg(generation_counterx_))
            .get();
}

void hpxfft::fft3D::distributed::loop::pencil::communicate_all_to_all_permuted_vec_y()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[1],
                                     std::move(permuted_values_prep2_),
                                     hpx::collectives::this_site_arg(pos_dir_y_),
                                     hpx::collectives::generation_arg(generation_countery_))
            .get();
}

// permute data for FFT in y-direction
void hpxfft::fft3D::distributed::loop::pencil::permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i)
{
    if (generation_countery_ == 2)
    {
        const std::size_t part = permuted_vec_.n_z() / localities_per_dir_;
        const std::size_t offset = part * i;

        for (std::size_t index_y = 0; index_y < part / 2; ++index_y)
        {
            for (std::size_t index_z = 0; index_z < permuted_vec_.n_y(); ++index_z)
            {
                permuted_vec_.at(slice_x, index_z, offset + 2 * index_y) =
                    communication_vec_[i].at(slice_x, index_y, 2 * index_z);
                permuted_vec_.at(slice_x, index_z, offset + 2 * index_y + 1) =
                    communication_vec_[i].at(slice_x, index_y, 2 * index_z + 1);
            }
        }
    }
    else
    {
        const std::size_t part = values_vec_.n_z() / localities_per_dir_;
        const std::size_t offset = part * i;

        for (std::size_t index_y = 0; index_y < part / 2; ++index_y)
        {
            for (std::size_t index_z = 0; index_z < values_vec_.n_y(); ++index_z)
            {
                values_vec_.at(slice_x, index_z, offset + 2 * index_y) =
                    communication_vec_[i].at(slice_x, index_y, 2 * index_z);
                values_vec_.at(slice_x, index_z, offset + 2 * index_y + 1) =
                    communication_vec_[i].at(slice_x, index_y, 2 * index_z + 1);
            }
        }
    }
}

// permute data after communication
void hpxfft::fft3D::distributed::loop::pencil::permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i)
{
    if (generation_counterx_ == 2)
    {
        const std::size_t part = values_vec_.n_z() / localities_per_dir_;
        const std::size_t offset = part * i;
        for (std::size_t index_z = 0; index_z < values_vec_.n_x(); index_z++)
        {
            for (std::size_t index_x = 0; index_x < part / 2; index_x++)
            {
                values_vec_.at(index_z, slice_y, offset + 2 * index_x) =
                    communication_vec_[i].at(index_x, slice_y, 2 * index_z);
                values_vec_.at(index_z, slice_y, offset + 2 * index_x + 1) =
                    communication_vec_[i].at(index_x, slice_y, 2 * index_z + 1);
            }
        }
    }
    else
    {
        const std::size_t part = permuted_vec_.n_z() / localities_per_dir_;
        const std::size_t offset = part * i;
        for (std::size_t index_z = 0; index_z < permuted_vec_.n_x(); index_z++)
        {
            for (std::size_t index_x = 0; index_x < part / 2; index_x++)
            {
                permuted_vec_.at(index_z, slice_y, offset + 2 * index_x) =
                    communication_vec_[i].at(index_x, slice_y, 2 * index_z);
                permuted_vec_.at(index_z, slice_y, offset + 2 * index_x + 1) =
                    communication_vec_[i].at(index_x, slice_y, 2 * index_z + 1);
            }
        }
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
        std::size_t part = values_vec_.n_z() / localities_per_dir_;
        for (std::size_t j = 0; j < localities_per_dir_; j++)
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
    else if (dummy == 2)
    {
        std::size_t part = values_vec_.n_z() / localities_per_dir_;
        for (std::size_t j = 0; j < localities_per_dir_; j++)
        {
            for (std::size_t y = 0; y < values_prep2_[j].n_y(); ++y)
            {
                for (std::size_t z = 0; z < part; z++)
                {
                    values_prep2_[j](x, y, z) = values_vec_(x, y, z + part * j);
                }
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
                for (std::size_t z = 0; z < part; z++)
                {
                    permuted_values_prep_[j](x, y, z) = permuted_vec_(x, y, z + part * j);
                }
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
                for (std::size_t z = 0; z < part; z++)
                {
                    permuted_values_prep2_[j](x, y, z) = permuted_vec_(x, y, z + part * j);
                }
            }
        }
    }
}

// 3D FFT algorithm
hpxfft::fft3D::distributed::vector_3d hpxfft::fft3D::distributed::loop::pencil::fft_3d_r2c()
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
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
        return values_vec_;
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
    values_vec_.rearrange(n_y_local_, n_z_local_, 2 * dim_c_x_);
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
    values_vec_.rearrange(n_x_local_, n_y_local_, 2 * dim_c_z_);
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
