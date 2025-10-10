#include "../../include/hpxfft/distributed/agas.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/modules/components.hpp>

// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<hpxfft::distributed::agas_server> agas_server_type;
HPX_REGISTER_COMPONENT(agas_server_type, agas_server)

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
// typedef hpxfft::distributed::agas_server::fft_2d_r2c_action fft_2d_r2c_action;
HPX_REGISTER_ACTION(fft_2d_r2c_action)

// typedef hpxfft::distributed::agas_server::initialize_action initialize_action;
HPX_REGISTER_ACTION(initialize_action)

// FFT backend
void hpxfft::distributed::agas_server::fft_1d_r2c_inplace(const std::size_t i)
{
    fftw_execute_dft_r2c(plan_1d_r2c_, values_vec_.row(i), reinterpret_cast<fftw_complex *>(values_vec_.row(i)));
}

void hpxfft::distributed::agas_server::fft_1d_c2c_inplace(const std::size_t i)
{
    fftw_execute_dft(plan_1d_c2c_,
                     reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)),
                     reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)));
}

// split data for communication
void hpxfft::distributed::agas_server::split_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j)
    {  // std::move same performance
        std::copy(values_vec_.row(i) + j * dim_c_y_part_,
                  values_vec_.row(i) + (j + 1) * dim_c_y_part_,
                  values_prep_[j].begin() + i * dim_c_y_part_);
    }
}

void hpxfft::distributed::agas_server::split_trans_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j)
    {  // std::move same performance
        std::copy(trans_values_vec_.row(i) + j * dim_c_x_part_,
                  trans_values_vec_.row(i) + (j + 1) * dim_c_x_part_,
                  trans_values_prep_[j].begin() + i * dim_c_x_part_);
    }
}

// scatter communication
void hpxfft::distributed::agas_server::communicate_scatter_vec(const std::size_t i)
{
    if (this_locality_ != i)
    {
        // receive from other locality
        communication_vec_[i] =
            hpx::collectives::scatter_from<std::vector<real>>(communicators_[i], hpx::collectives::generation_arg(1))
                .get();
    }
    else
    {
        // send from this locality
        communication_vec_[i] = hpx::collectives::scatter_to(
                                    communicators_[i], std::move(values_prep_), hpx::collectives::generation_arg(1))
                                    .get();
    }
}

void hpxfft::distributed::agas_server::communicate_scatter_trans_vec(const std::size_t i)
{
    if (this_locality_ != i)
    {
        // receive from other locality
        communication_vec_[i] =
            hpx::collectives::scatter_from<std::vector<real>>(communicators_[i], hpx::collectives::generation_arg(2))
                .get();
    }
    else
    {
        // send from this locality
        communication_vec_[i] =
            hpx::collectives::scatter_to(
                communicators_[i], std::move(trans_values_prep_), hpx::collectives::generation_arg(2))
                .get();
    }
}

// all to all communication
void hpxfft::distributed::agas_server::communicate_all_to_all_vec()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[0], std::move(values_prep_), hpx::collectives::generation_arg(1))
            .get();
}

void hpxfft::distributed::agas_server::communicate_all_to_all_trans_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(
                             communicators_[0], std::move(trans_values_prep_), hpx::collectives::generation_arg(2))
                             .get();
}

// transpose after communication
void hpxfft::distributed::agas_server::transpose_y_to_x(const std::size_t k, const std::size_t i)
{
    std::size_t index_in;
    std::size_t index_out;
    const std::size_t offset_in = 2 * k;
    const std::size_t offset_out = 2 * i;
    const std::size_t factor_in = dim_c_y_part_;
    const std::size_t factor_out = 2 * num_localities_;
    const std::size_t dim_input = communication_vec_[i].size() / factor_in;

    for (std::size_t j = 0; j < dim_input; ++j)
    {
        // compute indices once use twice
        index_in = factor_in * j + offset_in;
        index_out = factor_out * j + offset_out;
        // transpose
        trans_values_vec_(k, index_out) = communication_vec_[i][index_in];
        trans_values_vec_(k, index_out + 1) = communication_vec_[i][index_in + 1];
    }
}

void hpxfft::distributed::agas_server::transpose_x_to_y(const std::size_t j, const std::size_t i)
{
    std::size_t index_in;
    std::size_t index_out;
    const std::size_t offset_out = 2 * i;
    const std::size_t factor_in = dim_c_y_part_;
    const std::size_t factor_out = 2 * num_localities_;
    const std::size_t dim_input = communication_vec_[i].size() / factor_in;

    for (std::size_t k = 0; k < dim_input; ++k)
    {
        // compute indices once use twice
        std::size_t offset_in = 2 * k;
        index_in = factor_in * j + offset_in;
        index_out = factor_out * j + offset_out;
        // transpose
        values_vec_(k, index_out) = communication_vec_[i][index_in];
        values_vec_(k, index_out + 1) = communication_vec_[i][index_in + 1];
    }
}

// 2D FFT algorithm
hpxfft::distributed::vector_2d hpxfft::distributed::agas_server::fft_2d_r2c()
{
    // first dimension
    for (std::size_t i = 0; i < n_x_local_; ++i)
    {
        // 1d FFT r2c in y-direction
        r2c_futures_[i] = hpx::async(fft_1d_r2c_inplace_action(), get_id(), i);
        // prepare for communication
        split_vec_futures_[i] = r2c_futures_[i].then(
            [=, this](hpx::future<void> r)
            {
                r.get();
                return hpx::async(split_vec_action(), get_id(), i);
            });
    }
    // local synchronization step for communication
    hpx::shared_future<vector_future> all_split_vec_futures = hpx::when_all(split_vec_futures_);
    // communication for FFT in second dimension
    if (COMM_FLAG_ == "scatter")
    {
        // scatter operation from all localities
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            communication_futures_[i] = all_split_vec_futures.then(
                [=, this](hpx::shared_future<vector_future> r)
                {
                    r.get();
                    return hpx::async(communicate_scatter_vec_action(), get_id(), i);
                });
        }
        // tranpose from y-direction to x-direction
        for (std::size_t k = 0; k < n_y_local_; ++k)
        {
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                trans_y_to_x_futures_[k][i] = communication_futures_[i].then(
                    [=, this](hpx::shared_future<void> r)
                    {
                        r.get();
                        return hpx::async(transpose_y_to_x_action(), get_id(), k, i);
                    });
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        communication_futures_[0] = all_split_vec_futures.then(
            [=, this](hpx::shared_future<vector_future> r)
            {
                r.get();
                return hpx::async(communicate_all_to_all_vec_action(), get_id());
            });
        // tranpose from y-direction to x-direction
        for (std::size_t k = 0; k < n_y_local_; ++k)
        {
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                trans_y_to_x_futures_[k][i] = communication_futures_[0].then(
                    [=, this](hpx::shared_future<void> r)
                    {
                        r.get();
                        return hpx::async(transpose_y_to_x_action(), get_id(), k, i);
                    });
            }
        }
    }
    else
    {
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
    // second dimension
    for (std::size_t i = 0; i < n_y_local_; ++i)
    {
        // synchronize
        hpx::future<vector_future> all_trans_y_to_x_i_futures = hpx::when_all(trans_y_to_x_futures_[i]);
        // 1D FFT in x-direction
        c2c_futures_[i] = all_trans_y_to_x_i_futures.then(
            [=, this](hpx::future<vector_future> r)
            {
                r.get();
                return hpx::async(fft_1d_c2c_inplace_action(), get_id(), i);
            });
        // prepare for communication
        split_trans_vec_futures_[i] = c2c_futures_[i].then(
            [=, this](hpx::future<void> r)
            {
                r.get();
                return hpx::async(split_trans_vec_action(), get_id(), i);
            });
    }
    // local synchronization step for communication
    hpx::shared_future<vector_future> all_split_trans_vec_futures = hpx::when_all(split_trans_vec_futures_);
    /////////////////////////////////
    // communication to get original data layout
    if (COMM_FLAG_ == "scatter")
    {
        // scatter operation from all localities
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            communication_futures_[i] = all_split_trans_vec_futures.then(
                [=, this](hpx::shared_future<vector_future> r)
                {
                    r.get();
                    return hpx::async(communicate_scatter_trans_vec_action(), get_id(), i);
                });
        }
        // tranpose from x-direction to y-direction
        for (std::size_t j = 0; j < n_y_local_; ++j)
        {
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                trans_x_to_y_futures_[j][i] = communication_futures_[i].then(
                    [=, this](hpx::shared_future<void> r)
                    {
                        r.get();
                        return hpx::async(transpose_x_to_y_action(), get_id(), j, i);
                    });
            }
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        communication_futures_[0] = all_split_trans_vec_futures.then(
            [=, this](hpx::shared_future<vector_future> r)
            {
                r.get();
                return hpx::async(communicate_all_to_all_trans_vec_action(), get_id());
            });
        // tranpose from x-direction to y-direction
        for (std::size_t k = 0; k < n_x_local_; ++k)
        {
            for (std::size_t i = 0; i < num_localities_; ++i)
            {
                trans_x_to_y_futures_[k][i] = communication_futures_[0].then(
                    [=, this](hpx::shared_future<void> r)
                    {
                        r.get();
                        return hpx::async(transpose_x_to_y_action(), get_id(), k, i);
                    });
            }
        }
    }
    // wait for every task to finish
    for (std::size_t i = 0; i < n_x_local_; ++i)
    {
        hpx::wait_all(trans_x_to_y_futures_[i]);
    };
    return std::move(values_vec_);
}

// initialization
void hpxfft::distributed::agas_server::initialize(
    hpxfft::distributed::vector_2d values_vec, const std::string COMM_FLAG, const unsigned PLAN_FLAG)
{
    // move data into own structure
    values_vec_ = std::move(values_vec);
    // locality information
    this_locality_ = hpx::get_locality_id();
    num_localities_ = hpx::get_num_localities(hpx::launch::sync);
    // parameters
    n_x_local_ = values_vec_.n_row();
    dim_c_x_ = n_x_local_ * num_localities_;
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = 2 * dim_c_y_ - 2;
    n_y_local_ = dim_c_y_ / num_localities_;
    dim_c_y_part_ = 2 * dim_c_y_ / num_localities_;
    dim_c_x_part_ = 2 * dim_c_x_ / num_localities_;
    // resize other data structures
    trans_values_vec_ = std::move(hpxfft::distributed::vector_2d(n_y_local_, 2 * dim_c_x_));
    values_prep_.resize(num_localities_);
    trans_values_prep_.resize(num_localities_);
    for (std::size_t i = 0; i < num_localities_; ++i)
    {
        values_prep_[i].resize(n_x_local_ * dim_c_y_part_);
        trans_values_prep_[i].resize(n_y_local_ * dim_c_x_part_);
    }
    // create fftw plans
    PLAN_FLAG_ = PLAN_FLAG;
    // forward step one: r2c in y-direction
    plan_1d_r2c_ = fftw_plan_dft_r2c_1d(
        dim_r_y_, values_vec_.row(0), reinterpret_cast<fftw_complex *>(values_vec_.row(0)), PLAN_FLAG_);
    // forward step two: c2c in x-direction
    plan_1d_c2c_ = fftw_plan_dft_1d(
        dim_c_x_,
        reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
        reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
        FFTW_FORWARD,
        PLAN_FLAG_);
    // communication specific initialization
    COMM_FLAG_ = COMM_FLAG;
    if (COMM_FLAG_ == "scatter")
    {
        communication_vec_.resize(num_localities_);
        communication_futures_.resize(num_localities_);
        // setup communicators
        basenames_.resize(num_localities_);
        communicators_.resize(num_localities_);
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            basenames_[i] = std::move(std::to_string(i).c_str());
            communicators_[i] = std::move(hpx::collectives::create_communicator(
                basenames_[i],
                hpx::collectives::num_sites_arg(num_localities_),
                hpx::collectives::this_site_arg(this_locality_)));
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        communication_vec_.resize(1);
        communication_futures_.resize(1);
        // setup communicators
        basenames_.resize(1);
        communicators_.resize(1);
        basenames_[0] = std::move(std::to_string(0).c_str());
        communicators_[0] = std::move(hpx::collectives::create_communicator(
            basenames_[0],
            hpx::collectives::num_sites_arg(num_localities_),
            hpx::collectives::this_site_arg(this_locality_)));
    }
    else
    {
        std::cout << "Specify communication scheme: scatter or all_to_all\n";
        hpx::finalize();
    }
    // resize futures
    r2c_futures_.resize(n_x_local_);
    split_vec_futures_.resize(n_x_local_);
    c2c_futures_.resize(n_y_local_);
    split_trans_vec_futures_.resize(n_y_local_);
    trans_y_to_x_futures_.resize(n_y_local_);
    trans_x_to_y_futures_.resize(n_x_local_);
    for (std::size_t i = 0; i < n_y_local_; ++i)
    {
        trans_y_to_x_futures_[i].resize(num_localities_);
    }
    for (std::size_t i = 0; i < n_x_local_; ++i)
    {
        trans_x_to_y_futures_[i].resize(num_localities_);
    }
}
