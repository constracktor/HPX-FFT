#include <hpx/hpx_init.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>

#include "../../include/hpxfft/distributed/loop.hpp"

// FFT backend
void hpxfft::distributed::loop::fft_1d_r2c_inplace(const std::size_t i)
{
    fftw_execute_dft_r2c(plan_1d_r2c_, 
                            values_vec_.row(i), 
                            reinterpret_cast<fftw_complex*>(values_vec_.row(i)));
}

void hpxfft::distributed::loop::fft_1d_c2c_inplace(const std::size_t i)
{
    fftw_execute_dft(plan_1d_c2c_, 
                        reinterpret_cast<fftw_complex*>(trans_values_vec_.row(i)), 
                        reinterpret_cast<fftw_complex*>(trans_values_vec_.row(i)));
}

// split data for communication
void hpxfft::distributed::loop::split_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j) 
    { //std::move same performance
        std::copy(values_vec_.row(i) + j * dim_c_y_part_, 
                    values_vec_.row(i) + (j+1) * dim_c_y_part_,
                    values_prep_[j].begin() + i * dim_c_y_part_);
    }
}

void hpxfft::distributed::loop::split_trans_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j) 
    { //std::move same performance
        std::copy(trans_values_vec_.row(i) + j * dim_c_x_part_,
                    trans_values_vec_.row(i) + (j+1) * dim_c_x_part_,
                    trans_values_prep_[j].begin() + i * dim_c_x_part_);
    }
}

void hpxfft::distributed::loop::communicate_scatter_vec(const std::size_t i)
{
    if(this_locality_ != i)
    {
        // receive from other locality
        communication_futures_[i] = hpx::collectives::scatter_from<std::vector<real>>(communicators_[i], 
                hpx::collectives::generation_arg(1));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(communicators_[i], 
                std::move(values_prep_), 
                hpx::collectives::generation_arg(1));
    }   
} 

void hpxfft::distributed::loop::communicate_scatter_trans_vec(const std::size_t i)
{
    if(this_locality_ != i)
    {
        // receive from other locality
        communication_futures_[i] = hpx::collectives::scatter_from<std::vector<real>>(communicators_[i], 
                hpx::collectives::generation_arg(2));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(communicators_[i], 
                std::move(trans_values_prep_), 
                hpx::collectives::generation_arg(2));
    }
}

// all to all communication
void hpxfft::distributed::loop::communicate_all_to_all_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(communicators_[0], 
                std::move(values_prep_), 
                hpx::collectives::generation_arg(1)).get();
}

void hpxfft::distributed::loop::communicate_all_to_all_trans_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(communicators_[0], 
                std::move(trans_values_prep_), 
                hpx::collectives::generation_arg(2)).get();
}

// transpose after communication
void hpxfft::distributed::loop::transpose_y_to_x(const std::size_t k, const std::size_t i)
{
    std::size_t index_in;
    std::size_t index_out;
    const std::size_t offset_in = 2 * k;
    const std::size_t offset_out = 2 * i;
    const std::size_t factor_in = dim_c_y_part_;
    const std::size_t factor_out = 2 * num_localities_;
    const std::size_t dim_input = communication_vec_[i].size() / factor_in;

    for(std::size_t j = 0; j < dim_input; ++j)
    {
        // compute indices once use twice
        index_in = factor_in * j + offset_in;
        index_out = factor_out * j + offset_out;
        // transpose
        trans_values_vec_(k,index_out)     = communication_vec_[i][index_in];
        trans_values_vec_(k,index_out + 1) = communication_vec_[i][index_in + 1];
    }
}

void hpxfft::distributed::loop::transpose_x_to_y(const std::size_t j, const std::size_t i)
{
    std::size_t index_in;
    std::size_t index_out;
    const std::size_t offset_out = 2 * i;
    const std::size_t factor_in = dim_c_y_part_;
    const std::size_t factor_out = 2 * num_localities_;
    const std::size_t dim_input = communication_vec_[i].size() / factor_in;

    for(std::size_t k = 0; k < dim_input; ++k)
    {
        // compute indices once use twice
        std::size_t offset_in = 2 * k;
        index_in = factor_in * j + offset_in;
        index_out = factor_out * j + offset_out;
        // transpose
        values_vec_(k,index_out)     = communication_vec_[i][index_in];
        values_vec_(k,index_out + 1) = communication_vec_[i][index_in + 1];
    }
}

// 2D FFT algorithm
hpxfft::distributed::vector_2d hpxfft::distributed::loop::fft_2d_r2c()
{
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    hpx::experimental::for_loop(hpx::execution::par, 0, n_x_local_, [&](auto i)
    {
        // 1d FFT r2c in y-direction
        fft_1d_r2c_inplace(i);
    });
    auto start_first_split = t_.now();
    hpx::experimental::for_loop(hpx::execution::par, 0, n_x_local_, [&](auto i)
    {
        // rearrange for communication step
        split_vec(i);
    });
    // communication for FFT in second dimension
    auto start_first_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for(std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_vec(i);
        }
        // global sychronization
        for(std::size_t i = 0; i < num_localities_; ++i)
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
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
    auto start_first_trans = t_.now();
    hpx::experimental::for_loop(hpx::execution::par, 0, num_localities_, [&](auto i)
    {
        hpx::experimental::for_loop(hpx::execution::par, 0, n_y_local_, [&](auto k)
        {
                // transpose from y-direction to x-direction
                transpose_y_to_x(k, i);
        });
    });
    // second dimension
    auto start_second_fft = t_.now();
    hpx::experimental::for_loop(hpx::execution::par, 0, n_y_local_, [&](auto i)
    {
        // 1D FFT c2c in x-direction
        fft_1d_c2c_inplace(i);
    });
    auto start_second_split = t_.now();
    hpx::experimental::for_loop(hpx::execution::par, 0, n_y_local_, [&](auto i)
    {
        // rearrange for communication step
        split_trans_vec(i);
    });
    // communication to get original data layout
    auto start_second_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for(std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_trans_vec(i);
        }
        // global synchronization
        for(std::size_t i = 0; i < num_localities_; ++i)
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
    hpx::experimental::for_loop(hpx::execution::par, 0, num_localities_, [&](auto i)
    {
        hpx::experimental::for_loop(hpx::execution::par, 0, n_y_local_, [&](auto j)
        {
            // transpose from x-direction to y-direction
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
void hpxfft::distributed::loop::initialize(hpxfft::distributed::vector_2d values_vec, 
                     const std::string COMM_FLAG, 
                     const unsigned PLAN_FLAG)
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
    for(std::size_t i = 0; i < num_localities_; ++i)
    {
        values_prep_[i].resize(n_x_local_ * dim_c_y_part_);
        trans_values_prep_[i].resize(n_y_local_ * dim_c_x_part_);
    }
    //create fftw plans
    PLAN_FLAG_ = PLAN_FLAG;
    // forward step one: r2c in y-direction
    plan_1d_r2c_ = fftw_plan_dft_r2c_1d(dim_r_y_,
                                       values_vec_.row(0),
                                       reinterpret_cast<fftw_complex*>(values_vec_.row(0)),
                                       PLAN_FLAG_);
    // forward step two: c2c in x-direction
    plan_1d_c2c_ = fftw_plan_dft_1d(dim_c_x_, 
                                   reinterpret_cast<fftw_complex*>(trans_values_vec_.row(0)), 
                                   reinterpret_cast<fftw_complex*>(trans_values_vec_.row(0)), 
                                   FFTW_FORWARD,
                                   PLAN_FLAG_);
    // communication specific initialization
    hpx::util::format_to(std::cout, "BEFORE COMM INIT\n");
    COMM_FLAG_ = COMM_FLAG;
    if (COMM_FLAG_ == "scatter")
    {
        communication_vec_.resize(num_localities_);
        communication_futures_.resize(num_localities_);
        // setup communicators
        basenames_.resize(num_localities_);
        communicators_.resize(num_localities_);
        for(std::size_t i = 0; i < num_localities_; ++i)
        {
            basenames_[i] = std::move(std::to_string(i).c_str());
            communicators_[i] = std::move(hpx::collectives::create_communicator(basenames_[i],
                                          hpx::collectives::num_sites_arg(num_localities_), 
                                          hpx::collectives::this_site_arg(this_locality_)));
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        communication_vec_.resize(1);
        // setup communicators
        basenames_.resize(1);
        communicators_.resize(1);
        basenames_[0] = std::move(std::to_string(0).c_str());
        communicators_[0] = std::move(hpx::collectives::create_communicator(basenames_[0],
                                      hpx::collectives::num_sites_arg(num_localities_), 
                                      hpx::collectives::this_site_arg(this_locality_)));
    }
    else
    {
        std::cout << "Specify communication scheme: scatter or all_to_all\n";
        hpx::finalize();
    }
    hpx::util::format_to(std::cout, "AFTER COMM INIT\n");
}

// helpers
real hpxfft::distributed::loop::get_measurement(std::string name)
{
    return measurements_[name];
}
