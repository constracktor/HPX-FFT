#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/hpx_init.hpp>
#include <hpx/execution.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/timing/high_resolution_timer.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <fftw3.h>

#include "vector_2d.hpp"

typedef double real;

// FFT calculation struct
struct fft
{
    typedef fftw_plan fft_backend_plan;
    typedef std::vector<hpx::future<void>> vector_future;
    typedef std::vector<std::vector<real>> vector_comm;

    public:
        fft() = default;

        void initialize(vector_2d<real> values_vec, 
                        const std::string COMM_FLAG,
                        const unsigned PLAN_FLAG);
        
        vector_2d<real> fft_2d_r2c();

        real get_measurement(std::string name);

        ~fft()
        {
            fftw_destroy_plan(plan_1d_r2c_);
            fftw_destroy_plan(plan_1d_c2c_);
            fftw_cleanup();
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

        // transpose after communication
        void transpose_y_to_x(const std::size_t k, const std::size_t i);
        void transpose_x_to_y(const std::size_t j, const std::size_t i);

    private:
        // parameters
        std::size_t n_x_local_, n_y_local_;
        std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
        std::size_t dim_c_y_part_, dim_c_x_part_;
        // FFTW plans
        unsigned PLAN_FLAG_;
        fft_backend_plan plan_1d_r2c_;
        fft_backend_plan plan_1d_c2c_;
        // value vectors
        vector_2d<real> values_vec_;
        vector_2d<real> trans_values_vec_;
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
        // communicators
        std::string COMM_FLAG_;
        std::vector<const char*> basenames_;
        std::vector<hpx::collectives::communicator> communicators_;
};

// FFT backend
void fft::fft_1d_r2c_inplace(const std::size_t i)
{
    fftw_execute_dft_r2c(plan_1d_r2c_, 
                            values_vec_.row(i), 
                            reinterpret_cast<fftw_complex*>(values_vec_.row(i)));
}

void fft::fft_1d_c2c_inplace(const std::size_t i)
{
    fftw_execute_dft(plan_1d_c2c_, 
                        reinterpret_cast<fftw_complex*>(trans_values_vec_.row(i)), 
                        reinterpret_cast<fftw_complex*>(trans_values_vec_.row(i)));
}

// split data for communication
void fft::split_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j) 
    { //std::move same performance
        std::copy(values_vec_.row(i) + j * dim_c_y_part_, 
                    values_vec_.row(i) + (j+1) * dim_c_y_part_,
                    values_prep_[j].begin() + i * dim_c_y_part_);
    }
}

void fft::split_trans_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j) 
    { //std::move same performance
        std::copy(trans_values_vec_.row(i) + j * dim_c_x_part_,
                    trans_values_vec_.row(i) + (j+1) * dim_c_x_part_,
                    trans_values_prep_[j].begin() + i * dim_c_x_part_);
    }
}

void fft::communicate_scatter_vec(const std::size_t i)
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

void fft::communicate_scatter_trans_vec(const std::size_t i)
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
void fft::communicate_all_to_all_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(communicators_[0], 
                std::move(values_prep_), 
                hpx::collectives::generation_arg(1)).get();
}

void fft::communicate_all_to_all_trans_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(communicators_[0], 
                std::move(trans_values_prep_), 
                hpx::collectives::generation_arg(2)).get();
}

// transpose after communication
void fft::transpose_y_to_x(const std::size_t k, const std::size_t i)
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

void fft::transpose_x_to_y(const std::size_t j, const std::size_t i)
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
vector_2d<real> fft::fft_2d_r2c()
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
void fft::initialize(vector_2d<real> values_vec, 
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
    trans_values_vec_ = std::move(vector_2d<real>(n_y_local_, 2 * dim_c_x_));
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
real fft::get_measurement(std::string name)
{
    return measurements_[name];
}

void print_vector_2d(const vector_2d<real>& input)
{
    const std::string msg = "\n";
    
    const std::size_t dim_x = input.n_row();
    const std::size_t dim_y = input.n_col();

    std::size_t counter = 0;
    for( std::size_t i = 0; i < dim_x; ++i)
    {
        for( std::size_t j = 0; j < dim_y; ++j)
        {
            real element =  input(i,j);
            if(counter%2 == 0)
            {
                std::string msg = "({1} ";
                hpx::util::format_to(std::cout, msg, element) << std::flush;
            }
            else
            {
                std::string msg = "{1}) ";
                hpx::util::format_to(std::cout, msg, element) << std::flush;
            }
            ++counter;
        }    
        hpx::util::format_to(std::cout, msg) << std::flush;
    }
    hpx::util::format_to(std::cout, msg) << std::flush;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    hpx::util::format_to(std::cout, "ENTERED HPX MAIN\n");
    ////////////////////////////////////////////////////////////////
    // Parameters and Data structures
    const std::size_t this_locality = hpx::get_locality_id(); 
    const std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    const std::string run_flag = vm["run"].as<std::string>();
    const std::string plan_flag = vm["plan"].as<std::string>();
    bool print_result = vm["result"].as<bool>();
    bool print_header = vm["header"].as<bool>();
    // time measurement
    auto t = hpx::chrono::high_resolution_timer();  
    // FFT dimension parameters
    const std::size_t dim_c_x = vm["nx"].as<std::size_t>();//N_X; 
    const std::size_t dim_r_y = vm["ny"].as<std::size_t>();//N_Y;
    const std::size_t dim_c_y = dim_r_y / 2 + 1;
    // division parameter
    const std::size_t n_x_local = dim_c_x / num_localities;;
    // FFTW plans
    unsigned FFT_BACKEND_PLAN_FLAG = FFTW_ESTIMATE;
    if( plan_flag == "measure" )
    {
        FFT_BACKEND_PLAN_FLAG = FFTW_MEASURE;
    }
    else if ( plan_flag == "patient")
    {
        FFT_BACKEND_PLAN_FLAG = FFTW_PATIENT;
    }
    else if ( plan_flag == "exhaustive")
    {
        FFT_BACKEND_PLAN_FLAG = FFTW_EXHAUSTIVE;
    }

    ////////////////////////////////////////////////////////////////
    // Initialization
    vector_2d<real> values_vec(n_x_local, 2 * dim_c_y);
    for(std::size_t i = 0; i < n_x_local; ++i)
    {
        std::iota(values_vec.row(i), values_vec.row(i+1) - 2, 0.0);
    }

    ////////////////////////////////////////////////////////////////
    // Computation   
    fft fft_computer;
    auto start_total = t.now();
    fft_computer.initialize(std::move(values_vec), run_flag, FFT_BACKEND_PLAN_FLAG);
    auto stop_init = t.now();
    values_vec = fft_computer.fft_2d_r2c();
    auto stop_total = t.now();
    
    // optional: print results 
    if (print_result)
    {
        sleep(this_locality);
        print_vector_2d(values_vec);
    }

    ////////////////////////////////////////////////////////////////
    // Postprocessing
    // print and store runtimes if on locality 0
    if (this_locality==0)
    {
        auto total = stop_total - start_total;
        auto init = stop_init - start_total;
        std::string msg = "\nLocality {13} -  {1}:\n"
                          "Total runtime : {2}\n"
                          "Initialization: {3}\n"
                          "FFT 2D runtime: {4}\n"
                          "FFTW r2c      : {5}\n"
                          "First split   : {6}\n"
                          "First comm    : {7}\n"
                          "First trans   : {8}\n"
                          "FFTW c2c      : {9}\n"
                          "Second split  : {10}\n"
                          "Second comm   : {11}\n"
                          "Second trans  : {12}\n";
        hpx::util::format_to(std::cout, msg, 
                            run_flag, 
                            total,
                            init,
                            fft_computer.get_measurement("total"),
                            fft_computer.get_measurement("first_fftw"),
                            fft_computer.get_measurement("first_split"),
                            fft_computer.get_measurement("first_comm"),
                            fft_computer.get_measurement("first_trans"),
                            fft_computer.get_measurement("second_fftw"),
                            fft_computer.get_measurement("second_split"),
                            fft_computer.get_measurement("second_comm"),
                            fft_computer.get_measurement("second_trans"),
                            this_locality) << std::flush;
        std::ofstream runtime_file;
        runtime_file.open("result/runtimes_hpx_loop_dist.txt", std::ios_base::app);
        if(print_header)
        {
            runtime_file << "n_threads;n_x;n_y;plan;comm_flag;total;initialization;"
                    << "fft_2d_total;"
                    << "first_fftw;"
                    << "first_split;"
                    << "first_comm;"
                    << "first_trans;"
                    << "second_fftw;"
                    << "second_split;"
                    << "second_comm;"
                    << "second_trans;\n";
        }
        runtime_file << hpx::get_os_thread_count() << ";" 
                    << dim_c_x << ";"
                    << dim_r_y << ";"
                    << plan_flag << ";"
                    << run_flag << ";"
                    << total << ";"
                    << init << ";"
                    << fft_computer.get_measurement("total") << ";"
                    << fft_computer.get_measurement("first_fftw") << ";"
                    << fft_computer.get_measurement("first_split") << ";"
                    << fft_computer.get_measurement("first_comm") << ";"
                    << fft_computer.get_measurement("first_trans") << ";"
                    << fft_computer.get_measurement("second_fftw") << ";"
                    << fft_computer.get_measurement("second_split") << ";"
                    << fft_computer.get_measurement("second_comm") << ";"
                    << fft_computer.get_measurement("second_trans") << ";\n";
        runtime_file.close();
    }

    ////////////////////////////////////////////////////////////////
    // Finalize HPX runtime
    hpx::util::format_to(std::cout, "BEFORE FINALIZE\n");
    auto val = hpx::finalize();
    hpx::util::format_to(std::cout, "AFTER FINALIZE\n");
    return val;
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
    ("result", value<bool>()->default_value(0), 
    "Print generated results (default: false)")
    ("nx", value<std::size_t>()->default_value(8), 
    "Total x dimension")
    ("ny", value<std::size_t>()->default_value(14), 
    "Total y dimension")
    ("plan", value<std::string>()->default_value("estimate"), 
    "FFTW plan (default: estimate)")
    ("run",value<std::string>()->default_value("scatter"), 
    "Choose 2d FFT algorithm communication: scatter or all_to_all")
    ("header",value<bool>()->default_value(0), 
    "Write runtime file header");

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;
    hpx::util::format_to(std::cout, "BEFORE HPX MAIN\n");
    return hpx::init(argc, argv, init_args);
}

#endif