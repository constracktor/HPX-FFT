#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/future.hpp>
#include <hpx/execution.hpp>
#include <hpx/timing/high_resolution_timer.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <fftw3.h>

#include "vector_2d.hpp"

typedef double real;

// FFT calculation struct
struct fft
{
    typedef fftw_plan fft_backend_plan;
    typedef std::vector<hpx::future<void>> vector_future;

    public:
        fft() = default;

        void initialize(vector_2d<real> values_vec, 
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

        // transpose
        void transpose_shared_y_to_x(const std::size_t index);
        void transpose_shared_x_to_y(const std::size_t index_trans); 

        // static wrappers
        static void fft_1d_r2c_inplace_wrapper(fft *th, 
                                               const std::size_t i);
        static void fft_1d_c2c_inplace_wrapper(fft *th, 
                                               const std::size_t i);
        static void transpose_shared_y_to_x_wrapper(fft *th, 
                                                    const std::size_t index);
        static void transpose_shared_x_to_y_wrapper(fft *th, 
                                                    const std::size_t index_trans); 
    private:
        // parameters
        std::size_t dim_r_y_, dim_c_y_, dim_c_x_;
        // FFTW plans
        unsigned PLAN_FLAG_;
        fft_backend_plan plan_1d_r2c_;
        fft_backend_plan plan_1d_c2c_;
        // value vectors
        vector_2d<real> values_vec_;
        vector_2d<real> trans_values_vec_;
        // time measurement
        hpx::chrono::high_resolution_timer t_ = hpx::chrono::high_resolution_timer();
        std::map<std::string, real> measurements_;
        // future vectors
        vector_future r2c_futures_;
        vector_future trans_y_to_x_futures_; 
        vector_future c2c_futures_;
        vector_future trans_x_to_y_futures_; 
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

// transpose with write running index
void fft::transpose_shared_y_to_x(const std::size_t index)
{
    for( std::size_t index_trans = 0; index_trans < dim_c_x_; ++index_trans)
    {
        trans_values_vec_(index, 2 * index_trans) = values_vec_(index_trans, 2 * index);
        trans_values_vec_(index, 2 * index_trans + 1) = values_vec_(index_trans, 2 * index + 1);
    }     
}  

// transpose with read running index
void fft::transpose_shared_x_to_y(const std::size_t index_trans)
{
    for( std::size_t index = 0; index < dim_c_x_; ++index)
    {
        values_vec_(index, 2 * index_trans) = trans_values_vec_(index_trans, 2 * index);
        values_vec_(index, 2 * index_trans + 1) = trans_values_vec_(index_trans, 2 * index + 1);
    }     
}  

// wrappers
void fft::fft_1d_r2c_inplace_wrapper(fft *th, const std::size_t i)
{
    th->fft_1d_r2c_inplace(i);
}

void fft::fft_1d_c2c_inplace_wrapper(fft *th, const std::size_t i)
{
    th->fft_1d_c2c_inplace(i);
}

void fft::transpose_shared_y_to_x_wrapper(fft *th, 
                                          const std::size_t index)
{
    th->transpose_shared_y_to_x(index);  
}

void fft::transpose_shared_x_to_y_wrapper(fft *th, 
                                          const std::size_t index_trans)
{
    th->transpose_shared_x_to_y(index_trans);  
}

// 2D FFT algorithm
vector_2d<real> fft::fft_2d_r2c()
{
    auto start_total = t_.now();
    // first dimension
    for(std::size_t i = 0; i < dim_c_x_; ++i)
    {
        // 1d FFT r2c in y-direction
        r2c_futures_[i] = hpx::async(&fft_1d_r2c_inplace_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(r2c_futures_);
    auto start_first_trans = t_.now();
    //for(std::size_t i = 0; i < dim_c_x_; ++i) for other transpose
    for(std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // transpose from y-direction to x-direction
        trans_y_to_x_futures_[i] = hpx::async(&fft::transpose_shared_y_to_x_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(trans_y_to_x_futures_);
    // second dimension
    auto start_second_fft = t_.now();
    for(std::size_t i = 0; i < dim_c_y_; ++i)
    {
        // 1D FFT in x-direction
        c2c_futures_[i] = hpx::async(&fft::fft_1d_c2c_inplace_wrapper, this, i);  
    }
    // global synchronization step
    hpx::wait_all(c2c_futures_);
    auto start_second_trans = t_.now();
    for(std::size_t i = 0; i < dim_c_y_; ++i)
    {
        trans_x_to_y_futures_[i] = hpx::async(&fft::transpose_shared_x_to_y_wrapper, this, i);
    }
    // global synchronization step
    hpx::wait_all(trans_x_to_y_futures_);
    auto stop_total = t_.now();
    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_trans - start_total;
    measurements_["first_trans"] = start_second_fft - start_first_trans;
    measurements_["second_fftw"] = start_second_trans - start_second_fft;
    measurements_["second_trans"] = stop_total - start_second_trans;

    return std::move(values_vec_);
}

// initialization
void fft::initialize(vector_2d<real> values_vec, 
                     const unsigned PLAN_FLAG)
{
    // move data into own data structure
    values_vec_ = std::move(values_vec);
    // parameters
    dim_c_x_ = values_vec_.n_row();
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = 2 * dim_c_y_ - 2;
    // resize transposed data structure
    trans_values_vec_ = std::move(vector_2d<real>(dim_c_y_, 2 * dim_c_x_));
    //create FFTW plans
    PLAN_FLAG_ = PLAN_FLAG;
    // r2c in y-direction
    plan_1d_r2c_ = fftw_plan_dft_r2c_1d(dim_r_y_,
                                       values_vec_.row(0),
                                       reinterpret_cast<fftw_complex*>(values_vec_.row(0)),
                                       PLAN_FLAG);
    // c2c in x-direction
    plan_1d_c2c_ = fftw_plan_dft_1d(dim_c_x_, 
                                   reinterpret_cast<fftw_complex*>(trans_values_vec_.row(0)), 
                                   reinterpret_cast<fftw_complex*>(trans_values_vec_.row(0)), 
                                   FFTW_FORWARD,
                                   PLAN_FLAG);
    // resize futures
    r2c_futures_.resize(dim_c_x_);
    trans_y_to_x_futures_.resize(dim_c_x_);
    c2c_futures_.resize(dim_c_y_);
    trans_x_to_y_futures_.resize(dim_c_y_);
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
    ////////////////////////////////////////////////////////////////
    // Check if shared memory
    const std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    if (std::size_t(1) != num_localities)
    {
        std::cout << "Localities " << num_localities << " instead of 1: Abort runtime\n";
        return hpx::finalize();
    }
    ////////////////////////////////////////////////////////////////
    // Parameters and Data structures
    const std::string plan_flag = vm["plan"].as<std::string>();
    bool print_result = vm["result"].as<bool>();
    bool print_header = vm["header"].as<bool>();
    // time measurement
    auto t = hpx::chrono::high_resolution_timer();
    // FFT dimension parameters
    const std::size_t dim_c_x = vm["nx"].as<std::size_t>();//N_X; 
    const std::size_t dim_r_y = vm["ny"].as<std::size_t>();//N_Y;
    const std::size_t dim_c_y = dim_r_y / 2 + 1;
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
    vector_2d<real> values_vec(dim_c_x, 2 * dim_c_y);
    for(std::size_t i = 0; i < dim_c_x; ++i)
    {
        std::iota(values_vec.row(i), values_vec.row(i+1) - 2, 0.0);
    }

    ////////////////////////////////////////////////////////////////
    // Computation   
    fft fft_computer;
    auto start_total = t.now();
    fft_computer.initialize(std::move(values_vec), FFT_BACKEND_PLAN_FLAG);
    auto stop_init = t.now();
    values_vec = fft_computer.fft_2d_r2c();
    auto stop_total = t.now();

    // optional: print results 
    if (print_result)
    {
        print_vector_2d(values_vec);
    }

    ////////////////////////////////////////////////////////////////
    // Postprocessing
    // print and store runtimes
    auto total = stop_total - start_total;
    auto init = stop_init - start_total;
    std::string msg = "\nLocality 0 - task synchronous-\n"
                      "Total runtime : {1}\n"
                      "Initialization: {2}\n"
                      "FFT 2D runtime: {3}\n"
                      "FFTW r2c      : {4}\n"
                      "First trans   : {5}\n"
                      "FFTW c2c      : {6}\n"
                      "Second trans  : {7}\n";
    hpx::util::format_to(std::cout, msg,
                        total,
                        init,
                        fft_computer.get_measurement("total"),
                        fft_computer.get_measurement("first_fftw"),
                        fft_computer.get_measurement("first_trans"),
                        fft_computer.get_measurement("second_fftw"),
                        fft_computer.get_measurement("second_trans"))
                        << std::flush;
    std::ofstream runtime_file;
    runtime_file.open ("result/runtimes_hpx_task_sync_shared.txt", std::ios_base::app);
    if(print_header)
    {
        runtime_file << "n_threads;n_x;n_y;plan;total;initialization;"
                << "fft_2d_total;"
                << "first_fftw;"
                << "first_trans;"
                << "second_fftw;"
                << "second_trans;\n";
    }
    runtime_file << hpx::get_os_thread_count() << ";" 
                << dim_c_x << ";"
                << dim_r_y << ";"
                << plan_flag << ";"
                << total << ";"
                << init << ";"
                << fft_computer.get_measurement("total") << ";"
                << fft_computer.get_measurement("first_fftw") << ";"
                << fft_computer.get_measurement("first_trans") << ";"
                << fft_computer.get_measurement("second_fftw") << ";"
                << fft_computer.get_measurement("second_trans") << ";\n";
    runtime_file.close();
    
    ////////////////////////////////////////////////////////////////
    // Finalize HPX runtime
    return hpx::finalize();
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
    ("header",value<bool>()->default_value(0), 
    "Write runtime file header");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args); 
}

#endif