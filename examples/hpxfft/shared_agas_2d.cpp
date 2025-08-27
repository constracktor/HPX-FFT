#include <numeric> // for std::iota
#include <fstream> // for std::ofstream

#include <hpx/hpx_init.hpp>

#include "../../core/include/hpxfft/shared/hpxfft_shared_agas.hpp" // for hpxfft::shared::agas, hpx::shared::vector_2d
#include "../../core/include/hpxfft/util/print_vector_2d.hpp" // for hpxfft::util::print_vector_2d
#include "../../core/include/hpxfft/util/create_dir.hpp" // for hpxfft::util::create_parent_dir

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
    hpxfft::shared::vector_2d values_vec(dim_c_x, 2 * dim_c_y);
    for(std::size_t i = 0; i < dim_c_x; ++i)
    {
        std::iota(values_vec.row(i), values_vec.row(i+1) - 2, 0.0);
    }

    ////////////////////////////////////////////////////////////////
    // Computation    
    hpxfft::shared::agas fft_computer;
    auto start_total = t.now();
    hpx::future<void> future_initialize = \
    fft_computer.initialize(std::move(values_vec), 
                            FFT_BACKEND_PLAN_FLAG);
    future_initialize.get();
    auto stop_init = t.now();
    hpx::future<hpxfft::shared::vector_2d> future_result = fft_computer.fft_2d_r2c();
    values_vec = future_result.get();
    auto stop_total = t.now();

    // optional: print results 
    if (print_result)
    {
        hpxfft::util::print_vector_2d(values_vec);
    }

    ////////////////////////////////////////////////////////////////
    // Postprocessing
    // print and store runtimes
    auto total = stop_total - start_total;
    auto init = stop_init - start_total;
    auto fft2d = stop_total - stop_init;
    std::string msg = "\nLocality 0 - agas shared\n"
                      "Total runtime : {1}\n"
                      "Initialization: {2}\n"
                      "FFT 2D runtime: {3}\n";
    hpx::util::format_to(std::cout, msg,  
                         total,
                         init,
                         fft2d)
                        << std::flush;

    std::string runtime_file_path = "result/runtimes/runtimes_hpx_shared_agas.txt";
    hpxfft::util::create_parent_dir(runtime_file_path);
    std::ofstream runtime_file;
    runtime_file.open (runtime_file_path, std::ios_base::app);
    
    if(print_header)
    {
        runtime_file << "n_threads;n_x;n_y;plan;total;initialization;"
                << "fft_2d_total;\n";
    }
    runtime_file << hpx::get_os_thread_count() << ";" 
                << dim_c_x << ";"
                << dim_r_y << ";"
                << plan_flag << ";"
                << total << ";"
                << init << ";"
                << fft2d << ";\n";
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
