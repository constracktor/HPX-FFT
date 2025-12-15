#include "hpxfft/distributed/agas.hpp"      // for hpxfft::distributed::agas, hpxfft::distributed::vector_2d
#include "hpxfft/util/create_dir.hpp"       // for hpxfft::util::create_parent_dir
#include "hpxfft/util/print_vector_2d.hpp"  // for hpxfft::util::print_vector_2d
#include <fstream>                          // for std::ofstream
#include <hpx/hpx_init.hpp>
#include <numeric>  // for std::iota

int hpx_main(hpx::program_options::variables_map &vm)
{
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
    const std::size_t dim_c_x = vm["nx"].as<std::size_t>();  // N_X;
    const std::size_t dim_r_y = vm["ny"].as<std::size_t>();  // N_Y;
    const std::size_t dim_c_y = dim_r_y / 2 + 1;
    // division parameter
    const std::size_t n_x_local = dim_c_x / num_localities;
    ;
    // FFTW plans
    unsigned FFT_BACKEND_PLAN_FLAG = FFTW_ESTIMATE;
    if (plan_flag == "measure")
    {
        FFT_BACKEND_PLAN_FLAG = FFTW_MEASURE;
    }
    else if (plan_flag == "patient")
    {
        FFT_BACKEND_PLAN_FLAG = FFTW_PATIENT;
    }
    else if (plan_flag == "exhaustive")
    {
        FFT_BACKEND_PLAN_FLAG = FFTW_EXHAUSTIVE;
    }

    ////////////////////////////////////////////////////////////////
    // Initialization
    hpxfft::distributed::vector_2d values_vec(n_x_local, 2 * dim_c_y);
    for (std::size_t i = 0; i < n_x_local; ++i)
    {
        for (std::size_t j = 0; j < dim_r_y; ++j)
        {
            values_vec(i, j) = j;
        }
    }

    ////////////////////////////////////////////////////////////////
    // Computation
    hpxfft::distributed::agas fft_computer;
    auto start_total = t.now();
    hpx::future<void> future_initialize =
        fft_computer.initialize(std::move(values_vec), run_flag, FFT_BACKEND_PLAN_FLAG);
    future_initialize.get();
    auto stop_init = t.now();
    hpx::future<hpxfft::distributed::vector_2d> future_result = fft_computer.fft_2d_r2c();
    values_vec = future_result.get();
    auto stop_total = t.now();

    // optional: print results
    if (print_result)
    {
        sleep(this_locality);
        hpxfft::util::print_vector_2d(values_vec);
    }

    ////////////////////////////////////////////////////////////////
    // Postprocessing
    // print runtimes if on locality 0
    if (this_locality == 0)
    {
        auto total = stop_total - start_total;
        auto init = stop_init - start_total;
        auto fft2d = stop_total - stop_init;
        std::string msg =
            "\nLocality {5} - {1}\n"
            "Total runtime : {2}\n"
            "Initialization: {3}\n"
            "FFT 2D runtime: {4}\n";
        hpx::util::format_to(std::cout, msg, run_flag, total, init, fft2d, this_locality) << std::flush;

        std::string runtime_file_path = "runtimes/runtimes_hpx_distributed_agas.txt";
        hpxfft::util::create_parent_dir(runtime_file_path);
        std::ofstream runtime_file;
        runtime_file.open(runtime_file_path, std::ios_base::app);

        if (print_header)
        {
            runtime_file << "n_threads;n_x;n_y;plan;run_flag;total;initialization;" << "fft_2d_total;\n";
        }
        runtime_file << hpx::get_os_thread_count() << ";" << dim_c_x << ";" << dim_r_y << ";" << plan_flag << ";"
                     << run_flag << ";" << total << ";" << init << ";" << fft2d << ";\n";
        runtime_file.close();
    }

    ////////////////////////////////////////////////////////////////
    // Finalize HPX runtime
    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()(
        "result", value<bool>()->default_value(0), "Print generated results (default: false)")(
        "nx", value<std::size_t>()->default_value(8), "Total x dimension")(
        "ny", value<std::size_t>()->default_value(14), "Total y dimension")(
        "plan", value<std::string>()->default_value("estimate"), "FFTW plan (default: estimate)")(
        "run",
        value<std::string>()->default_value("scatter"),
        "Choose 2d FFT algorithm communication: scatter or all_to_all")(
        "header", value<bool>()->default_value(0), "Write runtime file header");

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    const std::vector<std::string> cfg = { "hpx.run_hpx_main!=1" };

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
