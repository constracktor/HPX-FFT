#include "hpxfft/3D/shared/loop.hpp"        // for hpxfft::fft3D::shared::loop, hpxfft::fft3D::shared::vector_3d
#include "hpxfft/util/create_dir.hpp"       // for hpxfft::util::create_parent_dir
#include "hpxfft/util/print_vector_3d.hpp"  // for hpxfft::util::print_vector_3d
#include <fstream>                          // for std::ofstream
#include <hpx/hpx_init.hpp>
#include <numeric>  // for std::iota

int hpx_main(hpx::program_options::variables_map &vm)
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
    const std::string run_flag = vm["run"].as<std::string>();
    const std::string plan_flag = vm["plan"].as<std::string>();
    bool print_result = vm["result"].as<bool>();
    bool print_header = vm["header"].as<bool>();
    // time measurement
    auto t = hpx::chrono::high_resolution_timer();
    // FFT dimension parameters
    const std::size_t dim_c_x = vm["nx"].as<std::size_t>();  // N_X;
    const std::size_t dim_c_y = vm["ny"].as<std::size_t>();  // N_Y;
    const std::size_t dim_r_z = vm["nz"].as<std::size_t>();  // N_Z;
    const std::size_t dim_c_z = dim_r_z / 2 + 1;

    ////////////////////////////////////////////////////////////////
    // Initialization
    hpxfft::fft3D::shared::vector_3d values_vec(dim_c_x, dim_c_y, 2 * dim_c_z);
    for (std::size_t i = 0; i < dim_c_x; ++i)
    {
        for (std::size_t j = 0; j < dim_c_y; ++j)
        {
            for (std::size_t k = 0; k < dim_r_z; ++k)
            {
                values_vec(i, j, k) = k;
            }
        }
    }

    ////////////////////////////////////////////////////////////////
    // Computation
    hpxfft::fft3D::shared::loop fft_computer;
    auto start_total = t.now();
    fft_computer.initialize(std::move(values_vec), plan_flag);
    auto stop_init = t.now();
    if (run_flag == "seq")
    {
        values_vec = fft_computer.fft_3d_r2c_seq();
    }
    else
    {
        values_vec = fft_computer.fft_3d_r2c_par();
    }
    auto stop_total = t.now();

    // optional: print results
    if (print_result)
    {
        hpxfft::util::print_vector_3d(values_vec);
    }

    ////////////////////////////////////////////////////////////////
    // Postprocessing
    // print and store runtimes
    auto total = stop_total - start_total;
    auto init = stop_init - start_total;
    std::string msg =
        "\nLocality 0 - shared - {1}\n"
        "Total runtime : {2}\n"
        "Initialization: {3}\n"
        "FFT 3D runtime: {4}\n"
        "FFTW r2c      : {5}\n"
        "First permute : {6}\n"
        "FFTW c2c 1    : {7}\n"
        "Second permute: {8}\n"
        "FFTW c2c 2    : {9}\n"
        "Third permute : {10}\n"
        "Plan time     : {11}\n"
        "Plan flops    : {12}\n";
    hpx::util::format_to(
        std::cout,
        msg,
        run_flag,
        total,
        init,
        fft_computer.get_measurement("total"),
        fft_computer.get_measurement("first_fftw"),
        fft_computer.get_measurement("first_permute"),
        fft_computer.get_measurement("second_fftw"),
        fft_computer.get_measurement("second_permute"),
        fft_computer.get_measurement("third_fftw"),
        fft_computer.get_measurement("third_permute"),
        fft_computer.get_measurement("plan"),
        fft_computer.get_measurement("plan_flops"))
        << std::flush;

    std::string runtime_file_path = "runtimes/runtimes_hpx_shared_loop_3d.txt";
    hpxfft::util::create_parent_dir(runtime_file_path);
    std::ofstream runtime_file;
    runtime_file.open(runtime_file_path, std::ios_base::app);

    if (print_header)
    {
        runtime_file << "n_threads;n_x;n_y;n_z;plan;run_flag;total;initialization;" << "fft_3d_total;" << "first_fftw;"
                     << "first_permute;" << "second_fftw;" << "second_permute;" << "third_fftw;" << "third_permute;"
                     << "plan_time;" << "plan_flops;\n";
    }
    runtime_file << hpx::get_os_thread_count() << ";" << dim_c_x << ";" << dim_c_y << ";" << dim_r_z << ";" << plan_flag
                 << ";" << run_flag << ";" << total << ";" << init << ";" << fft_computer.get_measurement("total")
                 << ";" << fft_computer.get_measurement("first_fftw") << ";"
                 << fft_computer.get_measurement("first_permute") << ";" << fft_computer.get_measurement("second_fftw")
                 << ";" << fft_computer.get_measurement("second_permute") << ";"
                 << fft_computer.get_measurement("third_fftw") << ";" << fft_computer.get_measurement("third_permute")
                 << ";" << fft_computer.get_measurement("plan") << ";" << fft_computer.get_measurement("plan_flops")
                 << ";\n";
    runtime_file.close();

    // store plan info
    std::string plan_file_path = "plans/plan_hpx_shared_loop_3d.txt";
    hpxfft::util::create_parent_dir(plan_file_path);
    std::ofstream plan_info_file;
    plan_info_file.open(plan_file_path, std::ios_base::app);
    plan_info_file
        << "n_threads;n_x;n_y;n_z;plan;run_flag;total;initialization;" << "fft_3d_total;" << "first_fftw;"
        << "first_permute;" << "second_fftw;" << "second_permute;" << "third_fftw;" << "third_permute;" << "plan_time;"
        << "plan_flops;\n"
        << hpx::get_os_thread_count() << ";" << dim_c_x << ";" << dim_c_y << ";" << dim_r_z << ";" << plan_flag << ";"
        << run_flag << ";" << total << ";" << init << ";" << fft_computer.get_measurement("total") << ";"
        << fft_computer.get_measurement("first_fftw") << ";" << fft_computer.get_measurement("first_permute") << ";"
        << fft_computer.get_measurement("second_fftw") << ";" << fft_computer.get_measurement("second_permute") << ";"
        << fft_computer.get_measurement("third_fftw") << ";" << fft_computer.get_measurement("third_permute") << ";"
        << fft_computer.get_measurement("plan") << ";" << fft_computer.get_measurement("plan_flops") << ";\n";
    plan_info_file.close();
    // store plan
    fft_computer.write_plans_to_file(plan_file_path);

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
        "nz", value<std::size_t>()->default_value(16), "Total z dimension")(
        "plan", value<std::string>()->default_value("estimate"), "FFTW plan (default: estimate)")(
        "run", value<std::string>()->default_value("par"), "Choose 2d FFT algorithm: par or seq")(
        "header", value<bool>()->default_value(0), "Write runtime file header");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
