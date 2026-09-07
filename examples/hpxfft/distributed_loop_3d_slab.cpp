#include "hpxfft/3D/distributed/loop.hpp"  // for hpxfft::fft3D::distributed::loop, hpxfft::fft3D::distributed::vector_3d
#include "hpxfft/util/create_dir.hpp"      // for hpxfft::util::create_parent_dir
#include "hpxfft/util/print_vector_3d.hpp"  // for hpxfft::util::print_vector_3d
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
    const std::size_t dim_c_y = vm["ny"].as<std::size_t>();  // N_Y;
    const std::size_t dim_r_z = vm["nz"].as<std::size_t>();
    const std::size_t dim_c_z = dim_r_z / 2 + 1;
    // division parameter
    std::size_t n_x_local = dim_c_x / num_localities;
    ;
    std::size_t n_y_local = dim_c_y / num_localities;
    ;
    if (n_y_local * num_localities != dim_c_y || n_x_local * num_localities != dim_c_x)
    {
        std::cout << "Matrix dimensions are not divisible by number of localities, ending program" << std::endl;
        std::cout << "dim_c_x: " << dim_c_x << " n_x_local*num_localities: " << n_x_local * num_localities
                  << "dim_c_y: " << dim_c_y << " n_y_local*num_localities: " << n_y_local * num_localities << std::endl;
        return hpx::finalize();
    }

    ////////////////////////////////////////////////////////////////
    // Initialization
    hpxfft::fft3D::distributed::vector_3d values_vec =
        hpxfft::fft3D::distributed::vector_3d(n_x_local, dim_c_y, 2 * dim_c_z);
    hpxfft::fft3D::distributed::loop::slab fft_computer;

    for (std::size_t i = 0; i < n_x_local; ++i)
    {
        for (std::size_t j = 0; j < dim_c_y; ++j)
        {
            for (std::size_t k = 0; k < dim_r_z; k++)
            {
                values_vec(i, j, k) = (this_locality * n_x_local + i) * 10'000 + j * 100 + k;
            }
        }
    }

    ////////////////////////////////////////////////////////////////
    // Computation
    if (print_result)
    {
        sleep(this_locality + 1);
        print_vector_3d(values_vec);
    }

    hpx::distributed::barrier("Starting Barrier").wait();
    auto start_total = t.now();
    fft_computer.initialize(std::move(values_vec), run_flag, plan_flag);
    hpx::distributed::barrier("initialize Barrier").wait();
    auto stop_init = t.now();
    values_vec = fft_computer.fft_3d_r2c();
    auto stop_total = t.now();

    // optional: print results
    if (print_result)
    {
        sleep(this_locality + 1);
        print_vector_3d(values_vec);
    }

    ////////////////////////////////////////////////////////////////
    // Postprocessing
    // print and store runtimes if on locality 0
    auto total = stop_total - start_total;
    auto init = stop_init - start_total;
    if (this_locality == 0)
    {
        std::string msg =
            "\nLocality {15} -  {1} slab decomposition:\n"
            "Total runtime : {2}\n"
            "Initialization: {3}\n"
            "FFT 2D runtime: {4}\n"
            "FFTW r2c z    : {5}\n"
            "First permute : {6}\n"
            "FFTW c2c y    : {7}\n"
            "First split   : {8}\n"
            "First comm    : {9}\n"
            "Second permute: {10}\n"
            "FFTW c2c x    : {11}\n"
            "Second split  : {12}\n"
            "Second comm   : {13}\n"
            "Third permite : {14}\n";
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
            fft_computer.get_measurement("first_split"),
            fft_computer.get_measurement("first_comm"),
            fft_computer.get_measurement("second_permute"),
            fft_computer.get_measurement("third_fftw"),
            fft_computer.get_measurement("second_split"),
            fft_computer.get_measurement("second_comm"),
            fft_computer.get_measurement("third_permute"),
            this_locality)
            << std::flush;

        std::string runtime_file_path = "runtimes/runtimes_hpx_distributed_loop_3d_slab.txt";
        hpxfft::util::create_parent_dir(runtime_file_path);
        std::ofstream runtime_file;
        runtime_file.open(runtime_file_path, std::ios_base::app);

        if (print_header)
        {
            runtime_file << "n_ranks;n_threads;n_x;n_y;n_z;plan;comm_flag;decomposition;total;initialization;"
                         << "fft_3d_total;" << "first_fftw;" << "first_permute;" << "second_fftw;" << "first_split;"
                         << "first_comm;" << "second_permute;" << "third_fftw;" << "second_split;" << "second_comm;"
                         << "third_permute\n";
        }
        runtime_file << num_localities << ";" << hpx::get_os_thread_count() << ";" << dim_c_x << ";" << dim_c_y << ";"
                     << dim_r_z << ";" << plan_flag << ";" << run_flag << ";" << total << ";" << init << ";"
                     << fft_computer.get_measurement("total") << ";" << fft_computer.get_measurement("first_fftw")
                     << ";" << fft_computer.get_measurement("first_permute") << ";"
                     << fft_computer.get_measurement("second_fftw") << ";"
                     << fft_computer.get_measurement("first_split") << ";" << fft_computer.get_measurement("first_comm")
                     << ";" << fft_computer.get_measurement("second_permute") << ";"
                     << fft_computer.get_measurement("third_fftw") << ";"
                     << fft_computer.get_measurement("second_split") << ";"
                     << fft_computer.get_measurement("second_comm") << ";"
                     << fft_computer.get_measurement("third_permute") << ";\n";
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
        "ny", value<std::size_t>()->default_value(8), "Total y dimension")(
        "nz", value<std::size_t>()->default_value(8), "Total z dimension")(
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
