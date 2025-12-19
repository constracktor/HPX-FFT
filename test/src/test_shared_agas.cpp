#include "../../core/include/hpxfft/shared/agas.hpp"
#include "../../core/include/hpxfft/util/print_vector_2d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <hpx/hpx_init.hpp>

using hpxfft::shared::agas;
using real = double;

int entrypoint_test1(int argc, char *argv[])
{
    // Parameters and Data structures
    const std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    // choose dimensions consistent with the implementation:
    const std::size_t n_row = 4;
    const std::size_t n_col = 6;
    const std::size_t n_x_local = n_row / num_localities;
    hpxfft::shared::vector_2d values_vec(n_row, n_col, 0.0);

    for (std::size_t i = 0; i < n_row; ++i)
    {
        values_vec(i, 0) = 1.0;
        values_vec(i, 1) = 2.0;
        values_vec(i, 2) = 3.0;
        values_vec(i, 3) = 4.0;
    }

    // expected output
    hpxfft::shared::vector_2d expected_output(n_row, n_col, 0.0);

    expected_output(0, 0) = 40.0;
    expected_output(0, 2) = -8.0;
    expected_output(0, 3) = 8.0;
    expected_output(0, 4) = -8.0;

    // Computation
    hpxfft::shared::agas fft;
    std::string plan_flag = "measure";
    hpx::future<void> init_future = fft.initialize(std::move(values_vec), plan_flag);
    init_future.get();
    hpx::future<hpxfft::shared::vector_2d> result_future = fft.fft_2d_r2c();
    hpxfft::shared::vector_2d out1 = result_future.get();
    REQUIRE(out1 == expected_output);

    return hpx::finalize();
}

TEST_CASE("shared agas fft 2d r2c runs and produces correct output", "[shared agas][fft]")
{
    hpx::init(&entrypoint_test1, 0, nullptr);
}
