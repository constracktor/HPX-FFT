#include "../../core/include/hpxfft/distributed/loop.hpp"
#include "../../core/include/hpxfft/util/print_vector.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <fftw3.h>
#include <hpx/hpx_init.hpp>

using hpxfft::distributed::loop;
using real = double;

int entrypoint_test1(int argc, char *argv[])
{
    // Parameters and Data structures
    const std::size_t this_locality = hpx::get_locality_id();
    const std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    // choose dimensions consistent with the implementation:
    const std::size_t n_row = 4;
    const std::size_t n_col = 6;
    const std::size_t n_x_local = n_row / num_localities;
    hpxfft::distributed::vector_2d values_vec(n_x_local, n_col, 0.0);

    for (std::size_t i = 0; i < n_x_local; ++i)
    {
        values_vec(i, 0) = 1.0;
        values_vec(i, 1) = 2.0;
        values_vec(i, 2) = 3.0;
        values_vec(i, 3) = 4.0;
    }

    // expected output
    hpxfft::distributed::vector_2d expected_output(n_x_local, n_col, 0.0);

    if (this_locality == 0)
    {
        expected_output(0, 0) = 40.0;
        expected_output(0, 2) = -8.0;
        expected_output(0, 3) = 8.0;
        expected_output(0, 4) = -8.0;
    }

    // Computation
    hpxfft::distributed::loop fft;
    std::string plan_flag = "estimate";
    fft.initialize(std::move(values_vec), "scatter", plan_flag);
    values_vec = fft.fft_2d_r2c();
    auto total = fft.get_measurement(std::string("total"));
    REQUIRE(total >= 0.0);
    REQUIRE(values_vec == expected_output);

    return hpx::finalize();
}

TEST_CASE("distributed loop fft 2d r2c runs and produces correct output", "[distributed loop][fft]")
{
    hpx::init(&entrypoint_test1, 0, nullptr);
}
