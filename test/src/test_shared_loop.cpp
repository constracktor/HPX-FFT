#include "../../core/include/hpxfft/shared/loop.hpp"
#include "../../core/include/hpxfft/util/print_vector_2d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <fftw3.h>
#include <hpx/hpx_init.hpp>

using hpxfft::shared::loop;
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
    hpxfft::shared::vector_2d expected_output(n_x_local, n_col, 0.0);

    expected_output(0, 0) = 40.0;
    expected_output(0, 2) = -8.0;
    expected_output(0, 3) = 8.0;
    expected_output(0, 4) = -8.0;

    // Computation
    /*
    hpxfft::shared::loop fft1;
    unsigned plan_flag = FFTW_ESTIMATE;
    fft1.initialize(std::move(values_vec), plan_flag);
    hpxfft::shared::vector_2d out1 = fft1.fft_2d_r2c_seq();
    auto total = fft1.get_measurement(std::string("total"));
    REQUIRE(total >= 0.0);
    REQUIRE(out1 == expected_output);
    */
    hpxfft::shared::loop fft2;
    unsigned plan_flag = FFTW_ESTIMATE;
    fft2.initialize(std::move(values_vec), plan_flag);
    hpxfft::shared::vector_2d out2 = fft2.fft_2d_r2c_par();
    auto total = fft2.get_measurement(std::string("total"));
    REQUIRE(total >= 0.0);
    REQUIRE(out2 == expected_output);

    return hpx::finalize();
}

TEST_CASE("shared loop fft 2d r2c seq and par runs and produces correct output", "[shared loop][fft]")
{
    hpx::init(&entrypoint_test1, 0, nullptr);
}
