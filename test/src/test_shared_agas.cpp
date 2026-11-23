#include "../../core/include/hpxfft/2D/shared/agas.hpp"
#include "../../core/include/hpxfft/util/print_vector_2d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <hpx/hpx_init.hpp>
#include <stdexcept>

using hpxfft::fft2D::shared::agas;
using real = double;

int entrypoint_test1(int argc, char *argv[])
{
    // Parameters and Data structures
    const std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    // choose dimensions consistent with the implementation:
    const std::size_t n_row = 4;
    const std::size_t n_col = 6;
    const std::size_t n_x_local = n_row / num_localities;
    hpxfft::fft2D::shared::vector_2d values_vec(n_row, n_col, 0.0);

    for (std::size_t i = 0; i < n_row; ++i)
    {
        values_vec(i, 0) = 1.0;
        values_vec(i, 1) = 2.0;
        values_vec(i, 2) = 3.0;
        values_vec(i, 3) = 4.0;
    }

    hpxfft::fft2D::shared::vector_2d values_copy = values_vec;

    // expected output
    hpxfft::fft2D::shared::vector_2d expected_output(n_row, n_col, 0.0);

    expected_output(0, 0) = 40.0;
    expected_output(0, 2) = -8.0;
    expected_output(0, 3) = 8.0;
    expected_output(0, 4) = -8.0;

    // Computation
    hpxfft::fft2D::shared::agas fft;
    std::string plan_flag = "measure";
    hpx::future<void> init_future = fft.initialize(std::move(values_vec), plan_flag);
    init_future.get();
    hpx::future<hpxfft::fft2D::shared::vector_2d> result_future = fft.fft_2d_r2c();
    hpxfft::fft2D::shared::vector_2d out1 = result_future.get();
    REQUIRE(out1 == expected_output);

    // the transform moved the data out: a second one without a new initialize throws
    REQUIRE_THROWS_AS(fft.fft_2d_r2c().get(), std::logic_error);
    // a new initialize allows one transform again
    fft.initialize(std::move(values_copy), plan_flag).get();
    REQUIRE(fft.fft_2d_r2c().get() == expected_output);

    // odd ny = 5: rows of 2 * (5 / 2 + 1) = 6 values (as for ny = 4), so ny is passed to initialize
    const std::size_t ny_odd = 5;
    hpxfft::fft2D::shared::vector_2d odd_vec(n_row, n_col, 0.0);
    for (std::size_t i = 0; i < n_row; ++i)
    {
        for (std::size_t j = 0; j < ny_odd; ++j)
        {
            odd_vec(i, j) = double(j + 1);
        }
    }
    fft.initialize(std::move(odd_vec), plan_flag, ny_odd).get();
    hpxfft::fft2D::shared::vector_2d odd_out = fft.fft_2d_r2c().get();
    REQUIRE(odd_out.n_row() == n_row);
    REQUIRE(odd_out.n_col() == n_col);
    // all rows are equal: only x frequency 0 is nonzero, n_row times the DFT of 1 .. 5 along y
    const double pi = std::acos(-1.0);
    for (std::size_t k = 0; k < n_col / 2; ++k)
    {
        std::complex<double> ref = 0.0;
        for (std::size_t m = 0; m < ny_odd; ++m)
        {
            ref += double(m + 1) * std::polar(1.0, -2.0 * pi * double(k * m) / double(ny_odd));
        }
        ref *= double(n_row);
        REQUIRE(std::abs(odd_out(0, 2 * k) - ref.real()) < 1e-12 * 60.0);
        REQUIRE(std::abs(odd_out(0, 2 * k + 1) - ref.imag()) < 1e-12 * 60.0);
        for (std::size_t i = 1; i < n_row; ++i)
        {
            REQUIRE(std::abs(odd_out(i, 2 * k)) < 1e-12 * 60.0);
            REQUIRE(std::abs(odd_out(i, 2 * k + 1)) < 1e-12 * 60.0);
        }
    }

    // an odd ny that does not match the row length is rejected: ny = 7 needs rows of 2 * (7 / 2 + 1) = 8 values
    hpxfft::fft2D::shared::vector_2d short_rows(n_row, n_col, 1.0);
    REQUIRE_THROWS_AS(fft.initialize(std::move(short_rows), plan_flag, 7).get(), std::invalid_argument);
    // a failed initialize leaves nothing to transform
    REQUIRE_THROWS_AS(fft.fft_2d_r2c().get(), std::logic_error);

    return hpx::finalize();
}

TEST_CASE("shared agas fft 2d r2c runs and produces correct output", "[shared agas][fft]")
{
    hpx::init(&entrypoint_test1, 0, nullptr);
}
