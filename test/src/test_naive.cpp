#include <catch2/catch_test_macros.hpp>
#include <fftw3.h>
#include <cmath>
#include "utils.cpp"

#include "../../core/include/hpxfft/shared/naive.hpp"
#include "../../core/include/hpxfft/util/vector_2d.hpp"

using hpxfft::shared::naive;
using hpxfft::util::vector_2d;
using real = double;

TEST_CASE("naive fft 2d r2c runs and produces measurements", "[naive][fft]") {
    // choose dimensions consistent with the implementation:
    // n_col must be even; the code computes dim_c_y = n_col/2 and dim_r_y = 2*dim_c_y - 2
    const std::size_t n_row = 4;
    const std::size_t n_col = 6; // even, >= 2

    vector_2d<real> input(n_row, n_col, 0.0);
    // simple impulse input
    input(0, 0) = 1.0;
    utils::start_hpx_runtime(0, nullptr);

    naive fft;
    unsigned plan_flag = FFTW_ESTIMATE;
    fft.initialize(std::move(input), plan_flag);

    auto out = fft.fft_2d_r2c();

    REQUIRE(out.n_row() == n_row);
    REQUIRE(out.n_col() == n_col);

    // a timing measurement named "total" should be present and non-negative
    auto total = fft.get_measurement(std::string("total"));
    utils::stop_hpx_runtime();
    REQUIRE(total >= 0.0);
}

TEST_CASE("naive fft 2d of zero input yields near-zero output", "[naive][fft][zero]") {
    const std::size_t n_row = 3;
    const std::size_t n_col = 8; // even

    vector_2d<real> input(n_row, n_col, 0.0);

    utils::start_hpx_runtime(0, nullptr);

    naive fft;
    unsigned plan_flag = FFTW_ESTIMATE;
    fft.initialize(std::move(input), plan_flag);

    auto out = fft.fft_2d_r2c();
    utils::stop_hpx_runtime();
    // All outputs should be (near) zero; use a small tolerance
    const double tol = 1e-12;
    for (std::size_t i = 0; i < out.n_row(); ++i) {
        for (std::size_t j = 0; j < out.n_col(); ++j) {
            REQUIRE(std::abs(out(i, j)) <= tol);
        }
    }
}
