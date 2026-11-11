#include "../../core/include/hpxfft/3D/shared/sync.hpp"
#include "../../core/include/hpxfft/util/print_vector_3d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <hpx/hpx_init.hpp>

using hpxfft::fft3D::shared::sync;
using real = double;

int entrypoint_test1(int argc, char *argv[])
{
    // Parameters and Data structures
    const std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    // choose dimensions consistent with the implementation:
    const std::size_t n_x = 3;
    const std::size_t n_y = 5;
    const std::size_t n_z_r = 4;
    const std::size_t n_z_c = n_z_r / 2 + 1;
    hpxfft::fft3D::shared::vector_3d values_vec(n_x, n_y, 2 * n_z_c, 0.0);

    for (std::size_t i = 0; i < n_x; ++i)
    {
        for (std::size_t j = 0; j < n_y; ++j)
        {
            for (std::size_t k = 0; k < n_z_r; ++k)
            {
                values_vec(i, j, k) = k;
            }
        }
    }

    // expected output
    hpxfft::fft3D::shared::vector_3d expected_output(n_x, n_y, 2 * n_z_c, 0.0);

    expected_output(0, 0, 0) = 90.0;
    expected_output(0, 0, 2) = -30.0;
    expected_output(0, 0, 3) = 30.0;
    expected_output(0, 0, 4) = -30.0;

    // Computation
    hpxfft::fft3D::shared::sync fft;
    std::string plan_flag = "estimate";
    fft.initialize(std::move(values_vec), plan_flag);
    hpxfft::fft3D::shared::vector_3d out = fft.fft_3d_r2c();
    auto total = fft.get_measurement(std::string("total"));
    auto flops = fft.get_measurement(std::string("plan_flops"));
    REQUIRE(total >= 0.0);
    REQUIRE(out == expected_output);

    return hpx::finalize();
}

TEST_CASE("shared sync fft 3d r2c runs and produces correct output", "[shared sync][3D][fft]")
{
    hpx::init(&entrypoint_test1, 0, nullptr);
}
