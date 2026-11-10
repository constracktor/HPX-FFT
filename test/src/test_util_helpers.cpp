// Small utilities: FFTW plan flag parsing, vector bounds/serialization/comparison helpers
#include "../../core/include/hpxfft/util/adapter_fftw.hpp"
#include "../../core/include/hpxfft/util/vector_2d.hpp"
#include "../../core/include/hpxfft/util/vector_3d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <vector>

namespace fa = hpxfft::util::fftw_adapter;

TEST_CASE("FFTW plan flag strings", "[util][adapter]")
{
    REQUIRE(fa::string_to_fftw_plan_flag("estimate") == fa::plan_flag::estimate);
    REQUIRE(fa::string_to_fftw_plan_flag("measure") == fa::plan_flag::measure);
    REQUIRE(fa::string_to_fftw_plan_flag("patient") == fa::plan_flag::patient);
    REQUIRE(fa::string_to_fftw_plan_flag("exhaustive") == fa::plan_flag::exhaustive);
    REQUIRE_THROWS_AS(fa::string_to_fftw_plan_flag("bogus"), std::invalid_argument);
}

TEST_CASE("vector_2d checked access and comparison", "[vector_2d]")
{
    const hpxfft::util::vector_2d<double> c(2, 3, 1.0);
    REQUIRE(c.at(1, 2) == 1.0);
    REQUIRE_THROWS_AS(c.at(2, 3), std::runtime_error);
    hpxfft::util::vector_2d<double> a(2, 3, 1.0), b(3, 2, 1.0);
    REQUIRE_FALSE(a == b);
    a.at(0, 0) = 5.0;
    REQUIRE(a.at(0, 0) == 5.0);
    REQUIRE_FALSE(a == c);
}

TEST_CASE("vector_2d swap and serialization round trip", "[vector_2d]")
{
    hpxfft::util::vector_2d<double> a(2, 3, 1.5), b(1, 1, 7.0);
    swap(a, b);
    REQUIRE(a.n_row() == 1);
    REQUIRE(b.n_col() == 3);

    std::vector<char> buffer;
    {
        hpx::serialization::output_archive oa(buffer);
        oa << b;
    }
    hpxfft::util::vector_2d<double> restored;
    {
        hpx::serialization::input_archive ia(buffer);
        ia >> restored;
    }
    REQUIRE(restored == b);
}

TEST_CASE("vector_3d checked access, rearrange and comparison", "[vector_3d]")
{
    const hpxfft::util::vector_3d<double> c(2, 3, 4, 1.0);
    REQUIRE(c.at(1, 2, 3) == 1.0);
    REQUIRE_THROWS_AS(c.at(2, 0, 0), std::runtime_error);
    hpxfft::util::vector_3d<double> a(2, 3, 4, 1.0);
    REQUIRE_THROWS_AS(a.rearrange(5, 5, 5), std::runtime_error);
    a.rearrange(4, 3, 2);
    REQUIRE_FALSE(a == c);
    hpxfft::util::vector_3d<double> d(2, 3, 4, 2.0);
    REQUIRE_FALSE(c == d);
}
