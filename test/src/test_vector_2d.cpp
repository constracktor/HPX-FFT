#define CATCH_CONFIG_MAIN
#include "../../core/include/hpxfft/util/vector_2d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

TEST_CASE("Vector 2D constant: Initialization", "[vector_2d][init]")
{
    hpxfft::util::vector_2d<double> vec(3, 3, 4.0);

    REQUIRE(vec.n_row() == 3);
    REQUIRE(vec.n_col() == 3);
    REQUIRE(vec(0, 0) == 4.0);
    REQUIRE(vec(2, 2) == 4.0);
    REQUIRE(vec(1, 2) == 4.0);
    REQUIRE(vec.size() == 9);
}

TEST_CASE("Vector 2D: Access Out of Range", "[vector_2d][exception]")
{
    hpxfft::util::vector_2d<double> vec(3, 3, 1.0);

    REQUIRE_THROWS_AS(vec.at(3, 3), std::runtime_error);
}

TEST_CASE("Compare two Vector 2D instances", "[vector_2d][compare]")
{
    hpxfft::util::vector_2d<double> vec1(2, 2, 5.0);
    hpxfft::util::vector_2d<double> vec2(2, 2, 5.0);
    hpxfft::util::vector_2d<double> vec3(2, 2, 6.0);

    REQUIRE(vec1 == vec2);
    REQUIRE(!(vec1 == vec3));
}
