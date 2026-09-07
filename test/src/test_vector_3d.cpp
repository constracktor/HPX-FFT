#define CATCH_CONFIG_MAIN
#include "../../core/include/hpxfft/util/vector_3d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

TEST_CASE("Vector 3D constant: Initialization", "[vector_3d][init]")
{
    hpxfft::util::vector_3d<double> vec(2, 2, 2, 3.0);

    REQUIRE(vec.n_x() == 2);
    REQUIRE(vec.n_y() == 2);
    REQUIRE(vec.n_z() == 2);
    REQUIRE(vec(0, 0, 0) == 3.0);
    REQUIRE(vec(1, 1, 1) == 3.0);
    REQUIRE(vec(0, 1, 1) == 3.0);
    REQUIRE(vec.size() == 8);
}

TEST_CASE("Vector 3D: Access Out of Range", "[vector_3d][exception]")
{
    hpxfft::util::vector_3d<double> vec(2, 2, 2, 1.0);

    REQUIRE_THROWS_AS(vec.at(2, 2, 2), std::runtime_error);
}

TEST_CASE("Compare two Vector 3D instances", "[vector_3d][compare]")
{
    hpxfft::util::vector_3d<double> vec1(2, 2, 2, 5.0);
    hpxfft::util::vector_3d<double> vec2(2, 2, 2, 5.0);
    hpxfft::util::vector_3d<double> vec3(2, 2, 2, 6.0);

    REQUIRE(vec1 == vec2);
    REQUIRE(!(vec1 == vec3));
}
