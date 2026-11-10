#include "hpxfft/util/create_dir.hpp"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

TEST_CASE("create_parent_dir creates a missing parent directory", "[util]")
{
    const auto base = std::filesystem::temp_directory_path() / "hpxfft_create_dir_test";
    std::filesystem::remove_all(base);
    const auto file = base / "a" / "b" / "result.csv";
    hpxfft::util::create_parent_dir(file);
    REQUIRE(std::filesystem::is_directory(base / "a" / "b"));
    // an existing directory is left alone
    REQUIRE_NOTHROW(hpxfft::util::create_parent_dir(file));
    std::filesystem::remove_all(base);
}

TEST_CASE("create_parent_dir reports a directory that cannot be created", "[util]")
{
    const auto base = std::filesystem::temp_directory_path() / "hpxfft_create_dir_test_file";
    std::filesystem::remove_all(base);
    // a regular file in the way of the parent directory
    { std::ofstream(base) << "x"; }
    REQUIRE_THROWS(hpxfft::util::create_parent_dir(base / "sub" / "result.csv"));
    std::filesystem::remove_all(base);
}
