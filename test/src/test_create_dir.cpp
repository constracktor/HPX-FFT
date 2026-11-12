#include "hpxfft/util/create_dir.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <unistd.h>

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

TEST_CASE("create_parent_dir accepts a bare file name", "[util]")
{
    REQUIRE_NOTHROW(hpxfft::util::create_parent_dir("result.csv"));
}

TEST_CASE("create_parent_dir reports a directory that cannot be created", "[util]")
{
    const auto base = std::filesystem::temp_directory_path() / "hpxfft_create_dir_test_file";
    std::filesystem::remove_all(base);
    // a regular file in the way of the parent directory
    {
        std::ofstream(base) << "x";
    }
    REQUIRE_THROWS_WITH(hpxfft::util::create_parent_dir(base / "sub" / "result.csv"),
                        Catch::Matchers::ContainsSubstring("Failed to create directory"));
    std::filesystem::remove_all(base);
}

TEST_CASE("create_parent_dir reports a parent that cannot be searched", "[util]")
{
    const auto base = std::filesystem::temp_directory_path() / "hpxfft_create_dir_test_locked";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base / "locked");
    // without search permission even checking whether locked/sub exists fails: this must not escape as
    // std::filesystem::filesystem_error but be reported as a directory that cannot be created
    std::filesystem::permissions(base / "locked", std::filesystem::perms::none);
    // root ignores permissions
    if (access((base / "locked").c_str(), X_OK) != 0)
    {
        REQUIRE_THROWS_WITH(hpxfft::util::create_parent_dir(base / "locked" / "sub" / "result.csv"),
                            Catch::Matchers::ContainsSubstring("Failed to create directory"));
    }
    std::filesystem::permissions(base / "locked", std::filesystem::perms::owner_all);
    std::filesystem::remove_all(base);
}
