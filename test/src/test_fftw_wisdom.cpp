#include "hpxfft/util/adapter_fftw.hpp"
#include "hpxfft/util/fftw_wisdom.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace
{
constexpr int n = 48;

// plans a forward c2c FFT of size n; returns whether FFTW produced a plan
bool plan_c2c(unsigned flags)
{
    fftw_complex *buf = fftw_alloc_complex(n);
    fftw_plan p = fftw_plan_dft_1d(n, buf, buf, FFTW_FORWARD, flags);
    const bool ok = p != nullptr;
    fftw_destroy_plan(p);
    fftw_free(buf);
    return ok;
}

std::filesystem::path test_dir(const std::string &name)
{
    const auto base = std::filesystem::temp_directory_path() / ("hpxfft_wisdom_test_" + name);
    std::filesystem::remove_all(base);
    return base;
}
}  // namespace

TEST_CASE("wisdom round trip allows planning with FFTW_WISDOM_ONLY", "[util][wisdom]")
{
    const auto base = test_dir("round_trip");
    const auto file = (base / "wisdom.txt").string();

    fftw_forget_wisdom();
    hpxfft::util::fftw_adapter::c2c_1d adapter;
    fftw_complex *buf = fftw_alloc_complex(n);
    adapter.plan(n, "measure", buf, buf, hpxfft::util::fftw_adapter::direction::forward);
    fftw_free(buf);
    hpxfft::util::export_wisdom(file);
    REQUIRE(std::filesystem::is_regular_file(file));
    // no temporary file is left behind
    REQUIRE(std::distance(std::filesystem::directory_iterator(base), std::filesystem::directory_iterator{}) == 1);

    fftw_forget_wisdom();
    REQUIRE_FALSE(plan_c2c(FFTW_MEASURE | FFTW_WISDOM_ONLY));
    REQUIRE(hpxfft::util::import_wisdom(file));
    REQUIRE(plan_c2c(FFTW_MEASURE | FFTW_WISDOM_ONLY));

    fftw_forget_wisdom();
    std::filesystem::remove_all(base);
}

TEST_CASE("import_wisdom returns false for a missing file", "[util][wisdom]")
{
    const auto base = test_dir("missing");
    REQUIRE_FALSE(hpxfft::util::import_wisdom((base / "does_not_exist.txt").string()));
    REQUIRE_FALSE(hpxfft::util::import_wisdom(""));
}

TEST_CASE("import_wisdom returns false for a file that is not wisdom", "[util][wisdom]")
{
    const auto base = test_dir("garbage");
    std::filesystem::create_directories(base);
    const auto file = base / "garbage.txt";
    {
        std::ofstream(file) << "this is not fftw wisdom\n";
    }
    REQUIRE_FALSE(hpxfft::util::import_wisdom(file.string()));
    // a directory is not a wisdom file either
    REQUIRE_FALSE(hpxfft::util::import_wisdom(base.string()));
    std::filesystem::remove_all(base);
}

TEST_CASE("export_wisdom creates missing nested directories", "[util][wisdom]")
{
    const auto base = test_dir("nested");
    const auto file = base / "a" / "b" / "wisdom.txt";
    REQUIRE_NOTHROW(hpxfft::util::export_wisdom(file.string()));
    REQUIRE(std::filesystem::is_regular_file(file));
    // the exported (possibly empty) wisdom can be read back
    REQUIRE(hpxfft::util::import_wisdom(file.string()));
    std::filesystem::remove_all(base);
}

TEST_CASE("export_wisdom reports a path that cannot be written", "[util][wisdom]")
{
    const auto base = test_dir("unwritable");
    std::filesystem::create_directories(base);

    SECTION("parent directory blocked by a regular file")
    {
        {
            std::ofstream(base / "file") << "x";
        }
        // the parent exists but is not a directory: writing the file fails
        REQUIRE_THROWS_WITH(hpxfft::util::export_wisdom((base / "file" / "wisdom.txt").string()),
                            Catch::Matchers::ContainsSubstring("Failed to write FFTW wisdom file")
                                && Catch::Matchers::ContainsSubstring("Not a directory"));
    }

    SECTION("parent directory cannot be created")
    {
        {
            std::ofstream(base / "file") << "x";
        }
        REQUIRE_THROWS_WITH(hpxfft::util::export_wisdom((base / "file" / "sub" / "wisdom.txt").string()),
                            Catch::Matchers::ContainsSubstring("Failed to create directory"));
    }

    SECTION("read-only directory")
    {
        const auto dir = base / "read_only";
        std::filesystem::create_directories(dir);
        std::filesystem::permissions(dir, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
        // root ignores permissions
        if (access(dir.c_str(), W_OK) != 0)
        {
            REQUIRE_THROWS_WITH(hpxfft::util::export_wisdom((dir / "wisdom.txt").string()),
                                Catch::Matchers::ContainsSubstring("Failed to write FFTW wisdom file")
                                    && Catch::Matchers::ContainsSubstring((dir / "wisdom.txt").string()));
        }
        std::filesystem::permissions(dir, std::filesystem::perms::owner_all);
    }

    SECTION("target path is a directory")
    {
        const auto dir = base / "is_a_dir";
        std::filesystem::create_directories(dir / "inner");
        REQUIRE_THROWS_WITH(hpxfft::util::export_wisdom(dir.string()),
                            Catch::Matchers::ContainsSubstring("Failed to write FFTW wisdom file"));
        REQUIRE(std::filesystem::is_directory(dir / "inner"));
    }

    std::filesystem::remove_all(base);
}
