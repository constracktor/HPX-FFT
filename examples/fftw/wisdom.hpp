#pragma once
// FFTW wisdom for the FFTW reference examples: planning with measure, patient or exhaustive is slow, wisdom stores
// what the planner learned so that later runs of the same problem plan almost instantly and use the same plans.
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fftw_wisdom
{
// Wisdom file of one benchmark configuration. Returns explicit_path if it is not empty. Otherwise, if the environment
// variable FFTW_WISDOM_DIR is set and plan is not "estimate" (estimate does not measure, so wisdom adds nothing),
// returns <FFTW_WISDOM_DIR>/<tag>_<plan>_<dims joined by x>_n<n_ranks>_t<n_threads>.wis, else "" (no wisdom).
inline std::string file_name(const std::string &explicit_path,
                             const std::string &tag,
                             const std::string &plan,
                             std::initializer_list<std::size_t> dims,
                             std::size_t n_ranks,
                             std::size_t n_threads)
{
    if (!explicit_path.empty())
    {
        return explicit_path;
    }
    const char *dir = std::getenv("FFTW_WISDOM_DIR");
    if (dir == nullptr || *dir == '\0' || plan == "estimate")
    {
        return "";
    }
    std::string name = std::string(dir) + "/" + tag + "_" + plan + "_";
    bool first = true;
    for (const std::size_t d : dims)
    {
        name += (first ? "" : "x") + std::to_string(d);
        first = false;
    }
    return name + "_n" + std::to_string(n_ranks) + "_t" + std::to_string(n_threads) + ".wis";
}

// merge the wisdom file into this process; false if the file is missing or invalid
inline bool import_file(const std::string &path)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
    {
        return false;
    }
    return fftw_import_wisdom_from_filename(path.c_str()) != 0;
}

// write the wisdom of this process to the file (temporary file + rename, so readers never see a partial file)
inline void export_file(const std::string &path)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    const std::string tmp_path = path + ".tmp." + host + "." + std::to_string(getpid());
    if (fftw_export_wisdom_to_filename(tmp_path.c_str()) == 0)
    {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        throw std::runtime_error("Failed to write FFTW wisdom file: " + path);
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec)
    {
        std::error_code ignored;
        std::filesystem::remove(tmp_path, ignored);
        throw std::runtime_error("Failed to write FFTW wisdom file: " + path + " (" + ec.message() + ")");
    }
}
}  // namespace fftw_wisdom
