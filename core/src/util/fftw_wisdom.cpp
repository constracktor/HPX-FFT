#include "../../include/hpxfft/util/fftw_wisdom.hpp"

#include "../../include/hpxfft/util/adapter_fftw.hpp"
#include "../../include/hpxfft/util/create_dir.hpp"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

bool hpxfft::util::import_wisdom(const std::string &path)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(fftw_adapter::planner_mutex());
    // fftw_import_wisdom_from_filename returns 0 for unreadable or invalid files
    return fftw_import_wisdom_from_filename(path.c_str()) != 0;
}

void hpxfft::util::export_wisdom(const std::string &path)
{
    create_parent_dir(path);
    // per-process temporary name: localities on other nodes may share the file system and the pid
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    const std::string tmp_path = path + ".tmp." + host + "." + std::to_string(getpid());
    {
        std::lock_guard<std::mutex> lock(fftw_adapter::planner_mutex());
        errno = 0;
        if (fftw_export_wisdom_to_filename(tmp_path.c_str()) == 0)
        {
            const std::string reason = errno != 0 ? std::strerror(errno) : "unknown error";
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            throw std::runtime_error(
                "Failed to write FFTW wisdom file: " + path + " (temporary file " + tmp_path + ": " + reason + ")");
        }
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
