#include "../../include/hpxfft/util/create_dir.hpp"

void hpxfft::util::create_parent_dir(std::filesystem::path file_path)
{
    // Create parent directory if does not exist; a bare file name has no parent directory to create
    std::filesystem::path dir_path = file_path.parent_path();
    if (dir_path.empty() || std::filesystem::exists(dir_path))
    {
        return;
    }
    std::error_code ec;
    if (std::filesystem::create_directories(dir_path, ec))
    {
        std::cout << "Directory created: " << dir_path << "\n";
    }
    // another locality may have created the directory in the meantime
    else if (!std::filesystem::exists(dir_path))
    {
        throw std::runtime_error("Failed to create directory: " + dir_path.string() + " (" + ec.message() + ")");
    }
}
