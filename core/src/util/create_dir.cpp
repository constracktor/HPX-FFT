#include "../../include/hpxfft/util/create_dir.hpp"

#include <system_error>

void hpxfft::util::create_parent_dir(std::filesystem::path file_path)
{
    // Create parent directory if does not exist; a bare file name has no parent directory to create
    std::filesystem::path dir_path = file_path.parent_path();
    // only the non-throwing overloads: an error of exists (e.g. no permission to search a parent) counts as "does not
    // exist", and create_directories reports it
    std::error_code exists_ec;
    if (dir_path.empty() || std::filesystem::exists(dir_path, exists_ec))
    {
        return;
    }
    std::error_code ec;
    if (std::filesystem::create_directories(dir_path, ec))
    {
        std::cout << "Directory created: " << dir_path << "\n";
    }
    // another locality may have created the directory in the meantime
    else if (!std::filesystem::exists(dir_path, exists_ec))
    {
        throw std::runtime_error("Failed to create directory: " + dir_path.string() + " (" + ec.message() + ")");
    }
}
