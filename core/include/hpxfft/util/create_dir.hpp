#ifndef create_dir_H_INCLUDED
#define create_dir_H_INCLUDED

#include <filesystem>

namespace hpxfft::util
{
    void create_parent_dir(std::filesystem::path file_path)
    {
        // Create parent directory if does not exist
        std::filesystem::path dir_path = file_path.parent_path();
        if (!std::filesystem::exists(dir_path)) {
            if (std::filesystem::create_directories(dir_path))
            {
                std::cout << "Directory created: " << dir_path << "\n";
            }
            else
            {
                throw std::runtime_error("Failed to create directory: " + dir_path.string());
            }
        }
    }
}
#endif // create_dir_H
