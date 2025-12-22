#ifndef create_dir_H_INCLUDED
#define create_dir_H_INCLUDED

#include <filesystem>
#include <iostream>

namespace hpxfft::util
{
void create_parent_dir(std::filesystem::path file_path);
}  // namespace hpxfft::util
#endif  // create_dir_H
