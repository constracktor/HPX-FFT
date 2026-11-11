#ifndef print_vector_2d_H_INCLUDED
#define print_vector_2d_H_INCLUDED

#include "vector_2d.hpp"
#include <iostream>

namespace hpxfft::util
{
template <typename T>
void print_vector_2d(const vector_2d<T> &input)
{
    const std::string msg = "\n";

    const std::size_t dim_x = input.n_row();
    const std::size_t dim_y = input.n_col();

    std::size_t counter = 0;
    for (std::size_t i = 0; i < dim_x; ++i)
    {
        for (std::size_t j = 0; j < dim_y; ++j)
        {
            real element = input(i, j);
            if (counter % 2 == 0)
            {
                std::string msg = "({1} ";
                hpx::util::format_to(std::cout, msg, element) << std::flush;
            }
            else
            {
                std::string msg = "{1}) ";
                hpx::util::format_to(std::cout, msg, element) << std::flush;
            }
            ++counter;
        }
        hpx::util::format_to(std::cout, msg) << std::flush;
    }
    hpx::util::format_to(std::cout, msg) << std::flush;
}
}  // namespace hpxfft::util
#endif  // print_vector_2d_H_INCLUDED
