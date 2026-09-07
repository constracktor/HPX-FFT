#ifndef print_vector_3d_H_INCLUDED
#define print_vector_3d_H_INCLUDED

#include "vector_3d.hpp"
#include <iostream>

typedef double real;

namespace hpxfft::util
{

inline void print_vector_3d(const vector_3d<real> &input)
{
    const std::string msg = "\n";

    const std::size_t dim_x = input.n_x();
    const std::size_t dim_y = input.n_y();
    const std::size_t dim_z = input.n_z();

    std::size_t counter = 0;
    for (std::size_t i = 0; i < dim_x; ++i)
    {
        std::string layer_msg = "[{1},:,:]:\n";
        hpx::util::format_to(std::cout, layer_msg, i) << std::flush;
        for (std::size_t j = 0; j < dim_y; ++j)
        {
            for (std::size_t k = 0; k < dim_z; ++k)
            {
                real element = input(i, j, k);
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
    hpx::util::format_to(std::cout, msg) << std::flush;
}
}  // namespace hpxfft::util
#endif  // print_vector_3d_H_INCLUDED
