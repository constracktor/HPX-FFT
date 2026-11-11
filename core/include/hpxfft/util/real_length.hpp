#pragma once
#ifndef hpxfft_util_real_length_H_INCLUDED
#define hpxfft_util_real_length_H_INCLUDED

#include <cstddef>
#include <stdexcept>
#include <string>

namespace hpxfft::util
{
// Number of real values per row of a real-to-complex transform whose rows hold dim_c complex values (dim_c = n / 2 + 1).
// n = 0 derives the even length 2 * (dim_c - 1); otherwise n must match dim_c. `name` is used in the error message.
inline std::size_t real_length(const std::size_t dim_c, const std::size_t n, const char *name)
{
    const std::size_t length = n != 0 ? n : 2 * dim_c - 2;
    if (length / 2 + 1 != dim_c)
    {
        throw std::invalid_argument(std::string(name) + " does not match the row length of the data (2 * (" + name +
                                    " / 2 + 1) values)");
    }
    return length;
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_real_length_H_INCLUDED
