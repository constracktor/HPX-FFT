#pragma once
#ifndef hpxfft_util_blocks_H_INCLUDED
#define hpxfft_util_blocks_H_INCLUDED

#include <cstddef>
#include <vector>

namespace hpxfft::util
{
// Offsets of the blocks of a dimension of n elements split into `parts` blocks: block i covers
// [offsets[i], offsets[i + 1]). The first n % parts blocks hold one element more, so any n can be split.
inline std::vector<std::size_t> block_offsets(const std::size_t n, const std::size_t parts)
{
    std::vector<std::size_t> offsets(parts + 1, 0);
    for (std::size_t i = 0; i < parts; ++i)
    {
        offsets[i + 1] = offsets[i] + n / parts + (i < n % parts ? 1 : 0);
    }
    return offsets;
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_blocks_H_INCLUDED
