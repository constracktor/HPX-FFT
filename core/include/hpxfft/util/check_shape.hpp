#pragma once
#ifndef hpxfft_util_check_shape_H_INCLUDED
#define hpxfft_util_check_shape_H_INCLUDED

#include <atomic>
#include <cstdint>
#include <hpx/modules/collectives.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpxfft::util
{
// Collective, called once during initialization on every locality: throws std::invalid_argument on every locality if
// the local blocks do not have the same shape everywhere. The distributed loops derive the global size from the local
// block, so unequal blocks would otherwise give wrong results or hangs.
inline void check_equal_local_shape(const std::vector<std::uint64_t> &shape)
{
    static std::atomic<std::size_t> counter{ 0 };
    const std::size_t num_sites = hpx::get_num_localities(hpx::launch::sync);
    if (num_sites == 1)
    {
        return;
    }
    const std::string basename = "hpxfft_shape_check_" + std::to_string(counter++);
    std::vector<std::vector<std::uint64_t>> all_shapes =
        hpx::collectives::all_gather(
            basename.c_str(),
            shape,
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(hpx::get_locality_id()),
            hpx::collectives::generation_arg(1))
            .get();
    for (const auto &s : all_shapes)
    {
        if (s != shape)
        {
            throw std::invalid_argument("The local blocks do not have the same shape on all localities");
        }
    }
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_check_shape_H_INCLUDED
