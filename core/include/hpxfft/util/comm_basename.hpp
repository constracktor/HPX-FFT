#pragma once
#ifndef hpxfft_util_comm_basename_H_INCLUDED
#define hpxfft_util_comm_basename_H_INCLUDED

#include <atomic>
#include <cstddef>
#include <string>

namespace hpxfft::util
{
// Prefix of the communicator basenames of one initialize call: "<tag><n>/", where n counts the calls in this process.
// Without it every object would register its communicators under the same names, and a second object (created while
// another one is alive, or after one whose names are still being unregistered asynchronously) could resolve to the
// communicators of the other one. Call it exactly once per initialize, at the same point on every locality: all
// localities create and initialize their objects in the same order (as for check_equal_local_shape), so they agree on
// n.
inline std::string unique_basename_prefix(const char *tag)
{
    static std::atomic<std::size_t> counter{ 0 };
    return std::string(tag) + std::to_string(counter++) + "/";
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_comm_basename_H_INCLUDED
