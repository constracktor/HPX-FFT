#pragma once
#ifndef hpxfft_util_measurement_H_INCLUDED
#define hpxfft_util_measurement_H_INCLUDED

#include "real.hpp"
#include <map>
#include <stdexcept>
#include <string>

namespace hpxfft::util
{
// Value of the measurement `name` (the get_measurement of the FFT classes). Throws std::out_of_range for a name that
// has not been recorded: a misspelled name, or a timing that is only recorded by initialize or by the transform before
// that ran. The message lists the names that are available.
inline real get_measurement(const std::map<std::string, real> &measurements, const std::string &name)
{
    const auto it = measurements.find(name);
    if (it == measurements.end())
    {
        std::string available;
        for (const auto &entry : measurements)
        {
            available += (available.empty() ? "" : ", ") + entry.first;
        }
        throw std::out_of_range("Unknown measurement '" + name + "'; available: "
                                + (available.empty() ? "none (run initialize and the transform first)" : available));
    }
    return it->second;
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_measurement_H_INCLUDED
