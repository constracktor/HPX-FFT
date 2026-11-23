#pragma once
#ifndef hpxfft_util_comm_flag_H_INCLUDED
#define hpxfft_util_comm_flag_H_INCLUDED

#include <initializer_list>
#include <stdexcept>
#include <string>

namespace hpxfft::util
{
// Throw std::invalid_argument unless `flag` is one of `supported`. `name` (the class) is used in the error message.
// The run flag is the same on every locality, so call it in initialize before any collective operation: then every
// locality throws and none waits for the others.
inline void check_comm_flag(const std::string &flag, std::initializer_list<const char *> supported, const char *name)
{
    std::string list;
    for (const char *s : supported)
    {
        if (flag == s)
        {
            return;
        }
        list += list.empty() ? s : std::string(", ") + s;
    }
    throw std::invalid_argument(
        "Unsupported communication flag '" + flag + "' for " + name + " (supported: " + list + ")");
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_comm_flag_H_INCLUDED
