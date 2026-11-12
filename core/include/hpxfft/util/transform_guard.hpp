#pragma once
#ifndef hpxfft_util_transform_guard_H_INCLUDED
#define hpxfft_util_transform_guard_H_INCLUDED

#include <stdexcept>
#include <string>

namespace hpxfft::util
{
// Every initialize allows one transform: the transform moves the result (the data of the object) out of the object.
// The classes keep a flag that initialize clears at its start and sets at its end; every transform entry point calls
// this first. It throws std::logic_error if the flag is not set (no initialize, a failed one, or a second transform)
// and clears it otherwise. `name` (the transform and the class) is used in the error message.
inline void begin_transform(bool &ready, const std::string &name)
{
    if (!ready)
    {
        throw std::logic_error(name
                               + " needs a preceding initialize: every initialize allows one transform, which moves "
                                 "the data out of the object");
    }
    ready = false;
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_transform_guard_H_INCLUDED
