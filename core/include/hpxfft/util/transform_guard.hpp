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

// The getters of the local output block (local_y_count, local_y_offset, ...) read state that initialize sets. The
// classes keep a second flag for them that initialize clears at its start and sets at its end; unlike the flag above,
// the transform does not clear it (the getters describe the result of the transform). Throws std::logic_error if the
// flag is not set (no initialize, or a failed one). `name` (the getter and the class) is used in the error message.
inline void require_initialized(const bool initialized, const char *name)
{
    if (!initialized)
    {
        throw std::logic_error(std::string(name) + " needs a successful initialize: call initialize first");
    }
}
}  // namespace hpxfft::util
#endif  // hpxfft_util_transform_guard_H_INCLUDED
