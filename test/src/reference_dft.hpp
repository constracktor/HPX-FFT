#pragma once
// Naive separable DFT reference for the distributed correctness tests (small sizes only).
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference
{
using cplx = std::complex<double>;

// deterministic pseudo random input value of the global index (X, Y, Z), in [-1, 1)
inline double input_value(std::uint64_t X, std::uint64_t Y, std::uint64_t Z)
{
    std::uint64_t h = X * 73'856'093ull ^ Y * 19'349'663ull ^ Z * 83'492'791ull;
    h ^= h >> 13;
    h *= 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    return double(h & 0xFFFFFF) / 8388608.0 - 1.0;
}

// forward DFT of `count` values with stride `stride` starting at data + offset, in place
inline void dft_inplace(std::vector<cplx> &data, std::size_t offset, std::size_t stride, std::size_t count)
{
    std::vector<cplx> in(count), out(count);
    for (std::size_t k = 0; k < count; ++k)
    {
        in[k] = data[offset + k * stride];
    }
    for (std::size_t k = 0; k < count; ++k)
    {
        cplx s = 0.0;
        for (std::size_t n = 0; n < count; ++n)
        {
            const double a = -2.0 * M_PI * double((k * n) % count) / double(count);
            s += in[n] * cplx(std::cos(a), std::sin(a));
        }
        out[k] = s;
    }
    for (std::size_t k = 0; k < count; ++k)
    {
        data[offset + k * stride] = out[k];
    }
}

// 3D real-to-complex transform of input_value on nx * ny * nz points; result [x][y][z] with z in 0 .. nz/2
struct rfft3
{
    std::size_t nx, ny, nz, dcz;
    std::vector<cplx> v;

    cplx &at(std::size_t x, std::size_t y, std::size_t z) { return v[(x * ny + y) * dcz + z]; }

    rfft3(std::size_t nx_, std::size_t ny_, std::size_t nz_) :
        nx(nx_),
        ny(ny_),
        nz(nz_),
        dcz(nz_ / 2 + 1)
    {
        // full complex transform along z, keep the first dcz outputs
        std::vector<cplx> full(nx * ny * nz);
        for (std::size_t x = 0; x < nx; ++x)
        {
            for (std::size_t y = 0; y < ny; ++y)
            {
                for (std::size_t z = 0; z < nz; ++z)
                {
                    full[(x * ny + y) * nz + z] = input_value(x, y, z);
                }
            }
        }
        for (std::size_t x = 0; x < nx; ++x)
        {
            for (std::size_t y = 0; y < ny; ++y)
            {
                dft_inplace(full, (x * ny + y) * nz, 1, nz);
            }
        }
        v.resize(nx * ny * dcz);
        for (std::size_t x = 0; x < nx; ++x)
        {
            for (std::size_t y = 0; y < ny; ++y)
            {
                for (std::size_t z = 0; z < dcz; ++z)
                {
                    v[(x * ny + y) * dcz + z] = full[(x * ny + y) * nz + z];
                }
            }
        }
        for (std::size_t x = 0; x < nx; ++x)
        {
            for (std::size_t z = 0; z < dcz; ++z)
            {
                dft_inplace(v, x * ny * dcz + z, dcz, ny);
            }
        }
        for (std::size_t y = 0; y < ny; ++y)
        {
            for (std::size_t z = 0; z < dcz; ++z)
            {
                dft_inplace(v, y * dcz + z, ny * dcz, nx);
            }
        }
    }
};

// 2D real-to-complex transform of input_value(x, y, 0) on nx * ny points; result [x][y] with y in 0 .. ny/2
struct rfft2
{
    std::size_t nx, ny, dcy;
    std::vector<cplx> v;

    cplx &at(std::size_t x, std::size_t y) { return v[x * dcy + y]; }

    rfft2(std::size_t nx_, std::size_t ny_) :
        nx(nx_),
        ny(ny_),
        dcy(ny_ / 2 + 1)
    {
        std::vector<cplx> full(nx * ny);
        for (std::size_t x = 0; x < nx; ++x)
        {
            for (std::size_t y = 0; y < ny; ++y)
            {
                full[x * ny + y] = input_value(x, y, 0);
            }
        }
        for (std::size_t x = 0; x < nx; ++x)
        {
            dft_inplace(full, x * ny, 1, ny);
        }
        v.resize(nx * dcy);
        for (std::size_t x = 0; x < nx; ++x)
        {
            for (std::size_t y = 0; y < dcy; ++y)
            {
                v[x * dcy + y] = full[x * ny + y];
            }
        }
        for (std::size_t y = 0; y < dcy; ++y)
        {
            dft_inplace(v, y, dcy, nx);
        }
    }
};
}  // namespace reference
