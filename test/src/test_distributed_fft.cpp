// Distributed FFT correctness test: one case per process invocation (see run_localities.sh).
// Every locality builds its block of a deterministic random input, runs the FFT and compares its output block with a
// naive DFT reference. With --expect_throw the case passes only if initialize rejects the input on every locality.
#include "hpxfft/2D/distributed/loop.hpp"
#include "hpxfft/3D/distributed/loop.hpp"
#include "hpxfft/3D/shared/loop.hpp"
#include "reference_dft.hpp"
#include <cmath>
#include <cstdio>
#include <hpx/hpx_init.hpp>
#include <optional>
#include <string>

namespace
{
int g_failed = 0;

struct params
{
    std::string mode, run;
    std::size_t nx, ny, nz;
    bool transposed, expect_throw, uneven_x;
};

void report(const params &p, std::size_t me, bool ok, const std::string &what)
{
    if (!ok) g_failed = 1;
    std::printf("%s locality %zu %s %zux%zux%zu run=%s transposed=%d %s\n",
                ok ? "PASS" : "FAIL",
                me,
                p.mode.c_str(),
                p.nx,
                p.ny,
                p.nz,
                p.run.c_str(),
                int(p.transposed),
                what.c_str());
    std::fflush(stdout);
}

// compare a block (nx_b x ny_b x nz_b complex numbers stored interleaved) with reference(i, j, k)
template <typename Vec, typename Ref>
bool compare_block(const Vec &out, std::size_t n0, std::size_t n1, std::size_t n2, double scale, Ref ref, std::string &what)
{
    if (out.n_x() != n0 || out.n_y() != n1 || out.n_z() != 2 * n2)
    {
        what = "shape " + std::to_string(out.n_x()) + "x" + std::to_string(out.n_y()) + "x" + std::to_string(out.n_z()) +
            " expected " + std::to_string(n0) + "x" + std::to_string(n1) + "x" + std::to_string(2 * n2);
        return false;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < n0; ++i)
        for (std::size_t j = 0; j < n1; ++j)
            for (std::size_t k = 0; k < n2; ++k)
            {
                const reference::cplx e = ref(i, j, k);
                worst = std::max(worst, std::abs(reference::cplx(out(i, j, 2 * k), out(i, j, 2 * k + 1)) - e));
            }
    what = "max_rel_err=" + std::to_string(worst / scale);
    return worst / scale < 1e-10;
}

template <typename Fft>
bool initialize_3d(Fft &f, hpxfft::fft3D::distributed::vector_3d &&v, const params &p)
{
    f.set_transposed_output(p.transposed);
    f.set_hierarchical_parameters(2, 0);
    try
    {
        f.initialize(std::move(v), p.run, "estimate", p.nz);
    }
    catch (const std::exception &e)
    {
        return false;
    }
    return true;
}

void run_3d(const params &p, bool pencil)
{
    const std::size_t me = hpx::get_locality_id();
    const std::size_t nl = hpx::get_num_localities(hpx::launch::sync);
    const std::size_t P = pencil ? static_cast<std::size_t>(std::sqrt(double(nl)) + 0.5) : 1;
    const std::size_t dcz = p.nz / 2 + 1;
    // input block
    std::size_t nxl, nyl, x0, y0;
    if (pencil)
    {
        nxl = p.nx / P;
        nyl = p.ny / P;
        x0 = (me / P) * nxl;
        y0 = (me % P) * nyl;
    }
    else
    {
        nxl = p.nx / nl + (p.uneven_x && me == 0 ? 1 : 0);
        nyl = p.ny;
        x0 = me * (p.nx / nl);
        y0 = 0;
    }
    hpxfft::fft3D::distributed::vector_3d v(nxl, nyl, 2 * dcz);
    for (std::size_t i = 0; i < nxl; ++i)
        for (std::size_t j = 0; j < nyl; ++j)
            for (std::size_t k = 0; k < 2 * dcz; ++k)
                v(i, j, k) = k < p.nz ? reference::input_value(x0 + i, y0 + j, k) : 0.0;

    // construct only the decomposition under test: a default constructed object that is never initialized must not be
    // destroyed
    std::optional<hpxfft::fft3D::distributed::loop::slab> slab;
    std::optional<hpxfft::fft3D::distributed::loop::pencil> pen;
    bool initialized;
    if (pencil)
        initialized = initialize_3d(pen.emplace(), std::move(v), p);
    else
        initialized = initialize_3d(slab.emplace(), std::move(v), p);

    if (p.expect_throw)
    {
        report(p, me, !initialized, initialized ? "accepted an invalid size" : "rejected as expected");
        return;
    }
    if (!initialized)
    {
        report(p, me, false, "initialize threw");
        return;
    }
    hpx::distributed::barrier("test_init").wait();
    hpxfft::fft3D::distributed::vector_3d out = pencil ? pen->fft_3d_r2c() : slab->fft_3d_r2c();

    reference::rfft3 ref(p.nx, p.ny, p.nz);
    double scale = 0.0;
    for (auto &c : ref.v) scale = std::max(scale, std::abs(c));
    std::string what;
    bool ok;
    if (!p.transposed && !pencil)
    {
        const std::size_t xl = p.nx / nl;
        ok = compare_block(out, xl, p.ny, dcz, scale, [&](auto i, auto j, auto k) { return ref.at(me * xl + i, j, k); }, what);
    }
    else if (!p.transposed)
    {
        ok = compare_block(
            out, nxl, nyl, dcz, scale, [&](auto i, auto j, auto k) { return ref.at(x0 + i, y0 + j, k); }, what);
    }
    else if (!pencil)
    {
        // y rows of this locality: blocks of ny / L, the first ny % L localities hold one more
        const auto yo = hpxfft::util::block_offsets(p.ny, nl);
        ok = compare_block(
            out, yo[me + 1] - yo[me], dcz, p.nx, scale, [&](auto i, auto j, auto k) { return ref.at(k, yo[me] + i, j); }, what);
    }
    else
    {
        const std::size_t zl = dcz / P;
        ok = compare_block(
            out, nyl, zl, p.nx, scale, [&](auto i, auto j, auto k) { return ref.at(k, (me / P) * nyl + i, (me % P) * zl + j); }, what);
    }
    report(p, me, ok, what);
}


// shared memory 3D loop (single locality), natural layout
void run_shared(const params &p)
{
    const std::size_t dcz = p.nz / 2 + 1;
    hpxfft::fft3D::distributed::vector_3d v(p.nx, p.ny, 2 * dcz);
    for (std::size_t i = 0; i < p.nx; ++i)
        for (std::size_t j = 0; j < p.ny; ++j)
            for (std::size_t k = 0; k < 2 * dcz; ++k) v(i, j, k) = k < p.nz ? reference::input_value(i, j, k) : 0.0;
    hpxfft::fft3D::shared::loop f;
    f.initialize(std::move(v), "estimate", p.nz);
    auto out = f.fft_3d_r2c_par();
    reference::rfft3 ref(p.nx, p.ny, p.nz);
    double scale = 0.0;
    for (auto &c : ref.v) scale = std::max(scale, std::abs(c));
    std::string what;
    const bool ok = compare_block(out, p.nx, p.ny, dcz, scale, [&](auto i, auto j, auto k) { return ref.at(i, j, k); }, what);
    report(p, 0, ok, what);
}

void run_2d(const params &p)
{
    const std::size_t me = hpx::get_locality_id();
    const std::size_t nl = hpx::get_num_localities(hpx::launch::sync);
    const std::size_t dcy = p.ny / 2 + 1;
    const std::size_t nxl = p.nx / nl + (p.uneven_x && me == 0 ? 1 : 0);
    const std::size_t x0 = me * (p.nx / nl);
    hpxfft::fft2D::distributed::vector_2d v(nxl, 2 * dcy);
    for (std::size_t i = 0; i < nxl; ++i)
        for (std::size_t j = 0; j < 2 * dcy; ++j) v(i, j) = j < p.ny ? reference::input_value(x0 + i, j, 0) : 0.0;

    hpxfft::fft2D::distributed::loop f;
    f.set_transposed_output(p.transposed);
    f.set_hierarchical_parameters(2, 0);
    bool initialized = true;
    try
    {
        f.initialize(std::move(v), p.run, "estimate", p.ny);
    }
    catch (const std::exception &)
    {
        initialized = false;
    }
    if (p.expect_throw)
    {
        report(p, me, !initialized, initialized ? "accepted an invalid size" : "rejected as expected");
        return;
    }
    if (!initialized)
    {
        report(p, me, false, "initialize threw");
        return;
    }
    hpx::distributed::barrier("test_init").wait();
    auto out = f.fft_2d_r2c();

    reference::rfft2 ref(p.nx, p.ny);
    double scale = 0.0;
    for (auto &c : ref.v) scale = std::max(scale, std::abs(c));
    // natural: rows of this locality, all dcy frequencies; transposed: (y_local, x) with y distributed
    const std::size_t rows = p.transposed ? dcy / nl : p.nx / nl;
    const std::size_t cols = p.transposed ? p.nx : dcy;
    std::string what;
    bool ok = true;
    if (out.n_row() != rows || out.n_col() != 2 * cols)
    {
        ok = false;
        what = "shape " + std::to_string(out.n_row()) + "x" + std::to_string(out.n_col()) + " expected " +
            std::to_string(rows) + "x" + std::to_string(2 * cols);
    }
    else
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
            {
                const reference::cplx e = p.transposed ? ref.at(j, me * rows + i) : ref.at(me * rows + i, j);
                worst = std::max(worst, std::abs(reference::cplx(out(i, 2 * j), out(i, 2 * j + 1)) - e));
            }
        what = "max_rel_err=" + std::to_string(worst / scale);
        ok = worst / scale < 1e-10;
    }
    report(p, me, ok, what);
}

int hpx_main(hpx::program_options::variables_map &vm)
{
    params p;
    p.mode = vm["mode"].as<std::string>();
    p.run = vm["run"].as<std::string>();
    p.nx = vm["nx"].as<std::size_t>();
    p.ny = vm["ny"].as<std::size_t>();
    p.nz = vm["nz"].as<std::size_t>();
    p.transposed = vm["transposed"].as<bool>();
    p.expect_throw = vm["expect_throw"].as<bool>();
    p.uneven_x = vm["uneven_x"].as<bool>();
    try
    {
        if (p.mode == "2d")
            run_2d(p);
        else if (p.mode == "shared")
            run_shared(p);
        else
            run_3d(p, p.mode == "pencil");
    }
    catch (const std::exception &e)
    {
        report(p, hpx::get_locality_id(), false, std::string("exception: ") + e.what());
    }
    return hpx::finalize();
}
}  // namespace

int main(int argc, char *argv[])
{
    using namespace hpx::program_options;
    options_description d;
    d.add_options()("mode", value<std::string>()->default_value("slab"), "slab, pencil or 2d")(
        "run", value<std::string>()->default_value("scatter"), "communication run flag")(
        "nx", value<std::size_t>()->default_value(8))("ny", value<std::size_t>()->default_value(8))(
        "nz", value<std::size_t>()->default_value(6))("transposed", value<bool>()->default_value(false))(
        "expect_throw", value<bool>()->default_value(false))("uneven_x", value<bool>()->default_value(false));
    hpx::init_params ip;
    ip.desc_cmdline = d;
    ip.cfg = {"hpx.run_hpx_main!=1"};
    int rc = hpx::init(hpx_main, argc, argv, ip);
    return rc != 0 ? rc : g_failed;
}
