// Distributed FFT correctness test: one case per process invocation (see run_localities.sh).
// Every locality builds its block of a deterministic random input, runs the FFT and compares its output block with a
// naive DFT reference. With --expect_throw the case passes only if initialize rejects the input on every locality.
#include "hpxfft/2D/distributed/agas.hpp"
#include "hpxfft/2D/distributed/loop.hpp"
#include "hpxfft/2D/shared/loop.hpp"
#include "hpxfft/2D/shared/naive.hpp"
#include "hpxfft/2D/shared/opt.hpp"
#include "hpxfft/2D/shared/sync.hpp"
#include "hpxfft/3D/distributed/loop.hpp"
#include "hpxfft/3D/shared/loop.hpp"
#include "hpxfft/3D/shared/naive.hpp"
#include "hpxfft/3D/shared/sync.hpp"
#include "hpxfft/util/adapter_fftw.hpp"
#include "hpxfft/util/fftw_wisdom.hpp"
#include "reference_dft.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <hpx/hpx_init.hpp>
#include <optional>
#include <string>

namespace
{
// value of the unused padding of every row: any path that reads it as data gives wrong results
constexpr double kPad = 1.0e6;

int g_failed = 0;
std::string g_error;  // message of the exception that rejected initialize
// when set, report() appends to it instead of printing (modes that run several cases print one line per locality)
std::string *g_steps = nullptr;

// barriers of one process need distinct names when a mode runs several cases: a barrier name is only unregistered
// asynchronously
std::string barrier_name(const char *name)
{
    static std::size_t counter = 0;
    return std::string(name) + "_" + std::to_string(counter++);
}

struct params
{
    std::string mode, run;
    std::size_t nx, ny, nz;
    bool transposed, expect_throw, uneven_x;
    bool call_twice = false;  // transform a second time without initialize: must throw std::logic_error
    std::string expect_msg;   // with --expect_throw: text the error message must contain
    long size_arg = -1;       // size passed to initialize instead of nz / ny (-1: the true size)
};

void report(const params &p, std::size_t me, bool ok, const std::string &what);

// pass only if initialize threw an error with the expected message (a throw for another reason must not pass)
void report_rejection(const params &p, std::size_t me, bool initialized)
{
    const bool msg_ok = p.expect_msg.empty() || g_error.find(p.expect_msg) != std::string::npos;
    report(p, me, !initialized && msg_ok, initialized ? "accepted an invalid size" : "rejected: " + g_error);
}

void report(const params &p, std::size_t me, bool ok, const std::string &what)
{
    if (!ok)
    {
        g_failed = 1;
    }
    if (g_steps != nullptr)
    {
        *g_steps += std::string(ok ? " [ok " : " [FAIL ") + p.mode + " run=" + p.run + " " + what + "]";
        return;
    }
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

// result of the second transform with --call_twice (ok and nothing to add without it)
struct second_call
{
    bool ok = true;
    std::string what;
};

// with --call_twice: the transform called again without a new initialize must throw std::logic_error (on every
// locality: the check is local, before any communication)
template <typename Transform>
second_call call_again(const params &p, Transform &&transform)
{
    second_call r;
    if (!p.call_twice)
    {
        return r;
    }
    try
    {
        transform();
        r.ok = false;
        r.what = " second_call=accepted";
    }
    catch (const std::logic_error &e)
    {
        r.ok = std::string(e.what()).find("needs a preceding initialize") != std::string::npos;
        r.what = std::string(" second_call=") + (r.ok ? "rejected" : "wrong_error: ") + (r.ok ? "" : e.what());
    }
    return r;
}

// running maximum of the errors; a NaN error counts as infinite (std::max would silently drop it)
inline double worse(const double worst, const double err) { return std::isnan(err) ? HUGE_VAL : std::max(worst, err); }

// compare a block (nx_b x ny_b x nz_b complex numbers stored interleaved) with reference(i, j, k)
template <typename Vec, typename Ref>
bool compare_block(
    const Vec &out, std::size_t n0, std::size_t n1, std::size_t n2, double scale, Ref ref, std::string &what)
{
    if (out.n_x() != n0 || out.n_y() != n1 || out.n_z() != 2 * n2)
    {
        what = "shape " + std::to_string(out.n_x()) + "x" + std::to_string(out.n_y()) + "x" + std::to_string(out.n_z())
               + " expected " + std::to_string(n0) + "x" + std::to_string(n1) + "x" + std::to_string(2 * n2);
        return false;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < n0; ++i)
    {
        for (std::size_t j = 0; j < n1; ++j)
        {
            for (std::size_t k = 0; k < n2; ++k)
            {
                const reference::cplx e = ref(i, j, k);
                worst = worse(worst, std::abs(reference::cplx(out(i, j, 2 * k), out(i, j, 2 * k + 1)) - e));
            }
        }
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
        f.initialize(std::move(v), p.run, "estimate", p.size_arg >= 0 ? std::size_t(p.size_arg) : p.nz);
    }
    catch (const std::exception &e)
    {
        g_error = e.what();
        return false;
    }
    return true;
}

// the test program splits the input evenly unless --uneven_x asks for an uneven first block
void require_divisible(const std::size_t n, const std::size_t parts, const char *what)
{
    if (n % parts != 0)
    {
        throw std::invalid_argument(std::string(what) + " must be divisible by the number of localities");
    }
}

void run_3d(const params &p, bool pencil)
{
    const std::size_t me = hpx::get_locality_id();
    const std::size_t nl = hpx::get_num_localities(hpx::launch::sync);
    const std::size_t P = pencil ? static_cast<std::size_t>(std::sqrt(double(nl)) + 0.5) : 1;
    const std::size_t dcz = p.nz / 2 + 1;
    require_divisible(p.nx, pencil ? P : nl, "nx");
    if (pencil)
    {
        require_divisible(p.ny, P, "ny");
    }
    // input block
    std::size_t nxl, nyl, x0, y0;
    if (pencil)
    {
        nxl = p.nx / P + (p.uneven_x && me == 0 ? 1 : 0);
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
    {
        for (std::size_t j = 0; j < nyl; ++j)
        {
            for (std::size_t k = 0; k < 2 * dcz; ++k)
            {
                v(i, j, k) = k < p.nz ? reference::input_value(x0 + i, y0 + j, k) : kPad;
            }
        }
    }

    // construct only the decomposition under test: a default constructed object that is never initialized must not be
    // destroyed
    std::optional<hpxfft::fft3D::distributed::loop::slab> slab;
    std::optional<hpxfft::fft3D::distributed::loop::pencil> pen;
    bool initialized;
    if (pencil)
    {
        initialized = initialize_3d(pen.emplace(), std::move(v), p);
    }
    else
    {
        initialized = initialize_3d(slab.emplace(), std::move(v), p);
    }

    if (p.expect_throw)
    {
        report_rejection(p, me, initialized);
        return;
    }
    if (!initialized)
    {
        report(p, me, false, "initialize threw");
        return;
    }
    hpx::distributed::barrier(barrier_name("test_init")).wait();
    hpxfft::fft3D::distributed::vector_3d out = pencil ? pen->fft_3d_r2c() : slab->fft_3d_r2c();
    const second_call again = call_again(p, [&] { return pencil ? pen->fft_3d_r2c() : slab->fft_3d_r2c(); });

    reference::rfft3 ref(p.nx, p.ny, p.nz);
    double scale = 0.0;
    for (auto &c : ref.v)
    {
        scale = std::max(scale, std::abs(c));
    }
    std::string what;
    bool ok;
    if (!p.transposed && !pencil)
    {
        const std::size_t xl = p.nx / nl;
        ok = compare_block(
            out, xl, p.ny, dcz, scale, [&](auto i, auto j, auto k) { return ref.at(me * xl + i, j, k); }, what);
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
            out,
            yo[me + 1] - yo[me],
            dcz,
            p.nx,
            scale,
            [&](auto i, auto j, auto k) { return ref.at(k, yo[me] + i, j); },
            what);
    }
    else
    {
        // z frequencies of this locality: blocks of dcz / P, the first dcz % P positions hold one more
        const auto zo = hpxfft::util::block_offsets(dcz, P);
        const std::size_t pz = me % P;
        // y rows of this locality: the block of its position in x direction
        const std::size_t y0_out = pen->local_y_offset();
        ok = compare_block(
            out,
            pen->local_y_count(),
            zo[pz + 1] - zo[pz],
            p.nx,
            scale,
            [&](auto i, auto j, auto k) { return ref.at(k, y0_out + i, zo[pz] + j); },
            what);
    }
    report(p, me, ok && again.ok, what + again.what);
}

// shared memory 3D variants (single locality, natural layout); --run picks the variant
void run_shared(const params &p)
{
    const std::size_t dcz = p.nz / 2 + 1;
    hpxfft::fft3D::distributed::vector_3d v(p.nx, p.ny, 2 * dcz);
    for (std::size_t i = 0; i < p.nx; ++i)
    {
        for (std::size_t j = 0; j < p.ny; ++j)
        {
            for (std::size_t k = 0; k < 2 * dcz; ++k)
            {
                v(i, j, k) = k < p.nz ? reference::input_value(i, j, k) : kPad;
            }
        }
    }
    hpxfft::fft3D::distributed::vector_3d out;
    if (p.run == "naive")
    {
        hpxfft::fft3D::shared::naive f;
        f.initialize(std::move(v), "estimate", p.nz);
        out = f.fft_3d_r2c();
    }
    else if (p.run == "sync")
    {
        hpxfft::fft3D::shared::sync f;
        f.initialize(std::move(v), "estimate", p.nz);
        out = f.fft_3d_r2c();
    }
    else if (p.run == "par" || p.run == "seq")
    {
        hpxfft::fft3D::shared::loop f;
        f.initialize(std::move(v), "estimate", p.nz);
        out = p.run == "seq" ? f.fft_3d_r2c_seq() : f.fft_3d_r2c_par();
    }
    else
    {
        throw std::invalid_argument("unknown shared variant " + p.run);
    }
    reference::rfft3 ref(p.nx, p.ny, p.nz);
    double scale = 0.0;
    for (auto &c : ref.v)
    {
        scale = std::max(scale, std::abs(c));
    }
    std::string what;
    const bool ok =
        compare_block(out, p.nx, p.ny, dcz, scale, [&](auto i, auto j, auto k) { return ref.at(i, j, k); }, what);
    report(p, 0, ok, what);
}

// shared memory 2D variants (single locality); --run picks the variant
void run_shared_2d(const params &p)
{
    const std::size_t dcy = p.ny / 2 + 1;
    hpxfft::fft2D::distributed::vector_2d v(p.nx, 2 * dcy);
    for (std::size_t i = 0; i < p.nx; ++i)
    {
        for (std::size_t j = 0; j < 2 * dcy; ++j)
        {
            v(i, j) = j < p.ny ? reference::input_value(i, j, 0) : kPad;
        }
    }
    hpxfft::fft2D::distributed::vector_2d out;
    if (p.run == "naive")
    {
        hpxfft::fft2D::shared::naive f;
        f.initialize(std::move(v), "estimate", p.ny);
        out = f.fft_2d_r2c();
    }
    else if (p.run == "sync")
    {
        hpxfft::fft2D::shared::sync f;
        f.initialize(std::move(v), "estimate", p.ny);
        out = f.fft_2d_r2c();
    }
    else if (p.run == "opt")
    {
        hpxfft::fft2D::shared::opt f;
        f.initialize(std::move(v), "estimate", p.ny);
        out = f.fft_2d_r2c();
    }
    else if (p.run == "par" || p.run == "seq")
    {
        hpxfft::fft2D::shared::loop f;
        f.initialize(std::move(v), "estimate", p.ny);
        out = p.run == "seq" ? f.fft_2d_r2c_seq() : f.fft_2d_r2c_par();
    }
    else
    {
        throw std::invalid_argument("unknown shared variant " + p.run);
    }
    reference::rfft2 ref(p.nx, p.ny);
    double scale = 0.0;
    for (auto &c : ref.v)
    {
        scale = std::max(scale, std::abs(c));
    }
    double worst = 0.0;
    std::string what;
    bool ok = out.n_row() == p.nx && out.n_col() == 2 * dcy;
    if (ok)
    {
        for (std::size_t i = 0; i < p.nx; ++i)
        {
            for (std::size_t j = 0; j < dcy; ++j)
            {
                worst = worse(worst, std::abs(reference::cplx(out(i, 2 * j), out(i, 2 * j + 1)) - ref.at(i, j)));
            }
        }
        what = "max_rel_err=" + std::to_string(worst / scale);
        ok = worst / scale < 1e-10;
    }
    else
    {
        what = "shape mismatch";
    }
    report(p, 0, ok, what);
}

// compare the natural-order result (rows x0 .. x0 + rows, all dcy frequencies) of a 2D agas variant with the reference
void check_agas_2d(const params &p,
                   std::size_t me,
                   std::size_t x0,
                   std::size_t nxl,
                   hpxfft::fft2D::distributed::vector_2d &out,
                   const second_call &again = {})
{
    const std::size_t dcy = p.ny / 2 + 1;
    reference::rfft2 ref(p.nx, p.ny);
    double scale = 0.0;
    for (auto &c : ref.v)
    {
        scale = std::max(scale, std::abs(c));
    }
    if (out.n_row() != nxl || out.n_col() != 2 * dcy)
    {
        report(p, me, false, "shape " + std::to_string(out.n_row()) + "x" + std::to_string(out.n_col()));
        return;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < out.n_row(); ++i)
    {
        for (std::size_t j = 0; j < dcy; ++j)
        {
            worst = worse(worst, std::abs(reference::cplx(out(i, 2 * j), out(i, 2 * j + 1)) - ref.at(x0 + i, j)));
        }
    }
    report(p, me, worst / scale < 1e-10 && again.ok, "max_rel_err=" + std::to_string(worst / scale) + again.what);
}

// 2D distributed agas component: ny must be even (the component derives ny from the row length). The shared agas
// component cannot be used in the same program: both define a global fft_2d_r2c_action (see test_shared_agas).
void run_agas_2d(const params &p)
{
    const std::size_t me = hpx::get_locality_id();
    const std::size_t nl = hpx::get_num_localities(hpx::launch::sync);
    const std::size_t dcy = p.ny / 2 + 1;
    require_divisible(p.nx, nl, "nx");
    const std::size_t nxl = p.nx / nl;
    hpxfft::fft2D::distributed::vector_2d v(nxl, 2 * dcy);
    for (std::size_t i = 0; i < nxl; ++i)
    {
        for (std::size_t j = 0; j < 2 * dcy; ++j)
        {
            v(i, j) = j < p.ny ? reference::input_value(me * nxl + i, j, 0) : kPad;
        }
    }
    hpxfft::fft2D::distributed::agas f;
    bool initialized = true;
    try
    {
        f.initialize(std::move(v), p.run, "estimate").get();
    }
    catch (const std::exception &e)
    {
        g_error = e.what();
        initialized = false;
    }
    if (p.expect_throw)
    {
        report_rejection(p, me, initialized);
        return;
    }
    hpx::distributed::barrier(barrier_name("test_init")).wait();
    hpxfft::fft2D::distributed::vector_2d out = f.fft_2d_r2c().get();
    const second_call again = call_again(p, [&] { return f.fft_2d_r2c().get(); });
    check_agas_2d(p, me, me * nxl, nxl, out, again);
}

// helpers of the shared classes: write_plans_to_file (also the error path) and get_measurement
template <typename Fft>
bool plans_file_ok(Fft &f, const std::string &path)
{
    std::remove(path.c_str());
    f.write_plans_to_file(path);
    std::FILE *file = std::fopen(path.c_str(), "r");
    const bool written = file != nullptr && std::fgetc(file) != EOF;
    if (file)
    {
        std::fclose(file);
    }
    std::remove(path.c_str());
    bool threw = false;
    try
    {
        f.write_plans_to_file("/nonexistent_dir_for_hpxfft_test/plans.txt");
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    return written && threw;
}

void run_helpers(const params &p)
{
    const std::size_t dcz = p.nz / 2 + 1;
    const std::size_t dcy = p.ny / 2 + 1;
    auto make3 = [&]
    {
        hpxfft::fft3D::distributed::vector_3d v(p.nx, p.ny, 2 * dcz);
        for (std::size_t i = 0; i < p.nx; ++i)
        {
            for (std::size_t j = 0; j < p.ny; ++j)
            {
                for (std::size_t k = 0; k < p.nz; ++k)
                {
                    v(i, j, k) = reference::input_value(i, j, k);
                }
            }
        }
        return v;
    };
    auto make2 = [&]
    {
        hpxfft::fft2D::distributed::vector_2d v(p.nx, 2 * dcy);
        for (std::size_t i = 0; i < p.nx; ++i)
        {
            for (std::size_t j = 0; j < p.ny; ++j)
            {
                v(i, j) = reference::input_value(i, j, 0);
            }
        }
        return v;
    };
    const std::string path = "hpxfft_plans_test.txt";
    bool ok = true;
    {
        hpxfft::fft3D::shared::loop f;
        f.initialize(make3(), "estimate", p.nz);
        f.fft_3d_r2c_par();
        ok = plans_file_ok(f, path) && ok;
    }
    {
        hpxfft::fft3D::shared::naive f;
        f.initialize(make3(), "estimate", p.nz);
        f.fft_3d_r2c();
        ok = plans_file_ok(f, path) && f.get_measurement("total") > 0.0 && ok;
    }
    {
        hpxfft::fft3D::shared::sync f;
        f.initialize(make3(), "estimate", p.nz);
        f.fft_3d_r2c();
        ok = plans_file_ok(f, path) && ok;
    }
    {
        hpxfft::fft2D::shared::loop f;
        f.initialize(make2(), "estimate", p.ny);
        f.fft_2d_r2c_par();
        ok = plans_file_ok(f, path) && f.get_measurement("total") > 0.0 && ok;
    }
    {
        hpxfft::fft2D::shared::naive f;
        f.initialize(make2(), "estimate", p.ny);
        f.fft_2d_r2c();
        ok = f.get_measurement("total") > 0.0 && ok;
    }
    {
        hpxfft::fft2D::shared::sync f;
        f.initialize(make2(), "estimate", p.ny);
        f.fft_2d_r2c();
        ok = f.get_measurement("total") > 0.0 && ok;
    }
    {
        hpxfft::fft2D::shared::opt f;
        f.initialize(make2(), "estimate", p.ny);
        f.fft_2d_r2c();
        ok = f.get_measurement("total") > 0.0 && ok;
    }
    report(p, 0, ok, "helpers");
}

// build the input block of this locality and initialize the 2D loop; false if initialize threw (message in g_error)
bool initialize_2d(hpxfft::fft2D::distributed::loop &f, const params &p)
{
    const std::size_t me = hpx::get_locality_id();
    const std::size_t nl = hpx::get_num_localities(hpx::launch::sync);
    const std::size_t dcy = p.ny / 2 + 1;
    require_divisible(p.nx, nl, "nx");
    const std::size_t nxl = p.nx / nl + (p.uneven_x && me == 0 ? 1 : 0);
    const std::size_t x0 = me * (p.nx / nl);
    hpxfft::fft2D::distributed::vector_2d v(nxl, 2 * dcy);
    for (std::size_t i = 0; i < nxl; ++i)
    {
        for (std::size_t j = 0; j < 2 * dcy; ++j)
        {
            v(i, j) = j < p.ny ? reference::input_value(x0 + i, j, 0) : kPad;
        }
    }

    f.set_transposed_output(p.transposed);
    f.set_hierarchical_parameters(2, 0);
    try
    {
        f.initialize(std::move(v), p.run, "estimate", p.size_arg >= 0 ? std::size_t(p.size_arg) : p.ny);
    }
    catch (const std::exception &e)
    {
        g_error = e.what();
        return false;
    }
    return true;
}

// compare the result of the 2D loop with the reference
void check_2d(const params &p, hpxfft::fft2D::distributed::vector_2d &out, const second_call &again = {})
{
    const std::size_t me = hpx::get_locality_id();
    const std::size_t nl = hpx::get_num_localities(hpx::launch::sync);
    const std::size_t dcy = p.ny / 2 + 1;
    reference::rfft2 ref(p.nx, p.ny);
    double scale = 0.0;
    for (auto &c : ref.v)
    {
        scale = std::max(scale, std::abs(c));
    }
    // natural: rows of this locality, all dcy frequencies; transposed: (y_local, x) with y distributed
    // transposed: the y frequencies are split into blocks of dcy / L, the first dcy % L localities hold one more
    const auto yo = hpxfft::util::block_offsets(dcy, nl);
    const std::size_t rows = p.transposed ? yo[me + 1] - yo[me] : p.nx / nl;
    const std::size_t cols = p.transposed ? p.nx : dcy;
    std::string what;
    bool ok = true;
    if (out.n_row() != rows || out.n_col() != 2 * cols)
    {
        ok = false;
        what = "shape " + std::to_string(out.n_row()) + "x" + std::to_string(out.n_col()) + " expected "
               + std::to_string(rows) + "x" + std::to_string(2 * cols);
    }
    else
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                const reference::cplx e = p.transposed ? ref.at(j, yo[me] + i) : ref.at(me * rows + i, j);
                worst = worse(worst, std::abs(reference::cplx(out(i, 2 * j), out(i, 2 * j + 1)) - e));
            }
        }
        what = "max_rel_err=" + std::to_string(worst / scale);
        ok = worst / scale < 1e-10;
    }
    report(p, me, ok && again.ok, what + again.what);
}

void run_2d(const params &p)
{
    const std::size_t me = hpx::get_locality_id();
    hpxfft::fft2D::distributed::loop f;
    const bool initialized = initialize_2d(f, p);
    if (p.expect_throw)
    {
        report_rejection(p, me, initialized);
        return;
    }
    if (!initialized)
    {
        report(p, me, false, "initialize threw");
        return;
    }
    hpx::distributed::barrier(barrier_name("test_init")).wait();
    auto out = f.fft_2d_r2c();
    const second_call again = call_again(p, [&] { return f.fft_2d_r2c(); });
    check_2d(p, out, again);
}

// several objects in one process: their communicators must not collide. A slab is run and destroyed, then a second
// slab with another run flag, then a slab while a 2D loop is alive (initialized before and run after it). Prints one
// line per locality.
void run_reuse(const params &p)
{
    const std::size_t me = hpx::get_locality_id();
    params slab_p = p;
    slab_p.mode = "slab";
    params loop_p = p;
    loop_p.mode = "2d";
    loop_p.nz = 0;
    std::string steps;
    const int failed_before = g_failed;
    g_failed = 0;
    g_steps = &steps;
    try
    {
        slab_p.run = "scatter";
        run_3d(slab_p, false);
        slab_p.run = "all_to_all";
        run_3d(slab_p, false);

        hpxfft::fft2D::distributed::loop f;
        loop_p.run = "scatter";
        if (!initialize_2d(f, loop_p))
        {
            report(loop_p, me, false, "initialize threw: " + g_error);
        }
        else
        {
            slab_p.run = "scatter";
            run_3d(slab_p, false);
            hpx::distributed::barrier(barrier_name("test_reuse")).wait();
            auto out = f.fft_2d_r2c();
            check_2d(loop_p, out);
        }
    }
    catch (const std::exception &e)
    {
        report(p, me, false, std::string("exception: ") + e.what());
    }
    g_steps = nullptr;
    const bool ok = g_failed == 0;
    g_failed = failed_before;
    report(p, me, ok, steps);
}

// wisdom: every locality plans a c2c FFT of size nx with measure and exports to the same file at the same time; the
// atomic rename must leave a complete file that every locality can import and plan from with FFTW_WISDOM_ONLY
void run_wisdom(const params &p)
{
    const std::size_t me = hpx::get_locality_id();
    const auto dir = std::filesystem::temp_directory_path() / "hpxfft_dist_wisdom_test";
    const std::string file = (dir / "wisdom.txt").string();
    if (me == 0)
    {
        std::filesystem::remove_all(dir);
    }
    hpx::distributed::barrier("wisdom_clean").wait();
    const int n = static_cast<int>(p.nx);
    fftw_complex *buf = fftw_alloc_complex(p.nx);
    {
        hpxfft::util::fftw_adapter::c2c_1d adapter;
        adapter.plan(n, "measure", buf, buf, hpxfft::util::fftw_adapter::direction::forward);
        hpxfft::util::export_wisdom(file);
    }
    hpx::distributed::barrier("wisdom_export").wait();
    fftw_forget_wisdom();
    const bool imported = hpxfft::util::import_wisdom(file);
    fftw_plan plan = fftw_plan_dft_1d(n, buf, buf, FFTW_FORWARD, FFTW_MEASURE | FFTW_WISDOM_ONLY);
    const bool ok = imported && plan != nullptr;
    fftw_destroy_plan(plan);
    fftw_free(buf);
    hpx::distributed::barrier("wisdom_import").wait();
    std::size_t leftovers = 0;
    for ([[maybe_unused]] const auto &entry : std::filesystem::directory_iterator(dir))
    {
        ++leftovers;
    }
    hpx::distributed::barrier("wisdom_count").wait();
    if (me == 0)
    {
        std::filesystem::remove_all(dir);
    }
    report(p,
           me,
           ok && leftovers == 1,
           "imported=" + std::to_string(imported) + " wisdom_only_plan=" + std::to_string(plan != nullptr)
               + " files=" + std::to_string(leftovers));
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
    p.call_twice = vm["call_twice"].as<bool>();
    p.size_arg = vm["size_arg"].as<long>();
    p.expect_msg = vm["expect_msg"].as<std::string>();
    try
    {
        if (p.mode == "2d")
        {
            run_2d(p);
        }
        else if (p.mode == "shared")
        {
            run_shared(p);
        }
        else if (p.mode == "helpers")
        {
            run_helpers(p);
        }
        else if (p.mode == "agas2d")
        {
            run_agas_2d(p);
        }
        else if (p.mode == "shared2d")
        {
            run_shared_2d(p);
        }
        else if (p.mode == "wisdom")
        {
            run_wisdom(p);
        }
        else if (p.mode == "reuse")
        {
            run_reuse(p);
        }
        else if (p.mode == "slab" || p.mode == "pencil")
        {
            run_3d(p, p.mode == "pencil");
        }
        else
        {
            throw std::invalid_argument("unknown mode " + p.mode);
        }
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
        "expect_throw", value<bool>()->default_value(false))("call_twice", value<bool>()->default_value(false))(
        "uneven_x", value<bool>()->default_value(false))("size_arg", value<long>()->default_value(-1))(
        "expect_msg", value<std::string>()->default_value(""));
    hpx::init_params ip;
    ip.desc_cmdline = d;
    ip.cfg = { "hpx.run_hpx_main!=1" };
    int rc = hpx::init(hpx_main, argc, argv, ip);
    return rc != 0 ? rc : g_failed;
}
