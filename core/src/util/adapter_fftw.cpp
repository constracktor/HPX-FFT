#include "../../include/hpxfft/util/adapter_fftw.hpp"

#include <cstddef>
#include <utility>

namespace
{
// Destroy a plan owned by an adapter (if any) and reset the pointer. Holds the planner mutex.
void destroy_plan(fftw_plan &plan)
{
    if (plan != nullptr)
    {
        std::lock_guard<std::mutex> lock(hpxfft::util::fftw_adapter::planner_mutex());
        fftw_destroy_plan(plan);
        plan = nullptr;
    }
}
}  // namespace

// FFTW adapter implementation
std::mutex &hpxfft::util::fftw_adapter::planner_mutex()
{
    static std::mutex mutex;
    return mutex;
}

namespace
{
// number of live cleanup_guard objects; guarded by planner_mutex()
std::size_t live_guards = 0;
}  // namespace

hpxfft::util::fftw_adapter::cleanup_guard::cleanup_guard()
{
    std::lock_guard<std::mutex> lock(planner_mutex());
    ++live_guards;
}

hpxfft::util::fftw_adapter::cleanup_guard::~cleanup_guard()
{
    std::lock_guard<std::mutex> lock(planner_mutex());
    // only the last live object cleans up: fftw_cleanup() invalidates the plans of every other object
    if (--live_guards == 0)
    {
        fftw_cleanup();
    }
}

void hpxfft::util::fftw_adapter::r2c_1d::plan(int dim_r, std::string plan_flag, double *in, fftw_complex *out)
{
    // parse the flag first: an invalid flag throws and leaves an owned plan untouched
    const auto flags = static_cast<unsigned>(string_to_fftw_plan_flag(plan_flag));
    std::lock_guard<std::mutex> lock(planner_mutex());
    // re-planning replaces (and frees) a plan this adapter already owns
    if (plan_r2c_1d_ != nullptr)
    {
        fftw_destroy_plan(plan_r2c_1d_);
    }
    // create FFTW plan
    plan_r2c_1d_ = fftw_plan_dft_r2c_1d(dim_r, in, out, flags);
}

hpxfft::util::fftw_adapter::r2c_1d::r2c_1d(r2c_1d &&other) noexcept :
    plan_r2c_1d_(std::exchange(other.plan_r2c_1d_, nullptr))
{ }

hpxfft::util::fftw_adapter::r2c_1d &hpxfft::util::fftw_adapter::r2c_1d::operator=(r2c_1d &&other) noexcept
{
    if (this != &other)
    {
        destroy_plan(plan_r2c_1d_);
        plan_r2c_1d_ = std::exchange(other.plan_r2c_1d_, nullptr);
    }
    return *this;
}

hpxfft::util::fftw_adapter::r2c_1d::~r2c_1d() { destroy_plan(plan_r2c_1d_); }

void hpxfft::util::fftw_adapter::r2c_1d::execute(double *in, fftw_complex *out)
{
    fftw_execute_dft_r2c(plan_r2c_1d_, in, out);
}

void hpxfft::util::fftw_adapter::r2c_1d::flops(double *add, double *mul, double *fma)
{
    fftw_flops(plan_r2c_1d_, add, mul, fma);
}

void hpxfft::util::fftw_adapter::r2c_1d::print_plan(FILE *stream) { fftw_fprint_plan(plan_r2c_1d_, stream); }

void hpxfft::util::fftw_adapter::c2c_1d::plan(
    int dim_c, std::string plan_flag, fftw_complex *in, fftw_complex *out, fftw_adapter::direction direction)
{
    // parse the flag first: an invalid flag throws and leaves an owned plan untouched
    const auto flags = static_cast<unsigned>(string_to_fftw_plan_flag(plan_flag));
    std::lock_guard<std::mutex> lock(planner_mutex());
    // re-planning replaces (and frees) a plan this adapter already owns
    if (plan_c2c_1d_ != nullptr)
    {
        fftw_destroy_plan(plan_c2c_1d_);
    }
    // create FFTW plan
    plan_c2c_1d_ = fftw_plan_dft_1d(dim_c, in, out, static_cast<int>(direction), flags);
}

hpxfft::util::fftw_adapter::c2c_1d::c2c_1d(c2c_1d &&other) noexcept :
    plan_c2c_1d_(std::exchange(other.plan_c2c_1d_, nullptr))
{ }

hpxfft::util::fftw_adapter::c2c_1d &hpxfft::util::fftw_adapter::c2c_1d::operator=(c2c_1d &&other) noexcept
{
    if (this != &other)
    {
        destroy_plan(plan_c2c_1d_);
        plan_c2c_1d_ = std::exchange(other.plan_c2c_1d_, nullptr);
    }
    return *this;
}

hpxfft::util::fftw_adapter::c2c_1d::~c2c_1d() { destroy_plan(plan_c2c_1d_); }

void hpxfft::util::fftw_adapter::c2c_1d::execute(fftw_complex *in, fftw_complex *out)
{
    fftw_execute_dft(plan_c2c_1d_, in, out);
}

void hpxfft::util::fftw_adapter::c2c_1d::flops(double *add, double *mul, double *fma)
{
    fftw_flops(plan_c2c_1d_, add, mul, fma);
}

void hpxfft::util::fftw_adapter::c2c_1d::print_plan(FILE *stream) { fftw_fprint_plan(plan_c2c_1d_, stream); }
