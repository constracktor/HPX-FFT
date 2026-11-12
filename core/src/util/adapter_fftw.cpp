#include "../../include/hpxfft/util/adapter_fftw.hpp"

// FFTW adapter implementation
std::mutex &hpxfft::util::fftw_adapter::planner_mutex()
{
    static std::mutex mutex;
    return mutex;
}

void hpxfft::util::fftw_adapter::cleanup()
{
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftw_cleanup();
}

void hpxfft::util::fftw_adapter::r2c_1d::plan(int dim_r, std::string plan_flag, double *in, fftw_complex *out)
{
    // create FFTW plan
    std::lock_guard<std::mutex> lock(planner_mutex());
    plan_r2c_1d_ = fftw_plan_dft_r2c_1d(dim_r, in, out, static_cast<unsigned>(string_to_fftw_plan_flag(plan_flag)));
}

hpxfft::util::fftw_adapter::r2c_1d::~r2c_1d()
{
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftw_destroy_plan(plan_r2c_1d_);
}

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
    // create FFTW plan
    std::lock_guard<std::mutex> lock(planner_mutex());
    plan_c2c_1d_ = fftw_plan_dft_1d(
        dim_c, in, out, static_cast<int>(direction), static_cast<unsigned>(string_to_fftw_plan_flag(plan_flag)));
}

hpxfft::util::fftw_adapter::c2c_1d::~c2c_1d()
{
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftw_destroy_plan(plan_c2c_1d_);
}

void hpxfft::util::fftw_adapter::c2c_1d::execute(fftw_complex *in, fftw_complex *out)
{
    fftw_execute_dft(plan_c2c_1d_, in, out);
}

void hpxfft::util::fftw_adapter::c2c_1d::flops(double *add, double *mul, double *fma)
{
    fftw_flops(plan_c2c_1d_, add, mul, fma);
}

void hpxfft::util::fftw_adapter::c2c_1d::print_plan(FILE *stream) { fftw_fprint_plan(plan_c2c_1d_, stream); }
