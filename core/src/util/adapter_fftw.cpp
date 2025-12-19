#include "../../include/hpxfft/util/adapter_fftw.hpp"

// FFTW adapter implementation
void hpxfft::util::fftw_adapter_r2c::initialize(int dim_r, fftw_plan_flag plan_flag, double *in, fftw_complex *out)
{
    // create FFTW plans
    plan_1d_r2c_ = fftw_plan_dft_r2c_1d(dim_r, in, out, static_cast<unsigned>(plan_flag));
}

void hpxfft::util::fftw_adapter_r2c::execute_r2c(double *in, fftw_complex *out)
{
    fftw_execute_dft_r2c(plan_1d_r2c_, in, out);
}

void hpxfft::util::fftw_adapter_r2c::flops(double *add, double *mul, double *fma)
{
    fftw_flops(plan_1d_r2c_, add, mul, fma);
}

void hpxfft::util::fftw_adapter_r2c::fprintf_plan(FILE *stream) { fftw_fprint_plan(plan_1d_r2c_, stream); }

void hpxfft::util::fftw_adapter_c2c::initialize(
    int dim_c, fftw_plan_flag plan_flag, fftw_complex *in, fftw_complex *out, fftw_direction direction)
{
    // create FFTW plans
    plan_1d_c2c_ = fftw_plan_dft_1d(dim_c, in, out, static_cast<int>(direction), static_cast<unsigned>(plan_flag));
}

void hpxfft::util::fftw_adapter_c2c::execute_c2c(fftw_complex *in, fftw_complex *out)
{
    fftw_execute_dft(plan_1d_c2c_, in, out);
}

void hpxfft::util::fftw_adapter_c2c::flops(double *add, double *mul, double *fma)
{
    fftw_flops(plan_1d_c2c_, add, mul, fma);
}

void hpxfft::util::fftw_adapter_c2c::fprintf_plan(FILE *stream) { fftw_fprint_plan(plan_1d_c2c_, stream); }
