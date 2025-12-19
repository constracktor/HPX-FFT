#ifndef fftw_adapter_H_INCLUDED
#define fftw_adapter_H_INCLUDED

#include <fftw3.h>
#include <stdexcept>
#include <string>

// Enums for fftw integration
namespace hpxfft::util
{
enum class fftw_direction { forward = FFTW_FORWARD, backward = FFTW_BACKWARD };

enum class fftw_plan_flag {
    estimate = FFTW_ESTIMATE,
    measure = FFTW_MEASURE,
    patient = FFTW_PATIENT,
    exhaustive = FFTW_EXHAUSTIVE
};

inline fftw_plan_flag string_to_fftw_plan_flag(const std::string &flag_str)
{
    if (flag_str == "estimate")
    {
        return fftw_plan_flag::estimate;
    }
    else if (flag_str == "measure")
    {
        return fftw_plan_flag::measure;
    }
    else if (flag_str == "patient")
    {
        return fftw_plan_flag::patient;
    }
    else if (flag_str == "exhaustive")
    {
        return fftw_plan_flag::exhaustive;
    }
    else
    {
        throw std::invalid_argument("Invalid FFTW plan flag string");
    }
}

struct fftw_adapter_r2c
{
  public:
    void initialize(int dim_r, fftw_plan_flag plan_flag, double *in, fftw_complex *out);

    void execute_r2c(double *in, fftw_complex *out);

    void flops(double *add, double *mul, double *fma);

    void fprintf_plan(FILE *stream);

    ~fftw_adapter_r2c()
    {
        fftw_destroy_plan(plan_1d_r2c_);
        fftw_cleanup();
    }

  private:
    fftw_plan plan_1d_r2c_;
};

struct fftw_adapter_c2c
{
  public:
    void initialize(int dim_c, fftw_plan_flag plan_flag, fftw_complex *in, fftw_complex *out, fftw_direction direction);

    void execute_c2c(fftw_complex *in, fftw_complex *out);

    void flops(double *add, double *mul, double *fma);

    void fprintf_plan(FILE *stream);

    ~fftw_adapter_c2c() { fftw_destroy_plan(plan_1d_c2c_); }

  private:
    fftw_plan plan_1d_c2c_;
};
}  // namespace hpxfft::util
#endif  // fftw_adapter_H_INCLUDED
