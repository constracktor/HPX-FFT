#ifndef fftw_adapter_H_INCLUDED
#define fftw_adapter_H_INCLUDED

#include <fftw3.h>
#include <mutex>
#include <stdexcept>
#include <string>

// Enums for fftw integration
namespace hpxfft::util::fftw_adapter
{
enum class direction { forward = FFTW_FORWARD, backward = FFTW_BACKWARD };

enum class plan_flag {
    estimate = FFTW_ESTIMATE,
    measure = FFTW_MEASURE,
    patient = FFTW_PATIENT,
    exhaustive = FFTW_EXHAUSTIVE
};

void cleanup();

// The FFTW planner is not thread-safe (only fftw_execute is): every planner call of this library (planning,
// destroying plans, cleanup, wisdom import/export) holds this mutex. Lock it as well when calling such FFTW
// functions directly while HPX-FFT objects may be planning on other threads.
std::mutex &planner_mutex();

// Calls fftw_cleanup() on destruction. Place it as the first data member of a class that owns plans.
struct cleanup_guard
{
    cleanup_guard() = default;
    cleanup_guard(const cleanup_guard &) = delete;
    cleanup_guard &operator=(const cleanup_guard &) = delete;
    ~cleanup_guard() { cleanup(); }
};

inline plan_flag string_to_fftw_plan_flag(const std::string &flag_str)
{
    if (flag_str == "estimate")
    {
        return plan_flag::estimate;
    }
    else if (flag_str == "measure")
    {
        return plan_flag::measure;
    }
    else if (flag_str == "patient")
    {
        return plan_flag::patient;
    }
    else if (flag_str == "exhaustive")
    {
        return plan_flag::exhaustive;
    }
    else
    {
        throw std::invalid_argument("Invalid FFTW plan flag string");
    }
}

struct r2c_1d
{
  public:
    void plan(int dim_r, std::string plan_flag, double *in, fftw_complex *out);

    void execute(double *in, fftw_complex *out);

    void flops(double *add, double *mul, double *fma);

    void print_plan(FILE *stream);

    ~r2c_1d();

  private:
    // null until planned: an adapter that never planned (initialize threw) must be destructible
    fftw_plan plan_r2c_1d_ = nullptr;
};

struct c2c_1d
{
  public:
    void plan(int dim_c, std::string plan_flag, fftw_complex *in, fftw_complex *out, fftw_adapter::direction direction);

    void execute(fftw_complex *in, fftw_complex *out);

    void flops(double *add, double *mul, double *fma);

    void print_plan(FILE *stream);

    ~c2c_1d();

  private:
    fftw_plan plan_c2c_1d_ = nullptr;
};
}  // namespace hpxfft::util::fftw_adapter
#endif  // fftw_adapter_H_INCLUDED
