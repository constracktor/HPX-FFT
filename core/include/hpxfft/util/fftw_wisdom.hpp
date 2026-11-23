#ifndef fftw_wisdom_H_INCLUDED
#define fftw_wisdom_H_INCLUDED

#include <cstddef>
#include <initializer_list>
#include <string>

// FFTW wisdom: saved planner knowledge that makes measure/patient/exhaustive planning fast and reproducible.
// Both calls hold hpxfft::util::fftw_adapter::planner_mutex(), so they never run concurrently with planning.
// Wisdom is process-wide and is discarded by fftw_cleanup(), which runs when the last live HPX-FFT object of the
// process is destroyed: export while at least one HPX-FFT object is alive (or before the last one is destroyed).
namespace hpxfft::util
{
// Merges the wisdom stored in path into the wisdom of this process.
// Returns false if the file is missing, unreadable or not valid FFTW wisdom; never throws.
bool import_wisdom(const std::string &path);

// Writes the wisdom of this process to path, creating the parent directory if needed.
// The file is written to a temporary file next to it and renamed atomically, so readers never see a partial
// file. Each locality is a separate process: if several localities export to the same path, the last writer
// wins; use one file per locality to keep all of them. Throws std::runtime_error on failure.
void export_wisdom(const std::string &path);

// Wisdom file of one benchmark configuration. Returns explicit_path if it is not empty. Otherwise, if the environment
// variable FFTW_WISDOM_DIR is set and plan is not "estimate" (estimate does not measure, so wisdom adds nothing),
// returns <FFTW_WISDOM_DIR>/<tag>_<plan>_<dims joined by x>_n<n_ranks>_t<n_threads>.wis, else "" (no wisdom).
// Every locality computes the same name, so a run on several localities shares one file.
std::string wisdom_file_name(const std::string &explicit_path,
                             const std::string &tag,
                             const std::string &plan,
                             std::initializer_list<std::size_t> dims,
                             std::size_t n_ranks,
                             std::size_t n_threads);
}  // namespace hpxfft::util
#endif  // fftw_wisdom_H_INCLUDED
