#ifndef fftw_wisdom_H_INCLUDED
#define fftw_wisdom_H_INCLUDED

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
}  // namespace hpxfft::util
#endif  // fftw_wisdom_H_INCLUDED
