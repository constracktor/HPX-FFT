#include <hpx/future.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <string>
#include <vector>

namespace utils
{
void start_hpx_runtime(int argc, char **argv) { hpx::start(nullptr, argc, argv); }

void resume_hpx_runtime() { hpx::resume(); }

void suspend_hpx_runtime() { hpx::suspend(); }

void stop_hpx_runtime()
{
    hpx::post([]() { hpx::finalize(); });
    hpx::stop();
}
}  // namespace utils
