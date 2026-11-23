#pragma once
// Allocation without zero-filling and a first touch of the pages by the threads that transform them.
// A std::vector<double> zero-fills its buffer from one thread, so all pages end up in one NUMA domain and every
// thread of the transform reads through the memory controllers of that domain (up to 10x slower at large sizes).
#include <cstddef>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace first_touch
{
// std::allocator that default-initializes: a vector of doubles is allocated but not touched
template <typename T>
struct default_init_allocator : std::allocator<T>
{
    template <typename U>
    struct rebind
    {
        using other = default_init_allocator<U>;
    };

    using std::allocator<T>::allocator;

    template <typename U>
    void construct(U *p) noexcept(std::is_nothrow_default_constructible_v<U>)
    {
        ::new (static_cast<void *>(p)) U;
    }

    template <typename U, typename... Args>
    void construct(U *p, Args &&...args)
    {
        ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }
};

// Calls f(begin, end) for n_threads contiguous blocks of 0 .. rows, one block per thread (the blocks have the same
// size as the loop blocks of FFTW). Uses OpenMP when compiled with it, std::thread otherwise.
template <typename F>
void for_each_row_block(std::size_t rows, int n_threads, F &&f)
{
    if (n_threads < 1)
    {
        n_threads = 1;
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
        const std::size_t n = omp_get_num_threads();
        const std::size_t t = omp_get_thread_num();
        f(rows * t / n, rows * (t + 1) / n);
    }
#else
    const std::size_t n = n_threads;
    std::vector<std::thread> threads;
    for (std::size_t t = 1; t < n; ++t)
    {
        threads.emplace_back([&f, rows, n, t]() { f(rows * t / n, rows * (t + 1) / n); });
    }
    f(0, rows / n);
    for (auto &th : threads)
    {
        th.join();
    }
#endif
}
}  // namespace first_touch
