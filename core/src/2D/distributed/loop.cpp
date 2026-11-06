#include <algorithm>
#include "../../../include/hpxfft/2D/distributed/loop.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>

// FFT backend
void hpxfft::fft2D::distributed::loop::fft_1d_r2c_inplace(const std::size_t i)
{
    fft_r2c_adapter_.execute(values_vec_.row(i), reinterpret_cast<fftw_complex *>(values_vec_.row(i)));
}

void hpxfft::fft2D::distributed::loop::fft_1d_c2c_inplace(const std::size_t i)
{
    fft_c2c_adapter_.execute(reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)),
                             reinterpret_cast<fftw_complex *>(trans_values_vec_.row(i)));
}

// split data for communication
void hpxfft::fft2D::distributed::loop::split_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j)
    {  // std::move same performance
        std::copy(values_vec_.row(i) + j * dim_c_y_part_,
                  values_vec_.row(i) + (j + 1) * dim_c_y_part_,
                  values_prep_[j].begin() + i * dim_c_y_part_);
    }
}

void hpxfft::fft2D::distributed::loop::split_trans_vec(const std::size_t i)
{
    for (std::size_t j = 0; j < num_localities_; ++j)
    {  // std::move same performance
        std::copy(trans_values_vec_.row(i) + j * dim_c_x_part_,
                  trans_values_vec_.row(i) + (j + 1) * dim_c_x_part_,
                  trans_values_prep_[j].begin() + i * dim_c_x_part_);
    }
}

void hpxfft::fft2D::distributed::loop::communicate_scatter_vec(const std::size_t i)
{
    if (this_locality_ != i)
    {
        // receive from other locality
        communication_futures_[i] =
            hpx::collectives::scatter_from<std::vector<real>>(communicators_[i], hpx::collectives::generation_arg(1));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[i], std::move(values_prep_), hpx::collectives::generation_arg(1));
    }
}

void hpxfft::fft2D::distributed::loop::communicate_scatter_trans_vec(const std::size_t i)
{
    if (this_locality_ != i)
    {
        // receive from other locality
        communication_futures_[i] =
            hpx::collectives::scatter_from<std::vector<real>>(communicators_[i], hpx::collectives::generation_arg(2));
    }
    else
    {
        // send from this locality
        communication_futures_[i] = hpx::collectives::scatter_to(
            communicators_[i], std::move(trans_values_prep_), hpx::collectives::generation_arg(2));
    }
}

// all to all communication
void hpxfft::fft2D::distributed::loop::communicate_all_to_all_vec()
{
    communication_vec_ =
        hpx::collectives::all_to_all(communicators_[0], std::move(values_prep_), hpx::collectives::generation_arg(1))
            .get();
}

void hpxfft::fft2D::distributed::loop::communicate_all_to_all_trans_vec()
{
    communication_vec_ = hpx::collectives::all_to_all(
                             communicators_[0], std::move(trans_values_prep_), hpx::collectives::generation_arg(2))
                             .get();
}

namespace
{
// Tile edge (in complex numbers) of the blocked transposes below: a tile of 16 x 16 complex numbers stays in L1 cache,
// so neither the strided reads nor the strided writes of a transpose thrash the cache.
constexpr std::size_t TILE = 16;
}  // namespace

// transpose after communication: processes the TILE rows k = tile * TILE ... of the output
void hpxfft::fft2D::distributed::loop::transpose_y_to_x(const std::size_t tile, const std::size_t i)
{
    const std::size_t factor_in = dim_c_y_part_;
    const std::size_t factor_out = 2 * num_localities_;
    const std::size_t dim_input = communication_vec_[i].size() / factor_in;
    const real *src = communication_vec_[i].data();
    const std::size_t k_begin = tile * TILE;
    const std::size_t k_end = std::min(k_begin + TILE, n_y_local_);

    for (std::size_t j_begin = 0; j_begin < dim_input; j_begin += TILE)
    {
        const std::size_t j_end = std::min(j_begin + TILE, dim_input);
        for (std::size_t k = k_begin; k < k_end; ++k)
        {
            real *dst = trans_values_vec_.row(k);
            for (std::size_t j = j_begin; j < j_end; ++j)
            {
                const std::size_t index_in = factor_in * j + 2 * k;
                const std::size_t index_out = factor_out * j + 2 * i;
                dst[index_out] = src[index_in];
                dst[index_out + 1] = src[index_in + 1];
            }
        }
    }
}

// processes the TILE rows j = tile * TILE ... of the input blocks
void hpxfft::fft2D::distributed::loop::transpose_x_to_y(const std::size_t tile, const std::size_t i)
{
    const std::size_t factor_in = dim_c_x_part_;
    const std::size_t factor_out = 2 * num_localities_;
    const std::size_t dim_input = communication_vec_[i].size() / dim_c_y_part_;
    const real *src = communication_vec_[i].data();
    const std::size_t j_begin = tile * TILE;
    const std::size_t j_end = std::min(j_begin + TILE, n_y_local_);

    for (std::size_t k_begin = 0; k_begin < dim_input; k_begin += TILE)
    {
        const std::size_t k_end = std::min(k_begin + TILE, dim_input);
        for (std::size_t k = k_begin; k < k_end; ++k)
        {
            real *dst = values_vec_.row(k);
            for (std::size_t j = j_begin; j < j_end; ++j)
            {
                const std::size_t index_in = factor_in * j + 2 * k;
                const std::size_t index_out = factor_out * j + 2 * i;
                dst[index_out] = src[index_in];
                dst[index_out + 1] = src[index_in + 1];
            }
        }
    }
}

// 2D FFT algorithm
hpxfft::fft2D::distributed::vector_2d hpxfft::fft2D::distributed::loop::fft_2d_r2c()
{
    const std::size_t num_chunks = effective_num_chunks();
    if (num_chunks > 1)
    {
        return fft_2d_r2c_pipelined(num_chunks);
    }
    /////////////////////////////////////////////////////////////////
    // first dimension
    auto start_total = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // 1d FFT r2c in y-direction
            fft_1d_r2c_inplace(i);
        });
    auto start_first_split = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_x_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_vec(i);
        });
    // communication for FFT in second dimension
    auto start_first_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_vec(i);
        }
        // global sychronization
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            communication_vec_[i] = communication_futures_[i].get();
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_vec();
    }
    else
    {
        std::cout << "Communication scheme not specified during initialization\n";
        hpx::finalize();
    }
    auto start_first_trans = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        num_localities_,
        [&](auto i)
        {
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                (n_y_local_ + TILE - 1) / TILE,
                [&](auto k)
                {
                    // blocked transpose from y-direction to x-direction
                    transpose_y_to_x(k, i);
                });
        });
    // second dimension
    auto start_second_fft = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_y_local_,
        [&](auto i)
        {
            // 1D FFT c2c in x-direction
            fft_1d_c2c_inplace(i);
        });
    auto start_second_split = t_.now();
    if (transposed_out_)
    {
        // result stays in (y_local, x) layout: no final transpose
        auto stop_total = t_.now();
        measurements_["total"] = stop_total - start_total;
        measurements_["first_fftw"] = start_first_split - start_total;
        measurements_["first_split"] = start_first_comm - start_first_split;
        measurements_["first_comm"] = start_first_trans - start_first_comm;
        measurements_["first_trans"] = start_second_fft - start_first_trans;
        measurements_["second_fftw"] = start_second_split - start_second_fft;
        measurements_["second_split"] = 0.0;
        measurements_["second_comm"] = 0.0;
        measurements_["second_trans"] = 0.0;
        return std::move(trans_values_vec_);
    }
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        n_y_local_,
        [&](auto i)
        {
            // rearrange for communication step
            split_trans_vec(i);
        });
    // communication to get original data layout
    auto start_second_comm = t_.now();
    if (COMM_FLAG_ == "scatter")
    {
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            // scatter operation from all localities
            communicate_scatter_trans_vec(i);
        }
        // global synchronization
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            communication_vec_[i] = communication_futures_[i].get();
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        // all to all operation
        // (implicit) global sychronization
        communicate_all_to_all_trans_vec();
    }
    auto start_second_trans = t_.now();
    hpx::experimental::for_loop(
        hpx::execution::par,
        0,
        num_localities_,
        [&](auto i)
        {
            hpx::experimental::for_loop(
                hpx::execution::par,
                0,
                (n_y_local_ + TILE - 1) / TILE,
                [&](auto j)
                {
                    // blocked transpose from x-direction to y-direction
                    transpose_x_to_y(j, i);
                });
        });
    auto stop_total = t_.now();

    ////////////////////////////////////////////////////////////////
    // additional runtimes
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = start_first_split - start_total;
    measurements_["first_split"] = start_first_comm - start_first_split;
    measurements_["first_comm"] = start_first_trans - start_first_comm;
    measurements_["first_trans"] = start_second_fft - start_first_trans;
    measurements_["second_fftw"] = start_second_split - start_second_fft;
    measurements_["second_split"] = start_second_comm - start_second_split;
    measurements_["second_comm"] = start_second_trans - start_second_comm;
    measurements_["second_trans"] = stop_total - start_second_trans;

    ////////////////////////////////////////////////////////////////
    return std::move(values_vec_);
}

// 2D FFT algorithm with communication overlapped with computation: both exchanges are split into chunks of rows that
// are sent as soon as they are computed.
// number of chunks used by the exchanges: automatic (0) enables overlap for all_to_all with large local data only, where
// it pays off (measured crossover on 8 nodes: 45% slower at 17 MiB, 35% faster at 64 MiB and 17% at 256 MiB per locality)
std::size_t hpxfft::fft2D::distributed::loop::effective_num_chunks() const
{
    constexpr std::size_t auto_chunks = 8;
    constexpr std::size_t auto_min_bytes = std::size_t(48) << 20;
    if (num_chunks_ != 0)
    {
        return num_chunks_;
    }
    if (COMM_FLAG_ == "all_to_all" && values_vec_.n_row() * values_vec_.n_col() * sizeof(real) >= auto_min_bytes)
    {
        return auto_chunks;
    }
    return 1;
}

hpxfft::fft2D::distributed::vector_2d hpxfft::fft2D::distributed::loop::fft_2d_r2c_pipelined(
    const std::size_t num_chunks)
{
    const std::size_t L = num_localities_;
    const bool scatter = (COMM_FLAG_ == "scatter");
    const std::size_t C1 = std::min(num_chunks, n_x_local_);  // chunks of the first exchange (rows of x)
    const std::size_t C2 = std::min(num_chunks, n_y_local_);  // chunks of the second exchange (rows of y)
    auto range = [](std::size_t n, std::size_t C, std::size_t c)
    { return std::pair<std::size_t, std::size_t>(n * c / C, n * (c + 1) / C); };
    using hpx::execution::par;
    using hpx::experimental::for_loop;

    struct pending
    {
        std::vector<hpx::future<std::vector<real>>> sc;
        hpx::future<std::vector<std::vector<real>>> a2a;
    };
    auto start_exchange = [&](std::vector<std::vector<real>> &&prep, std::size_t generation)
    {
        pending p;
        if (scatter)
        {
            p.sc.resize(L);
            for (std::size_t i = 0; i < L; ++i)
            {
                if (this_locality_ != i)
                {
                    p.sc[i] = hpx::collectives::scatter_from<std::vector<real>>(
                        communicators_[i], hpx::collectives::generation_arg(generation));
                }
                else
                {
                    p.sc[i] = hpx::collectives::scatter_to(
                        communicators_[i], std::move(prep), hpx::collectives::generation_arg(generation));
                }
            }
        }
        else
        {
            p.a2a = hpx::collectives::all_to_all(
                communicators_[0], std::move(prep), hpx::collectives::generation_arg(generation));
        }
        return p;
    };
    auto receive = [&](pending &p)
    {
        std::vector<std::vector<real>> recv;
        if (scatter)
        {
            recv.resize(L);
            for (std::size_t i = 0; i < L; ++i)
            {
                recv[i] = p.sc[i].get();
            }
        }
        else
        {
            recv = p.a2a.get();
        }
        return recv;
    };

    auto start_total = t_.now();
    double t_compute1 = 0, t_wait1 = 0, t_trans1 = 0, t_compute2 = 0, t_wait2 = 0, t_trans2 = 0;

    ////////////////////////////////////////////////////////////////
    // first exchange: r2c in y, split, send (per chunk of rows)
    std::vector<pending> pend1(C1);
    std::vector<std::pair<std::size_t, std::size_t>> xr(C1);
    for (std::size_t c = 0; c < C1; ++c)
    {
        auto t0 = t_.now();
        xr[c] = range(n_x_local_, C1, c);
        const std::size_t x0 = xr[c].first, cx = xr[c].second - xr[c].first;
        for_loop(par, x0, x0 + cx, [&](auto i) { fft_1d_r2c_inplace(i); });
        std::vector<std::vector<real>> prep(L);
        for (std::size_t j = 0; j < L; ++j)
        {
            prep[j].resize(cx * dim_c_y_part_);
        }
        for_loop(par,
                 0,
                 cx,
                 [&](auto xl)
                 {
                     for (std::size_t j = 0; j < L; ++j)
                     {
                         std::copy(values_vec_.row(x0 + xl) + j * dim_c_y_part_,
                                   values_vec_.row(x0 + xl) + (j + 1) * dim_c_y_part_,
                                   prep[j].begin() + xl * dim_c_y_part_);
                     }
                 });
        pend1[c] = start_exchange(std::move(prep), 1 + c);
        t_compute1 += t_.now() - t0;
    }

    ////////////////////////////////////////////////////////////////
    // receive the first exchange and transpose y -> x, chunk by chunk
    for (std::size_t c = 0; c < C1; ++c)
    {
        auto t0 = t_.now();
        auto recv = receive(pend1[c]);
        auto t1 = t_.now();
        t_wait1 += t1 - t0;
        const std::size_t x0 = xr[c].first, cx = xr[c].second - xr[c].first;
        for_loop(par,
                 0,
                 L * n_y_local_,
                 [&](auto idx)
                 {
                     const std::size_t i = idx / n_y_local_, k = idx % n_y_local_;
                     for (std::size_t jj = 0; jj < cx; ++jj)
                     {
                         const std::size_t index_in = dim_c_y_part_ * jj + 2 * k;
                         const std::size_t index_out = 2 * L * (x0 + jj) + 2 * i;
                         trans_values_vec_(k, index_out) = recv[i][index_in];
                         trans_values_vec_(k, index_out + 1) = recv[i][index_in + 1];
                     }
                 });
        t_trans1 += t_.now() - t1;
    }

    ////////////////////////////////////////////////////////////////
    // second exchange: c2c in x, split, send (per chunk of rows)
    std::vector<pending> pend2(C2);
    std::vector<std::pair<std::size_t, std::size_t>> yr(C2);
    if (transposed_out_)
    {
        auto t0 = t_.now();
        for_loop(par, 0, n_y_local_, [&](auto i) { fft_1d_c2c_inplace(i); });
        t_compute2 += t_.now() - t0;
        auto stop_total = t_.now();
        measurements_["total"] = stop_total - start_total;
        measurements_["first_fftw"] = t_compute1;
        measurements_["first_comm"] = t_wait1;
        measurements_["first_trans"] = t_trans1;
        measurements_["second_fftw"] = t_compute2;
        return std::move(trans_values_vec_);
    }
    for (std::size_t c = 0; c < C2; ++c)
    {
        auto t0 = t_.now();
        yr[c] = range(n_y_local_, C2, c);
        const std::size_t y0 = yr[c].first, cy = yr[c].second - yr[c].first;
        for_loop(par, y0, y0 + cy, [&](auto i) { fft_1d_c2c_inplace(i); });
        std::vector<std::vector<real>> prep(L);
        for (std::size_t j = 0; j < L; ++j)
        {
            prep[j].resize(cy * dim_c_x_part_);
        }
        for_loop(par,
                 0,
                 cy,
                 [&](auto yl)
                 {
                     for (std::size_t j = 0; j < L; ++j)
                     {
                         std::copy(trans_values_vec_.row(y0 + yl) + j * dim_c_x_part_,
                                   trans_values_vec_.row(y0 + yl) + (j + 1) * dim_c_x_part_,
                                   prep[j].begin() + yl * dim_c_x_part_);
                     }
                 });
        pend2[c] = start_exchange(std::move(prep), 1 + C1 + c);
        t_compute2 += t_.now() - t0;
    }

    ////////////////////////////////////////////////////////////////
    // receive the second exchange and transpose x -> y, chunk by chunk
    const std::size_t dim_input = dim_c_x_part_ / 2;
    for (std::size_t c = 0; c < C2; ++c)
    {
        auto t0 = t_.now();
        auto recv = receive(pend2[c]);
        auto t1 = t_.now();
        t_wait2 += t1 - t0;
        const std::size_t y0 = yr[c].first, cy = yr[c].second - yr[c].first;
        for_loop(par,
                 0,
                 L * cy,
                 [&](auto idx)
                 {
                     const std::size_t i = idx / cy, jj = idx % cy;
                     const std::size_t index_out = 2 * L * (y0 + jj) + 2 * i;
                     for (std::size_t k = 0; k < dim_input; ++k)
                     {
                         const std::size_t index_in = dim_c_x_part_ * jj + 2 * k;
                         values_vec_(k, index_out) = recv[i][index_in];
                         values_vec_(k, index_out + 1) = recv[i][index_in + 1];
                     }
                 });
        t_trans2 += t_.now() - t1;
    }
    auto stop_total = t_.now();

    // time measurements: compute stages, time waiting for data, and transposition after receiving
    measurements_["total"] = stop_total - start_total;
    measurements_["first_fftw"] = t_compute1;
    measurements_["first_comm"] = t_wait1;
    measurements_["first_trans"] = t_trans1;
    measurements_["second_fftw"] = t_compute2;
    measurements_["second_comm"] = t_wait2;
    measurements_["second_trans"] = t_trans2;
    return std::move(values_vec_);
}

// initialization
void hpxfft::fft2D::distributed::loop::initialize(
    hpxfft::fft2D::distributed::vector_2d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG)
{
    // move data into own structure
    values_vec_ = std::move(values_vec);
    // locality information
    this_locality_ = hpx::get_locality_id();
    num_localities_ = hpx::get_num_localities(hpx::launch::sync);
    // parameters
    n_x_local_ = values_vec_.n_row();
    dim_c_x_ = n_x_local_ * num_localities_;
    dim_c_y_ = values_vec_.n_col() / 2;
    dim_r_y_ = 2 * dim_c_y_ - 2;
    n_y_local_ = dim_c_y_ / num_localities_;
    dim_c_y_part_ = 2 * dim_c_y_ / num_localities_;
    dim_c_x_part_ = 2 * dim_c_x_ / num_localities_;
    // resize other data structures
    trans_values_vec_ = std::move(hpxfft::fft2D::distributed::vector_2d(n_y_local_, 2 * dim_c_x_));
    values_prep_.resize(num_localities_);
    trans_values_prep_.resize(num_localities_);
    for (std::size_t i = 0; i < num_localities_; ++i)
    {
        values_prep_[i].resize(n_x_local_ * dim_c_y_part_);
        trans_values_prep_[i].resize(n_y_local_ * dim_c_x_part_);
    }
    // create FFTW plans
    // r2c in y-direction
    fft_r2c_adapter_ = hpxfft::util::fftw_adapter::r2c_1d();
    fft_r2c_adapter_.plan(
        dim_r_y_, PLAN_FLAG, trans_values_vec_.row(0), reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)));
    // c2c in x-direction
    fft_c2c_adapter_ = hpxfft::util::fftw_adapter::c2c_1d();
    fft_c2c_adapter_.plan(dim_c_x_,
                          PLAN_FLAG,
                          reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
                          reinterpret_cast<fftw_complex *>(trans_values_vec_.row(0)),
                          hpxfft::util::fftw_adapter::direction::forward);
    // communication specific initialization
    COMM_FLAG_ = COMM_FLAG;
    if (COMM_FLAG_ == "scatter")
    {
        communication_vec_.resize(num_localities_);
        communication_futures_.resize(num_localities_);
        // setup communicators
        basename_storage_.resize(num_localities_);
        basenames_.resize(num_localities_);
        communicators_.resize(num_localities_);
        for (std::size_t i = 0; i < num_localities_; ++i)
        {
            basename_storage_[i] = std::to_string(i);
            basenames_[i] = basename_storage_[i].c_str();
            // Host the communicator on the locality that scatters through it
            // (root_site = i). With the default root site (0) every scatter is
            // staged on locality 0 first: the data crosses the network twice
            // and locality 0 becomes a hub for all of them.
            communicators_[i] = std::move(hpx::collectives::create_communicator(
                basenames_[i],
                hpx::collectives::num_sites_arg(num_localities_),
                hpx::collectives::this_site_arg(this_locality_),
                hpx::collectives::generation_arg(),
                hpx::collectives::root_site_arg(i)));
        }
        // The communicators are futures: on HPX master, scatter_to/scatter_from read num_sites from the
        // communicator's info, which is only attached once the communicator has been created/found. Make sure all
        // communicators are ready before they are used (otherwise num_sites is unset and scatter_to fails with
        // "the number of values to scatter must be equal to the number of participating sites").
        for (auto &communicator : communicators_)
        {
            communicator.wait();
        }
    }
    else if (COMM_FLAG_ == "all_to_all")
    {
        communication_vec_.resize(1);
        // setup communicators
        basename_storage_.resize(1);
        basenames_.resize(1);
        communicators_.resize(1);
        basename_storage_[0] = std::to_string(0);
        basenames_[0] = basename_storage_[0].c_str();
        communicators_[0] = std::move(hpx::collectives::create_communicator(
            basenames_[0],
            hpx::collectives::num_sites_arg(num_localities_),
            hpx::collectives::this_site_arg(this_locality_)));
    }
    else
    {
        std::cout << "Specify communication scheme: scatter or all_to_all\n";
        hpx::finalize();
    }
}

// helpers
real hpxfft::fft2D::distributed::loop::get_measurement(std::string name) { return measurements_[name]; }
