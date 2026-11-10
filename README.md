# HPX-FFT: Multidimensional Distributed FFTs leveraging Asynchronous Tasks and Communication Abstractions

HPX-FFT is an open-source library for FFT computation. Leveraging the asynchronous many-task runtime HPX, we aim to combine the performance of asynchronous parallelism in C++ and a powerfull communication abstraction based on an Active Global Adress Space with support for industry standard FFT backends such as FFTW3.

## Dependencies

HPX-FFT depends on [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for asynchronous task-based parallelization. Further one-dimensional FFTs are computed via an [FFTW](https://www.fftw.org/) backend.

As a reference implementation [P3DFFT++](https://p3dfft.readthedocs.io/en/latest/p3dfft++_documentation.html) is used. 

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack).
For HPX we recommend: `spack install hpx@1.11.0 networking={tcp/mpi/lci/all} max_cpu_count=256`
For FFTW we recommend `spack install fftw@3.3.10 +mpi +openmp`
For P3DFFT++ we recommend `spack install p3dfft3@develop +fftw +mpi`
The requirements can be also installed using the provided spack environments. 
The spack-repo directory add support for installing HPX with the LCI or all parcelports at once.

## How to Compile HPX-FFT

We provide a simple compilation script for HPX-FFT. The script assumes an gcc compiler version 14.2.0. However, it can be easily tailored towards specific needs.

## How to Compile Examples:

HPX-FFT comes with several examples for HPX-FFT itself as well as for a FFTW reference. The examples can be compiled with the respective compilation scripts.

## Problem Sizes

The size of the real input is `nx x ny x nz` (3D) or `nx x ny` (2D); even and odd sizes are supported. Every row of the last dimension holds `2 * (nz / 2 + 1)` doubles, of which the first `nz` are the real input. Pass `nz` (3D) or `ny` (2D) as last argument of `initialize`; without it the size is derived from the row length and assumed to be even.

Distributed runs split the input into equal blocks of rows in x (slab and 2D) or blocks in x and y (pencil, on a square number of localities). Every locality passes its block; the blocks must have the same shape on all localities, which is checked during initialization. The dimensions that are redistributed during the computation (y for slab and 2D, z for pencil) are split into blocks of `n / L` entries, where the first `n % L` blocks hold one more, so no divisibility is required. With `set_transposed_output(true)` the result stays in this distribution; `local_y_count()` and `local_y_offset()` (slab, 2D) or `local_z_count()` and `local_z_offset()` (pencil) give the block of the locality. The sizes must provide at least one entry per locality in the redistributed dimension (`ny` for slab, `ny / 2 + 1` for 2D, `nz / 2 + 1` for pencil).

## Tests

`ctest` runs the distributed tests on 1 to 9 localities of the local machine (`test/run_localities.sh`) and compares the result with a naive DFT. The tests use the default HPX ports and must not run concurrently.

## How to Run Examples

### HPX-FFT

Shared examples:
```
1=Base size of the problem
2=FFTW Planning flag (estimate/measure)
3=Print info into runtime file (true/false)
4=Number of HPX threads
./hpxfft_shared_loop_2d --nx=$1 --ny=$1 --plan=$2 --header=$3 --hpx:threads=$4
```
or
```
./hpxfft_shared_naive_3d --nx=$1 --ny=$1 --nz=$1 --plan=$2 --header=$3 --hpx:threads=$4
```

Distributed examples with slurm:
```
1=Base size of the problem
2=FFTW Planning flag (estimate/measure)
3=Print info into runtime file (true/false)
4=Collective operation (scatter/all_to_all)
5=Number of HPX threads
6=Number of HPX localities
7=Partition

srun --mpi=pmix -p $6 -N 2 -n $6 -c $5 ./examples/hpxfft/build/hpxfft_distributed_loop_2d --nx=$1 --ny=$1 --plan=$2 --header=$3 --run=$4 --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0
```
or
```
srun --mpi=pmix -p $6 -N 2 -n $6 -c $5 ./examples/hpxfft/build/hpxfft_distributed_loop_3d_slab --nx=$1 --ny=$1 --nz=$1 --plan=$2 --header=$3 --run=$4 --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0
```

### FFTW

Shared examples:
```
//            threads N_X N_Y  plan     header
./fftw_omp_2d   1      8  14 estimate    0
```
or
```
//            threads N_X N_Y N_Z  plan     header
./fftw_omp_3d   1      8  14  32 estimate    0
```

Distributed examples with srun or mpirun:
```
//              nodes ranks    prog       threads N_X N_Y    plan  header
srun --mpi=pmix -N 2  -n 4 fftw_mpi_omp_2d   4     8   14  estimate  0
               mpirun -n 4 fftw_mpi_omp_2d   4     8   14  estimate  0
```
or
```
//              nodes ranks    prog       threads N_X N_Y N_Z    plan  header
srun --mpi=pmix -N 2  -n 4 fftw_mpi_omp_3d   4     8   14 32   estimate  0
               mpirun -n 4 fftw_mpi_omp_3d   4     8   14 32   estimate  0
```


## How to Run Benchmarks (wip)

### Shared
All:

- `./shared_benchmark.sh estimate/measure`
- `./shared_benchmark_3d.sh estimate/measure`

Executables only:

- fft_hpx_loop_shared/fft_hpx_task_sync_shared/fft_hpx_task_shared/fft_hpx_task_naive_shared
/fft_hpx_task_agas_shared/fft_hpx_loop/fft_hpx_task_agas:
`sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh executable_name estimate/measure $THREAD_POW`


- fftw_hpx/fftw_mpi_threads/fftw_mpi_omp:
`sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh executable_name estimate/measure $THREAD_POW`

### Distributed

- `./distributed_benchmark.sh estimate/measure scatter/all_to_all tcp/mpi/lci`
- `./distributed_benchmark_3d.sh estimate/measure scatter/all_to_all/scatter_async tcp/mpi/lci`

## The Team

The HPX-FFT library is developed by the [Scientific Computing](https://www.ipvs.uni-stuttgart.de/departments/sc/)
department at IPVS at the University of Stuttgart.
The project is a joined effort of multiple graduate, and PhD students under the supervision of
[Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/).
We specifically thank the follow contributors:

- [Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/):
  Maintainer and [initial framework](https://doi.org/10.1007/978-3-031-32316-4_5).
- Heiko Häfner: Testing, CI and 3D support.

## How To Cite

```
@InProceedings{Strack2024_hpxfft,
  author={Strack, Alexander and Taylor, Christopher and Diehl, Patrick and Pfl{\"u}ger, Dirk},
  title={{Experiences Porting Distributed Applications to Asynchronous Tasks: A Multidimensional FFT Case-study}},
  booktitle={Asynchronous Many-Task Systems and Applications},
  year={2024},
  publisher={Springer Nature}
}
```
or
```
@techreport{haefner2026hpxfft,
  author      = {Heiko H{\"a}fner and Alexander Strack},
  title       = {{HPXFFT}: A Three-Dimensional {FFT} Framework Using Asynchronous Parallel Computation},
  institution = {Institute of Parallel and Distributed Systems, University of Stuttgart},
  type        = {Project Report},
  address     = {Stuttgart, Germany},
  year        = {2026}
}
```
