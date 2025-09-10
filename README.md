# HPX-FFT: Multidimensional Distributed FFTs leveraging Asynchronous Tasks and Communication Abstractions

HPX-FFT is an open-source library for FFT computation. Leveraging the asynchronous many-task runtime HPX, we aim to combine the performance of asynchronous parallelism in C++ and a powerfull communication abstraction based on an Active Global Adress Space with support for industry standard FFT backends such as FFTW3.

## Dependencies

HPX-FFT depends on [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for asynchronous task-based parallelization. Further one-dimensional FFTs are computed via an [FFTW](https://www.fftw.org/) backend.

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack).
For HPX we recommend: `spack install hpx@1.10.0 networking={tcp/mpi/lci} max_cpu_count=256`
For FFTW we recommend `spack install fftw@3.3.10 +mpi +openmp`

Note that the LCI variant of HPX currently requires to add the [Octo-Tiger repo](https://github.com/G-071/octotiger-spack) to spack.
Alternatively, both librarires can be built from source.

## How to Compile HPX-FFT

We provide a simple compilation script for HPX-FFT. The script assumes an gcc compiler version 14.2.0. However, it can be easily tailored towards specific needs.

## How to Compile Examples

HPX-FFT comes with several examples for HPX-FFT itself as well as for a FFTW reference. The examples can be compiled with the respective compilation scripts.

## How to Run Examples

### HPX-FFT

```
1=Base size of the problem
2=FFTW Planning flag (estimate/measure)
3=Print info into runtime file (true/false)
4=Number of HPX threads
./hpxfft_shared_loop --nx=$1 --ny=$1 --plan=$2 --header=$3 --hpx:threads=$4
```
```
1=Base size of the problem
2=FFTW Planning flag (estimate/measure)
3=Print info into runtime file (true/false)
4=Collective operation (scatter/all_to_all)
5=Number of HPX threads
./hpxfft_distributed_loop --nx=$1 --ny=$1 --plan=$2 --header=$3 --run=$4 --hpx:threads=$5
```

### FFTW

```
//        threads N_X N_Y  plan     header
./fftw_omp   1      8  14 estimate    0
```
```
//   nodes ranks    prog    threads N_X N_Y    plan  header
srun -N 2  -n 4 fftw_mpi_omp   4     8   14  estimate  0
    mpirun -n 4 fftw_mpi_omp   4     8   14  estimate  0
```

## How to Run Benchmarks (wip)

### Shared
All:

- `./shared_benchmark.sh estimate/measure partition_name`

Executables only:

- fft_hpx_loop_shared/fft_hpx_task_sync_shared/fft_hpx_task_shared/fft_hpx_task_naive_shared
/fft_hpx_task_agas_shared/fft_hpx_loop/fft_hpx_task_agas:
`sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh executable_name estimate/measure $THREAD_POW`


- fftw_hpx/fftw_mpi_threads/fftw_mpi_omp:
`sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh executable_name estimate/measure $THREAD_POW`

### Distributed
All:

- `./distributed_benchmark.sh estimate/measure partition_name`

Executables only: 

- fftw_hpx_loop/fftw_hpx_tasg_agas:
`sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_hpx_dist.sh executable_name estimate/measure $NODES`

- fftw_mpi_threads/fftw_mpi_omp:
`sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_fftw_dist.sh executable_name estimate/measure $NODES $THREADS`

## The Team

The HPX-FFT library is developed by the [Scientific Computing](https://www.ipvs.uni-stuttgart.de/departments/sc/)
department at IPVS at the University of Stuttgart.
The project is a joined effort of multiple graduate, and PhD students under the supervision of
[Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/).
We specifically thank the follow contributors:

- [Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/):
  Maintainer and [initial framework](https://doi.org/10.1007/978-3-031-32316-4_5).

## How To Cite

```
@InProceedings{GPRat2025,
  author={Strack, Alexander and Taylor, Christopher and Diehl, Patrick and Pfl{\"u}ger, Dirk},
  title={{Experiences Porting Distributed Applications to Asynchronous Tasks: A Multidimensional FFT Case-study}},
  booktitle={Asynchronous Many-Task Systems and Applications},
  year={2024},
  publisher={Springer Nature}
}
```
