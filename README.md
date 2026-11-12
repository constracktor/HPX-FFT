# HPX-FFT: Multidimensional Distributed FFTs leveraging Asynchronous Tasks and Communication Abstractions

HPX-FFT is an open-source library for FFT computation. Leveraging the asynchronous many-task runtime HPX, we aim to combine the performance of asynchronous parallelism in C++ and a powerful communication abstraction based on an Active Global Address Space with support for industry standard FFT backends such as FFTW3.

HPX-FFT computes 2D and 3D real-to-complex FFTs of even and odd sizes:
- Shared memory: a loop-based variant (`loop`) and task-based variants (`naive`, `sync`, `opt` and `agas` in 2D;
  `naive` and `sync` in 3D).
- Distributed memory: 2D with `loop` and `agas`, 3D with slab and pencil decomposition.

The one-dimensional FFTs use an FFTW backend; FFTW and P3DFFT++ serve as reference implementations.

## Dependencies

HPX-FFT depends on [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for asynchronous task-based parallelization. Further one-dimensional FFTs are computed via an [FFTW](https://www.fftw.org/) backend.
HPX-FFT requires HPX 2.0.0 or newer (built with C++20) and a C++20 compiler.

As a reference implementation [P3DFFT++](https://p3dfft.readthedocs.io/en/latest/p3dfft++_documentation.html) is used.

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack) (1.x).
`spack-repo/setup_spack.sh` (run it from inside `spack-repo/`) clones Spack v1.1.1 into `$HOME/spack`, finds compilers
and externals, and adds the HPX recipe in `spack-repo/spack_repo/hpxfft` that is used by the CI.

- For HPX we recommend: `spack install hpxfft.hpx@2.0.0 cxxstd=20 networking=tcp max_cpu_count=256`
  (HPX 2.0 requires `cxxstd=20`, which the recipe selects by default for 2.x; 1.x defaults to `cxxstd=17`).
  `networking` takes any combination of `tcp`, `mpi` and `lci`, or `all` for all three, e.g. `networking=tcp,mpi`.
- For FFTW we recommend `spack install fftw@3.3.10 +mpi +openmp`
- For P3DFFT++ we recommend `spack install p3dfft3@develop +fftw +mpi`

The distributed tests need the TCP parcelport (`tcp`, `all` or a combination with `tcp`), because the test launcher
starts the localities over TCP. With an MPI-only HPX every multi-locality test aborts with
"Could not find usable bootstrap parcelport".

The requirements can be also installed using the provided spack environments in `spack-repo/environments`, e.g.
`spack env create hpxfft_x86_buran spack-repo/environments/hpxfft_x86_buran.yaml` (the
environments add the HPX recipe in `spack-repo/spack_repo/hpxfft` themselves). `hpxfft_ci.yaml` is the CI
environment (HPX 2.0.0, TCP). The machine environments (`hpxfft_x86_buran`, `hpxfft_x86_epyc`, `hpxfft_arm_ookami`)
use HPX 2.0.0 (`cxxstd=20`) with GCC 14.2.0 and `networking=all` (buran) or `networking=tcp,mpi` (epyc, ookami), so
all of them can run the distributed tests.

## How to Compile HPX-FFT

We provide simple compilation scripts. On known machines (ipvs-epyc1, buran and ookami; `compile_tests.sh` only
knows ipvs-epyc1) they load GCC 14.2.0 and activate the matching spack environment if it exists; elsewhere they use
the compiler and packages of the current environment. They can be easily tailored towards specific needs.

- `compile.sh`: builds the library in `build/` and installs it to `install/`.
- `compile_hpxfft_examples.sh`: builds the HPX-FFT examples in `examples/hpxfft/build` against `install/`.
- `compile_fftw_examples.sh`: builds the FFTW and P3DFFT++ reference examples in `examples/fftw/build`.
- `compile_tests.sh`: builds the tests in `test/build`.

Submitted with `sbatch`, the scripts write their logs to `output/`.

## Problem Sizes

The size of the real input is `nx x ny x nz` (3D) or `nx x ny` (2D); even and odd sizes are supported. Every row of the last dimension holds `2 * (nz / 2 + 1)` doubles, of which the first `nz` are the real input. Pass `nz` (3D) or `ny` (2D) as last argument of `initialize`; without it the size is derived from the row length and assumed to be even.

Distributed runs split the input into equal blocks of rows in x (slab and 2D) or blocks in x and y (pencil, on a square number of localities). Every locality passes its block; the blocks must have the same shape on all localities, which is checked during initialization. The dimensions that are redistributed during the computation (y for slab and 2D, z for pencil) are split into blocks of `n / L` entries, where the first `n % L` blocks hold one more, so no divisibility is required. With `set_transposed_output(true)` the result stays in this distribution; `local_y_count()` and `local_y_offset()` (slab, 2D) or `local_z_count()` and `local_z_offset()` (pencil) give the block of the locality; for the pencil, whose transposed output has the layout `(y_local, z_local, x)`, `local_y_count()` and `local_y_offset()` give its y block (the block of its position in x, not the y block of its input). The sizes must provide at least one entry per locality in the redistributed dimension (`ny` for slab, `ny / 2 + 1` for 2D, `nz / 2 + 1` for pencil).

Every object computes one transform per `initialize`: the transform moves the data out of the object and returns it
as the result. To transform new data, call `initialize` again (with the new block) before the next transform. All
classes, shared and distributed, throw `std::logic_error` if a transform is called again without a new `initialize`
(for the shared `loop` classes this holds for `fft_*_r2c_par` and `fft_*_r2c_seq` alike: one of them per `initialize`).

## Tests

Build the library (`compile.sh`) and the tests (`compile_tests.sh`), then run `ctest` in `test/build`:
- `test_*`: Catch2 unit tests of the shared and distributed variants, the AGAS components, the vector types, the
  utilities and FFTW wisdom.
- `dist_*`: correctness tests that start 1 to 16 localities on the local machine (`test/run_localities.sh`, two
  threads each) and compare the result with a naive DFT reference. They cover slab (1-4 localities), pencil
  (1, 4, 9 and 16) and 2D (1-4) with all run flags, the shared variants on one locality (`dist_shared*`), the AGAS
  components (`dist_agas2d_*`), wisdom export and import on four localities (`dist_wisdom_L4`) and sizes that must
  be rejected on every locality (`dist_reject_*`).

Most distributed tests are named `dist_<mode>_L<n>_<nx>x<ny>x<nz>_<flag>_t<0|1>`: mode `slab`, `pencil` or `2d`
(`nz` is 0), `n` localities, the run flag, and whether the result stays transposed (`t1`). They use the default HPX
ports, so ctest serializes them with a resource lock; every locality is killed after 180 s (ctest timeout 200 s).
```
ctest                   # all tests
ctest -E '^dist_' -j 4  # unit tests only
ctest -R '^dist_'       # distributed tests only
ctest -R '^dist_pencil_L9'
```
The CI runs the two groups as separate steps.

## How to Run Examples

### HPX-FFT

The executables are built in `examples/hpxfft/build`:

| Executable | Dimension | Memory |
|---|---|---|
| `hpxfft_shared_loop_2d`, `hpxfft_shared_naive_2d`, `hpxfft_shared_sync_2d`, `hpxfft_shared_opt_2d`, `hpxfft_shared_agas_2d` | 2D | shared |
| `hpxfft_shared_loop_3d`, `hpxfft_shared_naive_3d`, `hpxfft_shared_sync_3d` | 3D | shared |
| `hpxfft_distributed_loop_2d`, `hpxfft_distributed_agas_2d` | 2D | distributed |
| `hpxfft_distributed_loop_3d_slab`, `hpxfft_distributed_loop_3d_pencil` | 3D | distributed |

All examples take `--nx`, `--ny` (and `--nz` in 3D), `--plan` (FFTW planning flag: estimate, measure, patient or
exhaustive), `--header` (write the header of the runtime file), `--result` (print the result) and `--wisdom_file`
(see below). The shared loop examples choose `--run=par` (default) or `seq`. In the distributed examples `--run`
selects the communication:

| `--run` | Description | Examples |
|---|---|---|
| `scatter` (default) | HPX scatter collectives | all distributed |
| `all_to_all` | loop examples: direct exchange between the localities (routed below three localities); agas: HPX all-to-all collective | all distributed |
| `all_to_all_routed` | all-to-all routed through a root site (reference) | loop 2D, slab, pencil |
| `scatter_hierarchical` | scatter over hierarchical communicators, tuned with `--hier_arity` (default 2) and `--hier_threshold` (default 16) | loop 2D, slab, pencil |
| `scatter_async` | scatter without global synchronization; each part waits only for its own data | slab, pencil |

With `--transposed_out=1` the loop examples (2D, slab, pencil) skip the final transpose or exchange and keep the
result in the transposed distribution (see Problem Sizes).

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
4=Communication (see --run above)
5=Number of HPX threads
6=Number of HPX localities
7=Partition

srun --mpi=pmix -p $7 -N 2 -n $6 -c $5 ./examples/hpxfft/build/hpxfft_distributed_loop_2d --nx=$1 --ny=$1 --plan=$2 --header=$3 --run=$4 --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0
```
or
```
srun --mpi=pmix -p $7 -N 2 -n $6 -c $5 ./examples/hpxfft/build/hpxfft_distributed_loop_3d_slab --nx=$1 --ny=$1 --nz=$1 --plan=$2 --header=$3 --run=$4 --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0
```
The pencil example needs a square number of localities. `scripts/` holds ready-made `sbatch` scripts for the 3D
examples on buran.

### FFTW wisdom

Planning with `measure`, `patient` or `exhaustive` can take long. FFTW wisdom stores what the planner learned, so a
later run with the same sizes and flags plans almost instantly and picks the same algorithms.
All HPX-FFT examples accept `--wisdom_file=<path>`: the file is imported (if it exists) before the first plan and
written after the run. Without the option nothing changes.
```
./hpxfft_shared_loop_3d --nx=256 --ny=256 --nz=256 --plan=patient --wisdom_file=wisdom/loop_3d.wis
```
In code, use `hpxfft/util/fftw_wisdom.hpp`:
- `bool hpxfft::util::import_wisdom(const std::string &path)` merges the file into the wisdom of this process and
  returns false if it is missing or invalid (it never throws for a missing file).
- `void hpxfft::util::export_wisdom(const std::string &path)` creates the parent directory, writes to a temporary
  file and renames it atomically; it throws `std::runtime_error` on failure.

Import before calling `initialize` and export while at least one HPX-FFT object is still alive: when the last live
HPX-FFT object of the process is destroyed, it calls `fftw_cleanup()`, which discards the wisdom (destroying one
object while others are alive keeps their plans and the wisdom). Both calls share a lock with the planning in HPX-FFT, as
the FFTW planner is not thread-safe. Wisdom is per process, so every locality imports for itself. Several
localities may export to the same file safely, but the last writer wins; use one file per locality (e.g. with the
locality id in the name) to keep all of them. The distributed examples export from locality 0 only. Wisdom is only
valid for the FFTW build and machine type it was created on.

### FFTW

The reference executables in `examples/fftw/build` are `fftw_omp_*`, `fftw_threads_*` (shared) and
`fftw_mpi_omp_*`, `fftw_mpi_threads_*` (distributed), each for `2d` and `3d`, plus `p3dfft_mpi_omp` (P3DFFT++, 3D).
The `threads` variants take the same arguments as the `omp` variants.

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
or with P3DFFT++ (`ndim` is the dimension of the process grid, 1 or 2)
```
//              nodes ranks    prog       threads N_X N_Y N_Z  ndim header
srun --mpi=pmix -N 2  -n 4 p3dfft_mpi_omp    4    512 512 512   2     1
```


## How to Run Benchmarks (wip)

Compile the examples first and run the scripts from the repository root, e.g.
`./benchmark/shared_3D_benchmark.sh measure`. They pick the cluster by hostname (rostam1 with the buran partition;
the shared scripts also know login1 and ipvsmisc) and exit elsewhere. The shared scripts submit `sbatch` jobs, the
distributed and message scripts call `srun` for every run. The jobs use the scripts in `benchmark/sbatch_scripts/`
and write the runtimes to `<shared|distributed|message>_benchmark_on_<partition>/`.

- `shared_3D_benchmark.sh estimate/measure`: strong, weak and problem size scaling of the 3D shared examples (and
  the slab example on one locality) against `fftw_{omp,threads,mpi_omp,mpi_threads}_3d`.
- `distributed_3D_benchmark.sh estimate/measure <run> tcp/mpi/lci`: strong scaling of slab (1 to 16 nodes) and pencil
  (1, 4, 9, 16 nodes) against FFTW MPI and P3DFFT++, weak scaling of slab against FFTW MPI; `<run>` is passed as
  `--run`.
- `message_3D_benchmark.sh <run> tcp/mpi/lci`: message size scaling of the slab example on two nodes.
- `shared_benchmark.sh`, `distributed_benchmark.sh`, `message_benchmark.sh`: the 2D counterparts (same arguments)
  with the `*_2d` examples. The distributed and message scripts skip `hpxfft_distributed_agas_2d`: the 2D agas
  component needs `ny / 2 + 1` divisible by the number of localities, which their power-of-two sizes never are.

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
