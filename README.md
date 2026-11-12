# BLQH

Exact diagonalization of quantum Hall systems on the torus, in the lowest Landau level. A monolayer, a spinless bilayer, and a spinful bilayer share one pipeline: a flavor list builds a Fock sector, the sector builds one sparse Hamiltonian, and a dense eigensolver or Lanczos diagonalizes it.

The previous three programs are kept on the `legacy` branch.

## Build

Requires C++17, OpenMP, and oneMKL (LP64, GNU OpenMP threading). The same link line is used on Intel and on AMD EPYC. `icpc` is not required.

```bash
cmake -S . -B build -DMKLROOT=/opt/intel/oneapi/mkl/2025.2
cmake --build build -j
```

That builds `build/fqhe`, including the MKL Lanczos kernel and the launcher that calls the Gluon kernel. The Gluon kernel itself is Python (`python/blqh/lanczos_gluon.py`). CMake does not compile it. At run time it needs `python3` on `PATH` (or `BLQH_PYTHON`), a visible GPU, and a `PYTHONPATH` that can import Triton. The program prepends `python/` from this tree.

The executable is `build/fqhe`. Regression:

```bash
OMP_NUM_THREADS=80 MKL_NUM_THREADS=80 ./build/fqhe check 80
./build/fqhe check 80 full    # also the 6-electron monolayer Laughlin state
```

The square-torus Laughlin alignment, ground-state energies and the dispersion $`E(k\ell)`$, is in [docs/alignment.md](docs/alignment.md). That is the numerical example a new build should match before any other scan.

`check` verifies:

- Monolayer Laughlin $`\nu=1/3`$: the full spectra for 4 and 5 electrons, and, with `full`, the 6-electron ground-state energy and its 3-fold degeneracy
- Bilayer Halperin 111: $`N_\phi=8`$, 4 electrons per layer, $`d=0`$, $`J=K=4`$, ground-state energy per electron $`-0.636707`$
- Analytic level shifts from bias, Zeeman, interlayer tunneling, and spin-flip tunneling

## Parameters

Describe the two layers with `Layers`, or pass the same fields on the command line.

| Flag | Meaning |
| --- | --- |
| `--nphi` | Number of flux quanta $`N_\phi`$ |
| `--upper`, `--lower` | Electrons in each layer, or a filling `p/q` (`nphi * p` must be divisible by `q`) |
| `--spin` | Four flavors: upper up, upper down, lower up, lower down. Omit it for two layers and no spin |
| `--free` | Fix only the total electron number. Required when tunneling changes layer or spin |
| `--d` | Layer separation $`d/\ell`$ |
| `--bias` | Interlayer potential. Positive raises each upper-layer electron by $`\Delta_V/2`$ |
| `--zeeman` | Zeeman energy. Positive raises each spin-up electron by $`\Delta_Z/2`$ |
| `--tunnel` | Spin-conserving interlayer tunneling |
| `--spin-flip` | Spin-changing interlayer tunneling |
| `--J`, `--K` | Magnetic-translation sector. `-1` means no projection |

If the four spin counts are omitted, a spinful run puts every electron in spin up. Set them with `--upper-up`, `--upper-down`, `--lower-up`, and `--lower-down`.

One determinant is a `uint64_t`. Each flavor occupies a contiguous block of $`N_\phi`$ bits, so $`N_\phi\le 16`$ with four flavors and $`N_\phi\le 32`$ with two. Pauli exclusion is the rule that a bit is set at most once.

## Commands

```bash
# Upper layer at 1/3, lower layer at 2/3, with spin
./build/fqhe point --nphi 6 --upper 1/3 --lower 2/3 --spin --free \
  --d 1 --bias 0 --zeeman 0.05 --tunnel 0.02 --wf 8

# Lowest energy versus layer separation
./build/fqhe scan-d --nphi 6 --upper 1/2 --lower 1/2 --spin --free \
  --d 0:2:0.25 --zeeman 0.05 --out scan_d.csv

# Bias versus tunneling. Use --y spin-flip to sweep the spin-flip channel
./build/fqhe scan-phase --nphi 6 --upper 1/2 --lower 1/2 --spin --free \
  --x bias --y tunnel --bias -0.2:0.2:0.05 --tunnel 0:0.2:0.05 \
  --d 1 --zeeman 0.05 --out phase.csv

# d/l versus filling imbalance Delta nu = nu_upper - nu_lower. Total filling defaults to 1
./build/fqhe scan-imbalance --nphi 6 --nu-total 1 \
  --d 0:2:0.25 --delta-nu 0:1 --out imbalance.csv
```

```bash
# Export one Laughlin CSR. J < 0 is the full Fock space.
./build/fqhe dump-ham --nel 6 --J -1 --out ham.npz

# Lanczos kernel: scalar (default), mkl, or gluon. Gluon accepts at most 40 steps.
./build/fqhe point --nphi 6 --upper 2 --lower 1 --lanczos --steps 40 \
  --lanczos-kernel mkl --threads 64
```

Dimensions up to 800 are diagonalized densely, and a few excitations above the ground state are reported. Larger sectors use Lanczos and keep the ground state. `--dense` or `--lanczos` forces the choice. `--lanczos-kernel mkl` or `gluon` also forces Lanczos. `scalar` is the loop in `src/solve.cpp`. `mkl` uses `mkl_sparse_z_mv` and the `--threads` count (the flag defaults to 80). `gluon` keeps the Krylov basis on the GPU.

Columns and order parameters are defined in [docs/usage.md](docs/usage.md).

## Layout

```text
include/blqh/fqhe.hpp    parameters, sector, Hamiltonian, solvers, order parameters
src/sector.cpp           flavors, combinations, magnetic translations
src/hamiltonian.cpp      Coulomb interaction, one-body fields, CSR
src/solve.cpp            dense ZHEEVD and the default scalar Lanczos
src/lanczos_mkl.cpp      optional MKL sparse Lanczos
src/lanczos_gluon.cpp    writes the CSR to the Gluon kernel and reads the ground state back
src/dump.cpp             fqhe dump-ham
src/observe.cpp          order parameters
src/cli.cpp              command line
src/check.cpp            regression
python/blqh/             Gluon CSR matvec and Lanczos
tests/ref/               monolayer reference spectra
```

The dense path calls MKL `LAPACKE_zheevd`. Lanczos reads the same CSR and returns the ground state in the Fock basis. `check` still uses the scalar Lanczos.

## License

[MIT](https://choosealicense.com/licenses/mit/)
