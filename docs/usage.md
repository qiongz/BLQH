# Usage and phase diagrams

The square-torus Laughlin alignment is in [alignment.md](alignment.md).

## Filling

The total filling is $`\nu_T=\nu_\mathrm{upper}+\nu_\mathrm{lower}`$. The imbalance follows Champagne's definition

```math
\Delta\nu=\nu_\mathrm{upper}-\nu_\mathrm{lower}.
```

$`\Delta\nu=0`$ is the balanced bilayer. $`\Delta\nu=1/3`$ at $`\nu_T=1`$ is the pair of fillings $`2/3`$ and $`1/3`$. `scan-imbalance` keeps every integer occupation inside the requested window. For $`N_\phi=6`$ and $`\nu_T=1`$ those occupations are $`(3{+}3)`$, $`(4{+}2)`$, $`(5{+}1)`$, and $`(6{+}0)`$.

That scan fixes the layer electron numbers, as a gate fixes the layer densities. The tunneling operator then leaves the charge sector, so `coh_sas` is zero. The exciton weight `ex_sas`, $`\lVert T_\mathrm{sas}\psi\rVert^2/N`$, is still defined. On a single Slater determinant at $`\nu_T=1`$ this weight is at most 1. A coherent superposition exceeds 1.

## Fields

| Quantity | Flag | Where it enters | Sign |
| --- | --- | --- | --- |
| Layer separation $`d/\ell`$ | `--d` | Interlayer Coulomb only | |
| Interlayer potential | `--bias` | Diagonal | Positive adds $`\Delta_V/2`$ on each upper-layer electron |
| Zeeman energy | `--zeeman` | Diagonal | Positive adds $`\Delta_Z/2`$ on each spin-up electron |
| Spin-conserving tunneling | `--tunnel` | Upper $`\sigma\leftrightarrow`$ lower $`\sigma`$ | Matrix element is half the flag value |
| Spin-flip tunneling | `--spin-flip` | Upper $`\uparrow\leftrightarrow`$ lower $`\downarrow`$, and the reverse pair | Same normalization |

`--free` fixes only the total electron number, which is what tunneling needs in order to stay inside the sector. With one population per flavor, a tunneling matrix element leaves the sector and its expectation value is zero.

## Scans

`scan-d` sweeps the layer separation and holds the other fields fixed. `scan-phase` sweeps two axes. Each axis is one of `bias`, `tunnel`, `spin-flip`, `zeeman`, or `d`:

```bash
fqhe scan-phase --nphi 6 --upper 1/2 --lower 1/2 --spin --free \
  --x bias --y spin-flip \
  --bias -0.2:0.2:0.05 --spin-flip 0:0.2:0.05 \
  --d 1 --zeeman 0.05 --out phase.csv
```

`scan-imbalance` sweeps the Champagne plane of $`d/\ell`$ and $`\Delta\nu`$ at fixed total filling, 1 by default:

```bash
fqhe scan-imbalance --nphi 6 --nu-total 1 \
  --d 0:2:0.25 --delta-nu 0:1 --out imbalance.csv
```

`--delta-nu 1/3` keeps that single imbalance. The window `0:1` keeps every integer occupation inside it.

## Order parameters and phase labels

Every CSV row is computed from the ground state $`\psi`$ at that point.

| Column | Definition |
| --- | --- |
| `E0`, `gap` | Ground-state energy. Dense diagonalization also reports $`E_1-E_0`$ |
| `sz` | $`(N_\uparrow-N_\downarrow)/N`$ |
| `layer` | $`(N_\mathrm{upper}-N_\mathrm{lower})/N`$ |
| `coh_sas` | Spin-conserving interlayer coherence inside the sector |
| `coh_sf` | Spin-flip interlayer coherence inside the sector |
| `fluc_af` | Staggered transverse spin, $`\lVert(S^+_\mathrm{upper}-S^+_\mathrm{lower})\psi\rVert^2/N^2`$ |
| `ex_sas` | Exciton weight, including hops that leave the sector |
| `phase` | Finite-size label defined below |

The labels refer to these papers:

- **S**, **CAF**, **FM**: Zheng, Radtke, and Das Sarma, [Phys. Rev. Lett. 78, 2453 (1997)](https://doi.org/10.1103/PhysRevLett.78.2453); Das Sarma, Sachdev, and Zheng, [Phys. Rev. B 58, 4672 (1998)](https://doi.org/10.1103/PhysRevB.58.4672). S is the spin singlet or the interlayer-coherent state. CAF is the canted antiferromagnet: a ferromagnetic component along the field, and opposite in-plane tilts in the two layers. FM is the state polarized along the field.
- **C**, **I**: Champagne, Eisenstein, Pfeiffer, and West, [Phys. Rev. Lett. 100, 096801 (2008)](https://doi.org/10.1103/PhysRevLett.100.096801) and [Phys. Rev. B 78, 205310 (2008)](https://doi.org/10.1103/PhysRevB.78.205310). C is the coherent $`\nu_T=1`$ exciton condensate. I has no interlayer coherence. At $`\Delta\nu=1/3`$, going from C to I is the transition into weakly coupled single-layer states at $`2/3`$ and $`1/3`$.

The label is a finite-size indicator, not a thermodynamic phase boundary. In the balanced $`\nu_T=1`$ experiments the coherent state extends to about $`d/\ell\sim 1.8`$.

## Solvers

The dense path in `src/solve.cpp` is MKL `LAPACKE_zheevd`. Lanczos consumes the same CSR and rotates the ground state back into the Fock basis, which is what the order parameters use.

```bash
cmake -S . -B build -DMKLROOT=/opt/intel/oneapi/mkl/2025.2
cmake --build build -j
./build/fqhe point --nphi 6 --upper 2 --lower 1 --lanczos --steps 40 \
  --lanczos-kernel mkl --threads 64
./build/fqhe dump-ham --nel 6 --J -1 --out ham.npz
```

`--lanczos-kernel` selects the Lanczos matvec. `scalar` is the default loop in `solve.cpp`. `mkl` is `mkl_sparse_z_mv` plus BLAS reorthogonalization (`src/lanczos_mkl.cpp`) and uses `--threads` (the flag defaults to 80; pass 0 only from the library, which then uses 64). `gluon` runs `python/blqh/lanczos_gluon.py` on the visible GPU and accepts at most 40 steps. CMake does not compile that file. The process needs `python3` or `BLQH_PYTHON`, and `PYTHONPATH` must already be able to import Triton; the program prepends this tree's `python/` directory. Naming `mkl` or `gluon` also selects Lanczos when the sector would otherwise be dense. The CSV `solver` column is `lanczos`, `lanczos-mkl`, or `lanczos-gluon`.

`dump-ham` writes one square-torus Laughlin CSR (`N_φ = 3N`, `γ = 1`). `J < 0` is the full Fock space.
