# ShellHomo: GPU-Accelerated Reissner–Mindlin Homogenization for Shellular Unit Cells

ShellHomo computes the effective (homogenized) elasticity tensor **C<sup>H</sup>** of a
periodic shellular unit cell whose mid-surface is given as an **implicit field**. The shell is
modeled as a Reissner–Mindlin shell (formulated with tangential differential calculus) and
discretized with an unfitted TraceFEM on a fixed background tetrahedral grid, so neither a
surface mesh nor a volumetric mesh of the shell is needed. The whole pipeline runs on the GPU.

This repository contains the reference implementation of the paper

> Ruihong Cen and Bohan Wang. GPU-Accelerated Reissner–Mindlin Homogenization for Shellular

The current release contains the **forward homogenization** and the **gradient-based inverse
design** in the space of six combined TPMS basis functions (sensitivity pass + Adam). 

---

## Requirements

| Component | Version |
|---|---|
| NVIDIA GPU | any GPU supported by your CUDA toolkit |
| CUDA Toolkit | 12.x or 13.x (nvcc, CUDA runtime, Thrust/CUB) |
| CMake | ≥ 3.24 |
| Host compiler | a C++17 compiler supported by your CUDA toolkit (e.g. GCC) |
| Python (optional) | 3.8+ with NumPy, to generate example inputs |

Tested on Ubuntu 24.04 with CUDA 13.1, GCC 13.3 and an NVIDIA H200 (`sm_90`).
The only third-party dependency, [RapidJSON](https://github.com/Tencent/rapidjson) (header-only,
MIT license), is included under `third_party/`.

## Build

```bash
git clone <this repository> shellhomo
cd shellhomo
cmake -S . -B build
cmake --build build -j
```

This produces two executables: `build/shellhomo` (forward homogenization) and
`build/shellhomo_inverse` (inverse design, see [below](#inverse-design-in-the-combined-tpms-space)).

By default the code is compiled for the GPU(s) found on the build machine
(`CMAKE_CUDA_ARCHITECTURES=native`). To build for other GPUs, for example on a machine without a
GPU, list the architectures explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="90;120"   # e.g. H100/H200 and RTX 50-series
```

If CMake picks the wrong CUDA installation, point it at the right one with
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

## Quick start

```bash
# 1) Generate example inputs (TPMS fields sampled on 257^3 grids; ~136 MB each)
python3 scripts/generate_tpms_sdf.py --out data --names Primitive Gyroid Diamond

# 2) Homogenize every .npy file in data/
./build/shellhomo configs/default.json data/ --log results.json

# 3) Inverse design: maximize the mean diagonal stiffness in the combined-TPMS space
./build/shellhomo_inverse configs/inverse.json --out inverse_stiffness
```

For each input the program prints per-stage timings and the 6×6 tensor C<sup>H</sup>, and
appends a record to `results.json`.

Running without inputs homogenizes a built-in analytic gyroid, which is a quick way to check
the build:

```bash
./build/shellhomo configs/default.json
```

With `configs/default.json` (48³ grid, t = 0.02, E = 2·10⁶, ν = 0.3) the gyroid gives
C<sub>11</sub> ≈ 5.91·10⁴, C<sub>12</sub> ≈ 2.90·10⁴ and C<sub>44</sub> ≈ 1.66·10⁴.

### Command line

```
shellhomo <config.json> [input ...] [--log FILE] [--warmup N]
```

| Argument | Meaning |
|---|---|
| `config.json` | grid resolution and material parameters (see below) |
| `input` | a `.npy` implicit field, or a directory (all `*.npy` files inside are processed in sorted order). Several inputs may be given. Without inputs the built-in gyroid is used. |
| `--log FILE` | JSON file the results are appended to (default `results.json`) |
| `--warmup N` | number of untimed, unlogged warm-up runs on the built-in gyroid before the inputs (default 3; they absorb CUDA context creation and first-launch overheads so that the logged timings are representative) |

### Configuration file

```json
{
    "nx": 48, "ny": 48, "nz": 48,
    "thickness": 0.02,
    "E": 2000000.0,
    "nu": 0.3
}
```

| Key | Meaning |
|---|---|
| `nx`, `ny`, `nz` | cells per axis of the background grid on the unit cube. Must be equal and should be even (the 5-tet decomposition alternates between neighbouring cells, so an even resolution is needed for a conforming periodic mesh). |
| `thickness` | shell thickness *t*, in the units of the unit cell (side length 1) |
| `E`, `nu` | Young's modulus and Poisson's ratio of the base material |

Additional keys are ignored.

## Input format

An input is a scalar field *f* whose zero level set is the shell mid-surface:

* NumPy `.npy`, **float64**, either 3D with shape `(257, 257, 257)` or flattened to length 257³;
* sample `[i, j, k]` holds *f* at the cell centre ((i+½)/257, (j+½)/257, (k+½)/257) of the unit
  cube, i.e. the array is indexed `[x][y][z]` (as produced by
  `np.meshgrid(..., indexing="ij")`);
* *f* should be periodic with period 1 along x, y and z. It does not need to be a signed
  distance field.

`scripts/generate_tpms_sdf.py` writes fields in exactly this format for 17 classical TPMS
(`--names` selects a subset; run with `-h` for the list).

## Output

Each run appends one record to the JSON log:

| Key | Meaning |
|---|---|
| `name` | input file (or `builtin_gyroid`) |
| `homogenization_matrix` | C<sup>H</sup>, 6×6, Voigt order (xx, yy, zz, xy, yz, xz) with engineering shear strains; same stress units as `E` |
| `matrix_size` | number of unknowns of the linear system (6 per active grid node) |
| `num_activeTets`, `num_quads` | number of cut tetrahedra and of quadrature slots (4 per element) |
| `total_time_ms` | wall-clock time of one homogenization query (excludes reading the input file) |
| `init_grid_time`, `construct_stiffness_time_ms`, `construct_stab_time_ms`, `construct_rhs_time_ms`, `solve_time_ms` | per-stage GPU times (ms) |
| `memory_usage_mb` | peak device memory in use during the query (sampled with `cudaMemGetInfo`) |

## Inverse design in the combined-TPMS space

`shellhomo_inverse` runs the gradient-based inverse-design example of the paper. The design
variables are the six coefficients c<sub>k</sub> of

f(**x**) = Σ<sub>k</sub> c<sub>k</sub> f<sub>k</sub>(**x**), with X = 2πx, Y = 2πy, Z = 2πz and

| k | f<sub>k</sub> |
|---|---|
| 1 | cos X + cos Y + cos Z |
| 2 | sin X sin Y + sin Y sin Z + sin Z sin X |
| 3 | cos X cos Y + cos Y cos Z + cos Z cos X |
| 4 | sin X cos Y + sin Y cos Z + sin Z cos X |
| 5 | sin X sin Y cos Z + sin Y sin Z cos X + sin Z sin X cos Y |
| 6 | cos X cos Y cos Z |

```bash
./build/shellhomo_inverse configs/inverse.json --out inverse_stiffness
```

The configuration file has the keys of the forward configuration plus

| Key | Meaning |
|---|---|
| `objective` | `"bulk"`, `"shear"` or `"stiffness"` (required, see below) |
| `seed` | seed of the random initial coefficients, drawn uniformly from [0, 1) (optional, default 42) |

| `objective` | maximized quantity |
|---|---|
| `bulk` | bulk modulus K = (C<sub>11</sub> + C<sub>22</sub> + C<sub>33</sub> + 2 (C<sub>12</sub> + C<sub>13</sub> + C<sub>23</sub>)) / 9 |
| `shear` | shear modulus G = (C<sub>44</sub> + C<sub>55</sub> + C<sub>66</sub>) / 3 |
| `stiffness` | mean diagonal stiffness (C<sub>11</sub> + C<sub>22</sub> + C<sub>33</sub> + C<sub>44</sub> + C<sub>55</sub> + C<sub>66</sub>) / 6 |

The program minimizes J = −(maximized quantity), so the logged objective is negative. Every
iteration homogenizes the current field, evaluates dJ/dc<sub>k</sub> with the sensitivity pass,
and takes one Adam step (β<sub>1</sub> = 0.95, β<sub>2</sub> = 0.999, ε = 10⁻⁸). The learning
rate decays from 5·10⁻³ to 10⁻⁴ with a cosine schedule over 300 iterations; these settings are
constants in `src/inverse_design.cu`. One run at 48³ takes about one minute on an H200.

The sensitivity pass (`rm_gradient.cuh`) computes, at every surface quadrature point, the
derivatives of all C<sup>H</sup><sub>IJ</sub> with respect to the field gradient and Hessian, and
chains them with the analytic gradients and Hessians of the basis functions
(`combined_field.cuh`). Quadrature points, weights and the set of cut elements are held fixed in
this evaluation and rebuilt at the next iteration, so the objective can increase temporarily when
a step changes the surface a lot (for example near a topology change).

Output written to the `--out` directory (default `inverse_<objective>`):

| File | Content |
|---|---|
| `optimization_history.csv` | one row per iteration: `iter, objective, lr, c0..c5, g0..g5` (coefficients and gradient before the Adam step) |
| `sdf_iter_NNNN.npy` | the field every 5 iterations and at the last iteration, 257³ in the [input format](#input-format); 61 files of 136 MB, about 8 GB per run |
| `sdf_best.npy` | the field with the lowest objective |
| `results.json` | the homogenization record of every iteration (fields as in [Output](#output); `name` is the iteration) |

The snapshots can be homogenized again directly, e.g.
`./build/shellhomo configs/default.json inverse_stiffness/sdf_best.npy`.

## Method overview

One query (`simulator::run_pipeline`) performs:

1. **Background grid** (`grid.cuh`). A regular grid of `nx³` cubes, each split into 5 tetrahedra.
   Grid vertices are relaxed away from the zero level set (40 red–black sweeps, launched as one
   CUDA graph) to avoid tiny cut polygons. Tetrahedra whose vertices change sign are the active
   elements.
2. **Surface quadrature** (`grid.cuh`). The zero level set is intersected with every active
   element (Newton on the edges); the cut triangle uses a 3-point rule and the cut quadrilateral a
   2×2 Gauss rule.
3. **Element matrices** (`rm_shell.cuh`). Membrane, bending and transverse-shear terms of the
   Reissner–Mindlin shell with linear shape functions for the mid-surface displacement **u** and
   the director rotation **w** (24 DOFs per element), plus the TraceFEM stabilization terms. One
   CUDA warp computes one element.
4. **Assembly and periodicity** (`simulator.cuh`). Element matrices are assembled into CSR.
   Periodicity is imposed by identifying the vertices on opposite faces of the unit cell; the
   rigid-translation null space is removed by fixing grid vertices.
5. **Six cell problems** (`block_pcg.cuh`). The six unit test strains share one matrix and are
   solved together by a Jacobi-preconditioned CG with fused kernels.
6. **Effective tensor**. C<sup>H</sup><sub>ij</sub> = Σ<sub>e</sub> y<sub>i</sub><sup>T</sup> K<sub>e</sub> y<sub>j</sub>, where
   y combines the affine test field with the computed periodic fluctuation.

The implicit field is stored in a 3D CUDA texture with trilinear filtering; normals and the
Weingarten map are obtained by finite differences on this texture.

### Default numerical parameters

| Parameter | Value | Location |
|---|---|---|
| precision | FP32 | everywhere |
| field texture | 257³ samples | `sdf.cuh` |
| CG tolerance / max. iterations | relative 10⁻² (Jacobi-preconditioned residual norm) / 2000 | `simulator.cuh` |
| volumetric stabilization ρ | (h + 1/h)·10⁴, h = 1/nx | `simulator.cuh` |
| normal penalty of **w**, ρ<sub>w</sub> | E·t | `rm_shell.cuh` |
| grid relaxation | 40 sweeps, h = 0.2, α = 0.1, k = 1.5, dt = 0.05 | `grid.cuh` |
| constitutive law | σ = 2με + λ tr(ε) I with μ = E/(2(1+ν)), λ = Eν/(1−ν²) | `simulator.cuh` |

## Repository layout

```
CMakeLists.txt
configs/default.json          example configuration (forward)
configs/inverse.json          example configuration (inverse design)
include/shellhomo/
  linalg.cuh                  small host/device vector and matrix types, CUDA helpers
  sdf.cuh                     .npy reader, 3D texture of the implicit field, built-in gyroid
  grid.cuh                    background grid, relaxation, cut detection, surface quadrature
  rm_shell.cuh                Reissner–Mindlin operators and element kernels
  block_pcg.cuh               multi-right-hand-side Jacobi-preconditioned CG
  simulator.cuh               assembly, cell problems, homogenized tensor, design gradient
  rm_gradient.cuh             sensitivity of C^H w.r.t. the field gradient and Hessian
  combined_field.cuh          the six TPMS basis functions, snapshot export
  inverse_design.cuh          objectives and the Adam optimizer
  logger.cuh                  JSON result log, peak GPU memory monitor
  config.h                    JSON configuration reader
src/main.cu                   forward homogenization driver (shellhomo)
src/inverse_design.cu         inverse-design driver (shellhomo_inverse)
scripts/generate_tpms_sdf.py  example input generator
third_party/rapidjson/        RapidJSON 1.1.0 (MIT)
```

## License

<!-- TODO: choose a license for this repository and add a LICENSE file. -->

RapidJSON (`third_party/rapidjson`) is distributed under the MIT license; see the notice at the top
of each of its headers.
