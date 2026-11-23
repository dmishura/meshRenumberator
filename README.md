# Mesh Renumbering Laboratory

Generate standalone OpenFOAM polyMesh variants and measure how cell numbering
affects Gauss–Seidel. The tool supports identity, deterministic Reverse
Cuthill–McKee (RCM), and import of a physical cell permutation from a previous
experiment. It uses the readers, schedule builders, and solver kernels from a
separate WavefrontGaussSeidel checkout.

See [MotorBike on intelM2](docs/motorbike-intelM2.md) for initial results.

## Requirements

- Python 3.10+ and NumPy. The analysis and tomography stages also use Pillow.
- A sibling WavefrontGaussSeidel checkout with the current mesh and locality
  helpers. Set `WFGS_ROOT` if it lives elsewhere.
- For benchmarks, a Linux CMake build of WavefrontGaussSeidel with the smoother
  harness, GAMG, and fwGAMG libraries and a compatible OpenMP compiler. Set
  `WFGS_BUILD` to override the default `build-cmake` directory.

## MotorBike workflow

Run these commands from this repository:

```bash
wfgs=${WFGS_ROOT:-../WavefrontGaussSeidel}
wbuild=${WFGS_BUILD:-$wfgs/build-cmake}
mesh="$wbuild/mesh-data/MTB_example_polyMesh"

python3 -m unittest discover -s tools/tests -v
bash tools/build_probe.sh
for algorithm in identity rcm; do
  experiment="experiments/motorbike/$algorithm"
  python3 tools/mesh_renumber.py renumber \
      --mesh "$mesh" --output "$experiment" --algorithm "$algorithm"
  python3 tools/mesh_renumber.py validate --experiment "$experiment"
  taskset -c 0,1,2,3 python3 tools/mesh_renumber.py benchmark \
      --experiment "$experiment" --threads 4 --widths 1024 2048
  python3 tools/mesh_renumber.py analyze --experiment "$experiment"
done
python3 tools/mesh_renumber.py compare \
    experiments/motorbike/identity/benchmark-* \
    experiments/motorbike/rcm/benchmark-*
```

Choose CPU IDs that correspond to distinct cores. Pass specific benchmark
directories to `compare` after rerunning a variant. To generate paired
projections with shared bounds and normalization, append
`--tomography PATH --centres-export "$wbuild/mesh_centres_export"`.
To report outlier components for individual variants, pass
`--centres-export "$wbuild/mesh_centres_export"` to `analyze`.

## Mesh and validation

Pass a serial polyMesh directory. This milestone supports little endian
32-bit compact faces, binary double precision points, ASCII or binary
owner/neighbour lists, simple uncoupled boundary patches, and the known
MotorBike metadata and mesh sets. Other metadata, nonempty zones, coupled
boundaries, and full case fields are outside the supported input format.

Each new experiment contains a polyMesh, configuration, topology report, and
`permutation.npz`. The array `old_to_new[oldCell]` gives the new cell ID and
`new_to_old[newCell]` gives the original ID. `new_face_to_old` maps each new
face to its source face and `face_flipped` records reversed orientation. Use
`--algorithm existing --initial-permutation FILE` to apply a saved physical
permutation defined against the same source mesh.

The benchmark adapter permutes the same symmetric LDU matrix and source
vector, checks inverse correspondence, and rebuilds each schedule. IH1 must
match Original GS after one sweep on the same numbering. The benchmark
reports the across-numbering one-sweep difference separately and checks
converged solutions against the same reference vector. The GAMG IH1W kernels
are serial. The separately reported IndexKahn Blocked64 kernels supply the
OpenMP timings. Repeated benchmark runs use new subdirectories for samples,
configuration, and validation and timing summaries.

The initial locality stage measures natural-order LDU psi and coefficient
access patterns. Local refinement, full packed-array diagnostics, and AHBody
experiments are later milestones.
