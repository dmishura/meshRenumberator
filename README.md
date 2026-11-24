# Mesh Renumbering Laboratory

Generate standalone OpenFOAM polyMesh variants and measure how cell numbering
affects Gauss–Seidel. The tool supports Original numbering, Reverse
Cuthill–McKee (RCM), and deterministic locality search from the original input
mesh. It uses the readers, schedule builders, and solver kernels from a
separate WavefrontGaussSeidel checkout.

See [MotorBike on intelM2](docs/motorbike-intelM2.md) for the initial baseline
and [MotorBike locality search](docs/locality-motorbike-intelM2.md) for the custom
numbering comparison. The [AHBody follow-up](docs/locality-ahbody-intelM2.md)
covers the larger mesh and outlier clusters.

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
for algorithm in original rcm; do
  experiment="experiments/motorbike/$algorithm"
  python3 tools/mesh_renumber.py renumber \
      --mesh "$mesh" --output "$experiment" --algorithm "$algorithm"
  python3 tools/mesh_renumber.py validate --experiment "$experiment"
  taskset -c 0,1,2,3 python3 tools/mesh_renumber.py benchmark \
      --experiment "$experiment" --threads 4 --widths 1024 2048
  python3 tools/mesh_renumber.py analyze --experiment "$experiment"
done
python3 tools/mesh_renumber.py compare \
    experiments/motorbike/original/benchmark-* \
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

## Locality search

```bash
bash tools/build_locality.sh
python3 tools/mesh_renumber.py renumber \
    --mesh "$mesh" --output-dir experiments/motorbike/locality_balanced \
    --algorithm locality --initial original --objective balanced \
    --balance-weight 0.5 --coeff-tolerance 0 --seed 42 \
    --candidates 2000000 --window 16
```

Use the source mesh as input. The search starts with its current labels
and rejects imported starting permutations. `--algorithm original` preserves
the mesh numbering. The previous `identity` spelling remains an alias for
existing scripts. `--output` and `--output-dir` are interchangeable.

The optimizer uses the incoming and outgoing neighbour reads in the written
mesh's face order. A row owns the transition from the preceding nonempty
row. Gaps of zero or one contribute zero. The total objective sums these
costs. The balanced objective adds `--balance-weight` times the RMS to the
mean. Locality requires a lexicographically sorted source internal-face list
so the initial and subsequent objectives follow the writer's LDU order.

The search samples all cells and proposes local swaps, bounded relocations,
swaps with graph neighbours, and random swaps. Only moved cells, their
neighbours, and affected row boundaries need new costs. It checks improving
candidates against the Original cache-line transition budget for the
symmetric L0 GS kernel's `upper` and `upperAddr` streams. The metric assumes
64-byte aligned arrays and accounts for both passes over each owner's face
range. The default tolerance is zero. That kernel reads no separate
`lower` array. Its `diag` and owner starts are sequential with invariant
transition counts. A blocked prefix tree evaluates the guard with the face
alignment of the candidate numbering. Full recomputation audits progress
every `--verify-every` evaluations and at the end. `--patience` counts
consecutive evaluations without improvement, and zero disables early stopping.

The optimization report records the settings, objective history, independent
mean and RMS, coefficient transitions, acceptance counts and elapsed time.
Benchmarks still use the existing solver probe and regenerate each schedule.

To include packed IH1 array diagnostics, build and pass the separate layout
probe. This inspection runs outside the benchmark.

```bash
bash tools/build_probe.sh experiments/build/gs_layout tools/gs_layout.C
python3 tools/mesh_renumber.py analyze \
    --experiment experiments/motorbike/locality_balanced \
    --centres-export "$wbuild/mesh_centres_export" \
    --layout-probe experiments/build/gs_layout
LOCALITY_OPTIMIZER="$PWD/experiments/build/locality_optimizer" \
    python3 -m unittest discover -s tools/tests -v
```

Pass all benchmark run directories to `compare`, starting with Original.
With `--tomography`, all variants use the same projection bounds and brightness
scales for jump magnitudes and nonsequential counts. The percentile masks use
the individual mesh thresholds.

## Bandwidth and working-set diagnostics

See the [intelM2 working-set report](docs/working-set-motorbike-intelM2.md)
for the MotorBike comparison and hardware counters.

```bash
bash tools/build_working_set.sh
python3 tools/mesh_renumber.py analyze-locality \
    --mesh experiments/motorbike/original/polyMesh \
    --face-permutation experiments/motorbike/original/permutation.npz \
    --window-sizes 8,32,128,512,1024,4096 --reuse-distance \
    --output-dir experiments/motorbike-working-set/original
```

Graph bandwidth and neighbour spans use integer cell labels. The command
reports full overlapping windows starting at row zero. The default stride is
one quarter of the window size, with a minimum of one. Use `--stride` to
override it. An empty read window has zero touches, footprint, density and
reuse factor. Footprints use 64-byte lines, 4 KiB pages and a separate aligned
address space for each array.

The reported traces distinguish the incoming/outgoing topology stream,
outgoing-only GS `psi` gathers, and logical GS `bPrime` reads and updates.
The coefficient CSV models the symmetric L0 kernel's two forward accesses to
each row's consecutive face IDs, once for the gather and once for updating
`bPrime`. It reports the upper coefficients and neighbour indices separately.
This kernel has no separate lower stream. Source copying and compiler-level
reloads are outside the logical trace. The optional stack-distance analysis
uses exact distinct-line distance over each stream starting from cold.
`--cache-kib` controls the capacity estimates. These are fully associative
LRU-style estimates for isolated arrays.

On intelM2, collect controlled GS counters for the four existing experiments:

```bash
bash tools/build_probe.sh experiments/build/gs_pmu
python3 tools/measure_gs_pmu.py experiments/motorbike experiments/build/gs_pmu \
    --output experiments/motorbike-working-set/pmu --cpus 0,1,2,3 \
    --sweeps 64 --samples 7
python3 tools/compare_working_set.py experiments/motorbike-working-set \
    experiments/motorbike
```

The PMU runner discovers which P-core events are supported and gates counting
around each GS sweep batch. Raw counts and coverage are saved per sample.
Samples below 99% coverage are excluded from the reported medians.
Use `--event-group` to select up to four events per group. The MotorBike
campaign measured the four `memory_stalls.*` events as a separate group
with the same run settings.

## Cache partitioning

See the [intelM2 cache-partition report](docs/cache-partition-intelM2.md)
for the MotorBike and AHBody results.

```bash
bash tools/build_cache_partition.sh
python3 tools/mesh_renumber.py renumber \
    --algorithm cache-partition --initial original --mesh "$mesh" \
    --output-dir experiments/motorbike/cache_rcm_16 \
    --partition-budget-kib 16 --partition-local-order rcm \
    --partition-coeff-tolerance 0.03 --partition-lru
```

The backend grows connected groups from the Original graph. It scores frontier
cells by internal adjacency, neighbour overlap, and boundary growth. The
budget estimates eight bytes per distinct cell in the group plus its read
neighbourhood, four bytes per directed graph neighbour entry, and four bytes
per row offset, including the terminating offset. A singleton can exceed the
budget and is reported. Actual aligned psi cache lines are measured after
numbering. Actual upper, upperAddr, diagonal, and row offset footprints are
reported separately. The budget is a graph heuristic and does not describe a
hardware cache's available capacity.

Use `--partition-local-order traversal` for connected growth order or `rcm`
for RCM of the induced graph of each group. Both families start from the same
Original graph. The coarse graph is weighted by shared mesh faces. Its
breadth and depth traversals consider heavier neighbours first. `auto`
tries both local orders or both coarse orders, up to four candidates per
budget. The score averages mean unique lines per row equally across
K=32,128,512,1024,2048, using the natural GS psi reads and row writes. The
coefficient guard limits the combined transition count and each original LDU
array independently. Zero tolerance is the default. The reported campaign
explicitly used 3%. Candidate ranking uses the working-set score alone.

`partition.json` records estimates, measured footprints, scores, coefficient
checks, seam statistics and runtime. `partitions.npz` includes group membership
before and after numbering, group offsets, and weighted coarse edges. The
optional LRU analysis assumes fully associative lines, write allocation and
one sweep starting cold. It separates cold misses from capacity misses.

Use the existing `benchmark`, `analyze` and `analyze-locality` commands to
check solver performance, packed IH1, and the independent GS access streams.

## Cache-aware global traversal

See the [intelM2 global traversal report](docs/cache-traversal-intelM2.md)
for the MotorBike schedules, timings, and AHBody seam analysis.

```bash
WFGS_ROOT="$wfgs" WFGS_BUILD="$wbuild" bash tools/build_cache_traversal.sh
python3 tools/mesh_renumber.py renumber \
    --algorithm cache-traversal --initial original --mesh "$mesh" \
    --output-dir experiments/motorbike/cache-traversal \
    --max-wave-ratio 1.5 --min-median-wave-width 512
```

The continuous traversal completes one graph component at a time.
It selects from a bounded number of queued cells in the oldest active
breadth shell. The first seed is `--traversal-seed` modulo the cell count.
Later components start their seed search at the lowest unassigned ID.
`peripheral` chooses a farthest cell after one graph sweep. `centre`
chooses the midpoint of an approximate diameter after two sweeps.

The default candidate set uses breadth (peripheral root, FIFO, weights
0/0/0/1), reuse (centre root, scan 24, weights 1/0.85/0.40/0.40) and
balanced (centre root, scan 12, weights 0.65/0.50/0.35/1.20).
Use `--traversal-profile` to run just one or override the root, frontier
scan, and `--traversal-weights NEW,REUSE,SPAN,FRONT`. The 128-row
recent model can be resized with `--traversal-recent-rows`.

- New and reused line estimates are divided by eight and capped at one.
  Known labels use their actual line IDs. Each distinct future neighbour
  contributes one eighth of a line to a graph-only estimate.
- The span term is the distance to the earliest assigned neighbour divided
  by the recent-row count, capped at one.
- The front term combines same-shell predecessor depth (55%), queue
  position (25%), newly exposed neighbour fraction (15%), and lack of
  already assigned contacts (5%). Each component is normalized to [0,1].

Candidates use the existing mesh writer and the actual packed and blocked
IndexKahn schedule builders. The wave cap and median floor apply to both
schedules. The effective floor is capped at Original's median for small
graphs. Zero disables either threshold. Per-array and combined coefficient
transitions use `--traversal-coeff-tolerance` (default zero).
The benchmark command rejects failed candidates and mismatched widths.

The top eligible ordering is published at the output directory. All
candidate meshes and rejection reasons are under `candidates/`;
`traversal.json` records the ranking and Original baseline. If all
candidates fail, the reports are still published and the command returns
an error. Ranking uses the five-scale exact GS psi read/write footprint.
The one-sweep LRU model assumes write allocation and starts cold.
