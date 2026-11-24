# AHBody follow-up on intelM2

After MotorBike passed validation, the balanced search ran on the original
2,845,652-cell AHBody mesh. It used the same two million candidate budget,
seed 42, window 16, lambda 0.5, and zero tolerance on the combined LDU
guard. The core search took 16.63 s. Mesh generation and validation
together with the search took 25.71 s with peak RSS of approximately
1.48 GiB. It accepted 507,450 candidates and rejected 21,126 improving
candidates on the coefficient guard.

## Solver performance

Medians and sample standard deviations in ms/sweep. The run used the same
probe, CPUs 0–3, seven samples of 64 sweeps, and eight warmup sweeps as
MotorBike. Fresh schedules and correctness tests ran outside the timings.

| Numbering | Original GS 1T | IH1W1024 1T | IndexKahn1024Blocked64OMP 4T |
|---|---:|---:|---:|
| Original | 17.700 ± 0.038 | 18.970 ± 0.040 | 13.350 ± 0.580 |
| Locality balanced | 29.550 ± 0.047 | 20.622 ± 0.013 | 14.260 ± 0.710 |
| Existing renumbered | 14.510 ± 0.039 | 16.371 ± 0.046 | 13.051 ± 0.352 |

Original and locality used the same physical matrix and right-hand side.
The existing renumbered mesh was benchmarked as a separate input with
its own synthetic matrix. Its timings provide a reference for that input.
Each IH1 variant passed the one-sweep exactness check on its own
numbering. The original-to-locality converged GS difference was
8.88e-16 after remapping, and cell centres agreed within 4.58e-13.

## Outliers

| Numbering | LDU psi total | P99.99 | Maximum | Largest P99.99 component | Share of selected jump magnitude |
|---|---:|---:|---:|---:|---:|
| Original | 2,369,934,524,461 | 5,373,704 | 5,602,327 | 64 cells | 22.57% |
| Locality balanced | 2,323,686,939,761 | 5,331,732 | 5,602,324 | 24 cells | 8.33% |
| Existing renumbered | 408,261,545,580 | 3,931,073 | 4,704,225 | 285 cells | 100.00% |

All masks selected 285 cells at the lower-method P99.99 threshold. The
locality search fragmented the extreme tail compared with Original and
reduced total jump magnitude by 1.95%. Its maximum barely changed, and
the largest P99.9 component still accounted for 95.44% of selected jump
magnitude. The existing renumbering had much better overall locality, with
its P99.99 cells concentrated in one component. The two numberings with
smaller P99.99 components had higher maximum jumps. Both the jump scale and
component statistics matter when comparing these projections.

The combined LDU transition guard moved from 3,836,489 to 3,835,733, and
LDU psi cache line transitions increased from 13.12 to 13.73 million.
The bounded search did not meet the goal of improving GS performance.

The experiment directory on intelM2 is
`~/locality-20261123T231847/experiments/AHBody/`. It contains the
actual commands in `run.sh`, output logs, original-mesh references,
permutations, generated meshes, and the three sets of tomography images
rendered with shared projection bounds and brightness scales.
