# Original-start locality search on MotorBike

MotorBike has 352,253 cells and 1,053,964 internal faces. Both searches
started from Original labels, using seed 42, a 16-label relocation window,
two million candidate evaluations, and zero tolerance in the combined
LDU guard. Balanced used lambda 0.5. The controls were applied to the same
base mesh, and the benchmark used the same coefficients in physical space.

## Solver performance

Medians and sample standard deviations in ms/sweep on intelM2. The same
RelWithDebInfo solver probe as the previous campaign ran on CPUs 0–3, with
seven samples of 64 sweeps and eight warmup sweeps. Timing excluded mesh
generation, matrix and schedule setup, and correctness checks.

| Numbering | Original GS 1T | IH1W1024 1T | IndexKahn1024Blocked64OMP 4T |
|---|---:|---:|---:|
| Original | 2.325 ± 0.004 | 1.779 ± 0.004 | 1.475 ± 0.149 |
| RCM | 2.119 ± 0.004 | 1.743 ± 0.006 | 1.854 ± 0.016 |
| Locality total | 3.132 ± 0.038 | 1.896 ± 0.005 | 1.800 ± 0.023 |
| Locality balanced | 3.146 ± 0.015 | 1.894 ± 0.003 | 1.760 ± 0.161 |

| Locality objective | GS 1T speedup vs Original / RCM | IH1W1024 1T vs Original / RCM | IndexKahn 4T vs Original / RCM |
|---|---:|---:|---:|
| Total | 0.74× / 0.68× | 0.94× / 0.92× | 0.82× / 1.03× |
| Balanced | 0.74× / 0.67× | 0.94× / 0.92× | 0.84× / 1.05× |

Both locality variants were substantially slower on Original GS and about
6% slower on serial IH1 compared with Original numbering. The 4T medians
were below RCM's but above both the current and previous Original medians.
The current Original 4T range was 1.400–1.754 ms/sweep, compared with its
previous 1.731 ms/sweep median. The balanced 4T range was 1.432–1.797
ms/sweep. Treat the small 4T difference against RCM as a preliminary result.

## Locality and validation

| Numbering | LDU psi total | LDU psi P99 | LDU psi max | Upper + upperAddr line transitions |
|---|---:|---:|---:|---:|
| Original | 77,134,582,640 | 650,757 | 700,261 | 468,591 |
| RCM | 17,672,691,649 | 82,189 | 82,575 | 458,958 |
| Locality total | 74,420,505,323 | 645,786 | 700,225 | 468,465 |
| Locality balanced | 74,410,934,364 | 622,241 | 699,937 | 468,500 |

The total and balanced searches reduced jump magnitude by 3.52% and 3.53%.
Balanced cut RMS from 273,713 to 255,405 and P99 by 4.38%. LDU psi cache
line transitions increased from 1.61 million to 1.81 and 1.84 million.
The combined guard passed for both. Upper alone rose from 312,027 to 312,409
and 312,459, while upperAddr transitions fell. Packed IH1 coefficients,
column indices, diagonal, and offsets remain sequential in the serial
kernel. Its psi line transitions increased from 1.75 to 1.90 and 1.91 million.

| Objective | Core search time | Total generation and validation | Accepted swaps | Accepted moves | Guard rejections |
|---|---:|---:|---:|---:|---:|
| Total | 12.39 s | 13.60 s | 205,633 | 152,202 | 88,447 |
| Balanced | 11.84 s | 13.01 s | 202,677 | 120,140 | 81,483 |

Peak generation RSS was about 245 MiB per variant. Each candidate rescored
only the rows adjacent to moved cells and the affected row boundaries.
The guard recomputed changed 32-row blocks and combined 16 alignment
phases up the prefix tree. Full scans were used for periodic audits.

All four variants passed topology validation and the exact one-sweep
comparison of each IH1 implementation with its own GS reference. The
largest locality cell-centre difference was 3.42e-13. In Original labels,
the maximum converged GS difference was 1.11e-15 and the maximum absolute
residual was 1.96e-14. The full objective matched the tomography tool's
per-row values on all variants.

## Reproduction

The experiment directory on intelM2 is
`~/locality-20261123T231847/experiments/MotorBike/`. It contains
`generate.sh`, `measure.sh`, traced logs, meshes, permutations, checksums,
optimization histories, benchmark samples, and shared-normalization tomography.
Run the scripts from a fresh experiment directory with the same layout and
dependencies. The equivalent CLI options are in the repository README.
