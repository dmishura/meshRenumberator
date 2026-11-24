# MotorBike on intelM2

Original and deterministic RCM were evaluated on the same 352,253 cell mesh with
1,053,964 internal faces. Both runs used the same probe binary, an Intel Core
Ultra 7 356H, CPU affinity 0–3, eight warmup sweeps, and seven samples of 64
sweeps. Setup and numerical validation ran outside the timed intervals. The probe
linked the existing RelWithDebInfo libraries from the intelM2
WavefrontGaussSeidel worktree at dc067cd with local changes. Locality analysis
used the current sibling Python tools, including its updated visualizer.

| Kernel | Original, ms/sweep | RCM, ms/sweep | Original / RCM | IH1 vs GS, original / RCM |
|---|---:|---:|---:|---:|
| Original GS 1T | 2.323 ± 0.004 | 2.120 ± 0.004 | 1.10× | — |
| IH1W1024 1T | 1.780 ± 0.005 | 1.732 ± 0.004 | 1.03× | 1.31× / 1.22× |
| IndexKahn1024Blocked64OMP 1T | 1.942 ± 0.002 | 1.928 ± 0.004 | 1.01× | 1.20× / 1.10× |
| IndexKahn1024Blocked64OMP 4T | 1.731 ± 0.011 | 1.861 ± 0.015 | 0.93× | 1.34× / 1.14× |
| IH1W2048 1T | 1.799 ± 0.002 | 1.757 ± 0.042 | 1.02× | 1.29× / 1.21× |
| IndexKahn2048Blocked64OMP 1T | 1.962 ± 0.005 | 1.970 ± 0.007 | 1.00× | 1.18× / 1.08× |
| IndexKahn2048Blocked64OMP 4T | 1.108 ± 0.025 | 1.249 ± 0.005 | 0.89× | 2.10× / 1.70× |

Values are medians with sample standard deviations. Each speedup compares
identical kernels and thread counts across numberings, or compares with Original
GS 1T on the same numbering in the last column.

| LDU locality metric | Original | RCM |
|---|---:|---:|
| Total psi read jump magnitude | 77,134,582,640 | 17,672,691,649 |
| P99 per-cell jump magnitude | 650,757 | 82,189 |
| P99.9 per-cell jump magnitude | 680,711 | 82,473 |
| P99.99 per-cell jump magnitude | 693,432 | 82,559 |
| Maximum per-cell jump magnitude | 700,261 | 82,575 |
| Nonsequential psi transitions | 82.41% | 85.95% |
| Psi cache line transitions | 1,608,032 | 1,634,164 |
| Psi page transitions | 1,101,376 | 1,120,811 |
| Mean coefficient cold line fraction, 32-row window | 22.21% | 22.11% |

RCM reduced total psi jump magnitude by 77%. Original GS improved by 1.10×
while both four-thread IndexKahn variants slowed down. The LDU transition
counts increased, and the coefficient metric barely changed. The result
supports evaluating timings alongside locality on each kernel.

Both meshes passed inverse topology validation. The maximum cell centre
difference after remapping was 4.96e-13. Each IH1 variant matched Original GS
after one sweep on its own numbering. Original GS differed by 0.417 after one
sweep across numberings, as the update order changed. After 192 sweeps the
maximum mapped difference between the two GS solutions was 8.88e-16 and both
had maximum absolute residual 1.24e-14 for the same permuted system.

The experiment on intelM2 is in
`~/meshRenumberator-initial.KkEaOa8i/experiments/motorbike`. It includes the
generated meshes, topology reports, timing samples, locality and cluster
statistics, and paired psi-read projections with shared normalization.
