# Cache partitioning on intelM2

The experiment tried graph growth order and partition-local RCM at 16, 64 and
256 KiB. Each family and budget evaluated weighted coarse breadth and depth
traversal. Selection used equal weights for mean GS psi read/write unique
lines divided by K at 32, 128, 512, 1024 and 2048 rows. No benchmark timings
were used to choose the weights, orderings or AHBody budgets.

The guard checks upper, upperAddr, diagonal, row offsets, lower (unused by
this symmetric kernel), and the original upper-plus-upperAddr sum
independently. The first 64 KiB traversal pilot was rejected at zero
tolerance because upper and upperAddr transitions rose about 2%. The timed
campaign used a declared 3% tolerance on each metric. Zero remains the
CLI default. Estimates use aligned logical addresses and do not establish
cache residency.

## MotorBike

The benchmark was the established probe binary on CPUs 0–3. Each variant
used seven samples of 64 sweeps, eight warmup sweeps and the same compiler,
solver, thread counts and schedule width. References are the existing
measurements from the identical binary and settings. All candidates were
validated against Original after remapping. Converted topology and face
orientation passed checks, and each regenerated IH1 schedule matched the
same-numbering GS reference after one sweep.

| Numbering | GS 1T, ms/sweep | IH1W1024 1T, ms/sweep | IndexKahn 4T, ms/sweep | Psi read/write lines K1024 | Reuse P90, lines | Upper + upperAddr transitions |
|---|---:|---:|---:|---:|---:|---:|
| Original | 2.325 ± 0.004 | 1.779 ± 0.004 | 1.475 ± 0.149 | 541.0 | 325 | 468,591 |
| Global RCM | 2.119 ± 0.004 | 1.743 ± 0.006 | 1.854 ± 0.016 | 318.4 | 7 | 458,958 |
| Locality total | 3.132 ± 0.038 | 1.896 ± 0.005 | 1.800 ± 0.023 | 908.2 | 7,205 | 468,465 |
| Locality balanced | 3.146 ± 0.015 | 1.894 ± 0.003 | 1.760 ± 0.161 | 917.0 | 7,268 | 468,500 |
| Traversal 16 KiB | 2.745 ± 0.016 | 1.844 ± 0.003 | 18.959 ± 0.152 | 362.1 | 54 | 476,451 |
| Traversal 64 KiB | 2.688 ± 0.008 | 1.818 ± 0.004 | 22.586 ± 0.212 | 421.7 | 96 | 478,062 |
| Traversal 256 KiB | 2.630 ± 0.005 | 1.832 ± 0.003 | 18.616 ± 0.152 | 481.4 | 186 | 478,280 |
| Local RCM 16 KiB | 2.857 ± 0.003 | 1.923 ± 0.003 | 9.071 ± 0.088 | 389.4 | 67 | 469,021 |
| Local RCM 64 KiB | 2.799 ± 0.004 | 1.912 ± 0.030 | 10.146 ± 0.083 | 452.7 | 112 | 468,943 |
| Local RCM 256 KiB | 2.786 ± 0.014 | 1.954 ± 0.005 | 6.223 ± 0.069 | 471.6 | 187 | 469,712 |

1. **Working set versus global RCM.** No candidate beat RCM on the
   multi-scale score or K=1024. The best candidate, traversal 16 KiB, used
   33% fewer lines than Original but 14% more than RCM. Its multi-scale
   score was 0.510, compared with 0.600 for Original and 0.380 for RCM.
   Its matrix bandwidth remained 352,235 labels versus 20,844 for RCM.
2. **Temporal reuse.** All partition candidates improved exact stack
   distance P90 versus Original, but RCM was still best. Traversal 16
   KiB had 44,032 cold and 43,116 capacity misses in the 64 KiB,
   fully associative, write-allocate psi model. These are model counts.
3. **Coefficient and addressing streams.** The guard passed per array
   at 3%. The combined score rose by under 0.25% for local RCM and by
   1.68–2.07% for traversal. Independent upper, upperAddr, diagonal,
   row-start, psi and bPrime window results are saved. Serial packed
   IH1 has separate line-transition diagnostics for its psi accesses,
   coefficients, indices, diagonal and row offsets.
4. **Solver performance.** None improved on Original GS. The fastest GS
   variant was 13% slower than Original. Regenerated IndexKahn
   schedules explain the risk of optimizing only the serial psi footprint:
   they had 2,033–8,462 waves with median widths of 35–50 rows.
   Original had 354 waves with median width 1,024, and RCM had 394
   with median width 1,024. This difference is consistent with higher
   synchronization cost in the partitioned cases.
5. **Seams.** The largest MotorBike P99.99 components had 1–6 cells.
   Every selected P99.99 outlier touched a crossing edge, compared with
   92.7–95.7% of all cells. The tail is widespread at the boundaries
   even though the individual components were small. Max jump values
   remained near 668k–705k, compared with about 83k for global RCM.

| Budget | Groups | Median / max group cells | Crossing edges | Estimated max bytes | Actual max psi bytes, traversal / local RCM |
|---|---:|---:|---:|---:|---:|
| 16 KiB | 22,796 | 1 / 447 | 628,644 (59.6%) | 16,384 | 21,248 / 25,600 |
| 64 KiB | 23,098 | 1 / 1,772 | 613,147 (58.2%) | 65,536 | 70,592 / 85,568 |
| 256 KiB | 22,650 | 1 / 6,663 | 593,484 (56.3%) | 262,144 | 214,976 / 242,112 |

Many cells were stranded in singleton groups. Coarse bandwidth was
19,703–23,096 groups. Packing an individual group's halo gives no bound
on the separation between its group and its neighbours. The estimates
also missed cache-line fragmentation in some partitions. Complete
renumbering, including writing and validating the mesh, took 3.00–3.12 s
and approximately 256–261 MiB peak RSS. Graph growth took about 0.18 s.
Schedule construction was outside the sweep measurements.

## Artifacts

The experiments are under `experiments/cache-partition-20261124T072210/` locally and
`~/cache-partition-20261124T072210/experiments/` on intelM2. The MotorBike
directory includes `pilot.sh`, `pilot-3pct.sh`, `campaign.sh`,
`comparison.csv`, the generated meshes and permutations, timing
samples, both scored coarse orders for each configuration, and exact
cache-line and boundary reports.

## AHBody follow-up

The best working-set budget in each MotorBike ordering family was 16 KiB,
so both families used that budget on AHBody Original. Group growth took
1.6 s. The full renumber, write and validation took 25.6–27.0 s with
1.57–1.60 GiB peak RSS. The run produced 221,753 groups with median size
one and 62.8% crossing edges. The higher group count and singleton tail
persisted on the larger mesh.

| Numbering | GS 1T, ms/sweep | IH1W1024 1T, ms/sweep | IndexKahn 4T, ms/sweep | Psi read/write lines K1024 | Reuse P90, lines |
|---|---:|---:|---:|---:|---:|
| Original | 17.700 ± 0.038 | 18.970 ± 0.040 | 13.350 ± 0.580 | 454.8 | 29 |
| Existing renumbered | 14.510 ± 0.039 | 16.371 ± 0.046 | 13.051 ± 0.352 | 309.6 | 5 |
| Traversal 16 KiB | 29.996 ± 0.051 | 36.116 ± 0.073 | 53.973 ± 0.316 | 398.5 | 78 |
| Local RCM 16 KiB | 29.767 ± 0.075 | 33.836 ± 0.045 | 26.100 ± 0.947 | 425.8 | 100 |

The two generated variants used the same physical LDU matrix and source
as Original. The existing renumbered input was previously measured with
its own synthetic matrix and is a separate reference. Both new
numberings preserved the converged solution within 1.12e-15 and passed
the regenerated one-sweep IH1 exactness checks.

Both partitions cut the mean 1,024-row footprint compared with Original,
while P90 stack distance grew. Their P99.99 jump thresholds were about
5.683 million, compared with 5.374 million on Original and 3.931 million
on the existing renumbered mesh. The largest selected components were
36 and 23 cells, versus 64 on Original and 285 on the existing numbering.
All selected cells touched partition seams. Most cells overall did too
(97.8%). The smaller components therefore accompanied a more widespread,
higher jump tail rather than a reduction in extreme read jumps.
