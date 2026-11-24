# Global cache-aware traversal on intelM2

All profiles start from the Original graph and use a connected frontier.
A profile considers at most the first 1, 24, or 12 queued cells in the
oldest active breadth shell. Breadth uses a peripheral root and FIFO.
Reuse and balanced start at a two-sweep graph-diameter midpoint and use
different normalized score weights. Original labels break ties and
determine the initial seed and component restarts.

The recent-row model resolves earlier reads to actual cache lines as
new labels become available. An unknown neighbour contributes an estimate
of one eighth of a line, with graph overlap used as a reuse proxy. The
candidate score includes separately normalized estimates for new and
reused lines, assigned-label span, and frontier compactness. The final
ranking uses exact GS read/write line footprints rather than the proxy.
It averages mean unique lines divided by K with equal weights at
K=32,128,512,1024,2048. Benchmark time was excluded from selection.

## MotorBike

All three profiles passed full topology validation and the combined and
per-array zero-tolerance coefficient guard. The actual serial packed and
blocked IndexKahn width-1024 schedules were built for each candidate.
Each passed the wave gate of at most 1.5 times the Original wave count
and a median of at least 512 rows. Balanced was eligible and ranked third,
so the top two were timed.

| Profile | Multi-scale score | Waves | Median rows | Coefficient/addressing transitions | Timed |
|---|---:|---:|---:|---:|---|
| Breadth | 0.3818 | 389 | 1,024 | 460,124 | Yes |
| Reuse | 0.4450 | 375 | 1,024 | 461,168 | Yes |
| Balanced | 0.4518 | 373 | 1,024 | 462,234 | No |

Fresh Original and RCM controls used copies of their established meshes.
All runs used the existing probe binary, CPUs 0–3, seven 64-sweep samples,
and eight warmup sweeps. The GS matrix and source were mapped from the
same physical problem. Mesh writing, schedule setup and validation ran
outside the numerical timings.

| Numbering | GS 1T | IH1W1024 1T | IndexKahn 4T | Psi lines K1024 | Reuse P90 | Waves | Coeff + address transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 2.326 ± 0.003 | 1.804 ± 0.004 | 1.410 ± 0.163 | 541.0 | 325 | 354 | 468,591 |
| Global RCM | 2.118 ± 0.004 | 1.761 ± 0.004 | 1.852 ± 0.017 | 318.4 | 7 | 394 | 458,958 |
| Traversal breadth | 2.417 ± 0.005 | 1.756 ± 0.005 | 1.845 ± 0.015 | 314.6 | 8 | 389 | 460,124 |
| Traversal reuse | 2.652 ± 0.005 | 1.783 ± 0.005 | 1.824 ± 0.019 | 359.8 | 20 | 375 | 461,168 |

Timings are medians ± sample standard deviation in ms/sweep. Psi counts
include outgoing reads and the write to the row label. Reuse is the exact
isolated-stream P90 stack distance in cache lines. Hardware cache hits
also depend on other arrays and on the cache implementation.

1. **Working set.** Breadth beat RCM at K=1024 by 1.2% and was within
   0.4% on the aggregate score (0.3818 versus 0.3803). Its graph
   bandwidth was 14,643 versus 20,844 for RCM, with mean row span
   17,978 versus 25,086. RCM retained slightly better P90 reuse.
2. **Parallel schedule shape.** All traversal profiles retained full
   median wave widths. The earlier cache-partition candidates produced
   2,033–8,462 waves, with median widths of 35–50. The new gate kept
   every candidate near the Original and RCM wave counts.
3. **Coefficients and addressing.** Each original LDU array passed at
   zero tolerance. Breadth reduced combined upper and upperAddr
   transitions by 1.8% relative to Original. Individual streams,
   including packed IH1 coefficients, indices, diagonal and row offsets,
   are saved separately.
4. **Original GS.** Breadth achieved 0.962× the Original speed and
   0.876× the RCM speed. Reuse achieved 0.877× and 0.799×, respectively.
   Neither improved one-thread GS performance.
5. **Four-thread performance.** Both timed variants were close to RCM
   (1.004× and 1.016× its speed). The differences are small relative
   to sample variability. Both were slower than the fresh Original
   median (0.764× and 0.773× of Original's speed). Breadth's serial
   IH1 run was faster than Original and within run variation of RCM.

Generation, full write, validation and diagnostics for all three profiles
took 6.55 s and about 263 MiB peak RSS. The experiment directory retains
full candidate permutations, topology reports, score terms, wave metrics,
LRU estimates, and the serial benchmark logs.

## Reproduction

Public CLI commands are in the README. The actual scripts and
comparison CSVs are under `experiments/cache-traversal-20261124T074650/` locally
and `~/cache-traversal-20261124T074650/experiments/` on intelM2.

## AHBody

The two selected MotorBike profiles were carried forward with the same
weights and gate thresholds. Both passed, with 2,807 and 2,810 waves
versus 2,787 for Original. The Original and existing renumbered reference
measurements are archived runs with the same probe and settings. The two
new variants were generated from AHBody Original and timed on its mapped
physical matrix. The existing renumbered reference used its own synthetic
matrix, so its close timings should be interpreted as a separate control.

| Numbering | GS 1T | IH1W1024 1T | IndexKahn 4T | Psi lines K1024 | Reuse P90 | Waves | Coeff + address transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 17.700 ± 0.038 | 18.970 ± 0.040 | 13.350 ± 0.580 | 454.8 | 29 | 2,787 | 3,836,489 |
| Existing renumbered | 14.510 ± 0.039 | 16.371 ± 0.046 | 13.051 ± 0.352 | 309.6 | 5 | 2,824 | 3,768,862 |
| Traversal breadth | 14.478 ± 0.042 | 16.075 ± 0.038 | 12.618 ± 0.629 | 314.3 | 6 | 2,807 | 3,764,199 |
| Traversal reuse | 17.742 ± 0.040 | 17.250 ± 0.063 | 13.217 ± 0.394 | 341.3 | 10 | 2,810 | 3,771,584 |

Breadth gave a 1.22× Original GS speedup over Original and matched the
existing renumbered GS reference within sample variation. Its serial
IH1 run was 1.18× Original. Its 4T median was 1.06× Original, with
substantial run variation. Breadth achieved a matrix bandwidth of 85,778
and reuse 200,644. Neither introduced the schedule fragmentation seen in
the previous explicit partitioning experiment.

### Outlier seam

Breadth's strict P99.99 threshold equals its maximum, so a mask of strictly
greater values is empty. To compare a fixed tail population, the supplementary
analysis selects the top 285 cells in every case, with ties broken by ascending
source cell ID. It also evaluates the same physical bounding box of the
existing renumbered mesh's largest P99.99 component.

| Numbering | P99.99 jump | Maximum jump | Largest component among top 285 | Top 285 in old seam box | Max jump in old seam box |
|---|---:|---:|---:|---:|---:|
| Original | 5,373,704 | 5,602,327 | 64 | 0 | 2,717,119 |
| Existing renumbered | 3,931,073 | 4,704,225 | 285 | 285 | 4,704,225 |
| Traversal breadth | 342,465 | 342,465 | 4 | 0 | 337,655 |
| Traversal reuse | 491,786 | 601,158 | 6 | 0 | 387,163 |

The box contains 828 cell centres in all four meshes. Neither traversal
reproduced the old concentration of extreme jumps there. Breadth reduced
the maximum inside the box approximately 14×. The spatial selection rule
and the original bounding box are recorded in the AHBody seam comparison.

The full 31-test suite passed on intelM2. Both AHBody variants passed the
physical mesh checks and the exact one-sweep comparisons with their
same-numbering GS references. Each AHBody run took about 20–23 s for
generation, complete mesh output and diagnostics with about 1.49 GiB peak RSS.
