# MotorBike bandwidth and working-set analysis on intelM2

The four existing meshes and their original timing records were reused.
The input topology checksums and edge-distance mean, P50, P90, P99 and
maximum match the previous `analyze_mesh_ordering.py` results. No mesh,
physical permutation, optimizer or solver kernel changed.

## What distinguishes RCM and Locality

| Metric | Original | RCM | Locality total | Locality balanced |
|---|---:|---:|---:|---:|
| Original GS, prior median ms/sweep | 2.325 | 2.119 | 3.132 | 3.146 |
| Matrix bandwidth, cell labels | 350,140 | 20,844 | 351,053 | 350,143 |
| Mean neighbour span, cell labels | 109,487 | 25,086 | 105,617 | 105,601 |
| P99 neighbour span, cell labels | 331,511 | 41,095 | 330,826 | 330,647 |
| GS psi unique lines, K=128 | 80.6 | 52.1 | 138.2 | 140.0 |
| GS psi unique lines, K=1024 | 538.5 | 315.0 | 905.1 | 914.6 |
| GS psi unique pages, K=1024 | 47.0 | 12.3 | 127.0 | 128.1 |
| GS psi reuse factor, K=128 | 5.93 | 7.53 | 3.83 | 3.76 |
| GS psi reuse factor, K=1024 | 7.61 | 9.88 | 5.11 | 5.04 |
| GS psi stack distance P90, distinct lines | 4,193 | 15 | 9,465 | 9,560 |
| GS bPrime stack distance P90, distinct lines | 325 | 7 | 7,204 | 7,268 |
| Upper mean unique lines, K=1024 | 384.1 | 383.9 | 384.1 | 384.2 |
| Upper P99 unique lines, K=1024 | 730 | 415 | 719 | 722 |
| upperAddr P99 unique lines, K=1024 | 366 | 208 | 360 | 361 |

1. **RCM sharply reduced graph bandwidth and spans.** Bandwidth fell by
   94%, mean row span by 77%, and P99 row span by 88%. Locality left global
   bandwidth and the P99 span close to Original. RCM improved the long
   edge-distance tail even though median edge distance rose from 49 to 4,177.
2. **RCM had a more compact working set.** At K=1024, actual GS psi reads
   occupied 42% fewer lines and 74% fewer pages than Original. The full
   incoming/outgoing adjacency trace showed the same ordering of variants.
3. **Locality reduced jump distance while spreading accesses across more lines.**
   Its GS psi footprint grew by 68–70% at K=1024, and the page footprint
   grew about 2.7×. Reuse distances also grew, especially for the indirect
   `bPrime` stream. The fraction of GS psi reads with stack distance
   below 512 lines, counting cold accesses in the denominator, was
   83.3%, 91.1%, 71.0%, and 70.7% in table order. These isolated stream
   estimates use fully associative LRU and do not model hardware hit rates.
4. **The coefficient guard omits important indirect accesses.** Upper and
   upperAddr each see 2,107,928 touches per sweep. Their mean footprints
   and reuse factors barely changed, and Locality slightly reduced their
   P99 footprint versus Original. RCM made the upper and addressing
   footprint tails substantially smaller. Locality's much poorer
   `psi` and `bPrime` reuse is outside the guard. Face permutations changed
   which physical face occupies each slot, while the kernel still walks
   its grouped face IDs forward twice. No separate lower array is read.

Window rows are natural cell labels. Each CSV contains all six window
sizes and the mean, P50, P90, P99 and maximum of each reported measure.
The adjacency trace reads incoming then outgoing neighbours. The actual
symmetric L0 kernel gathers only outgoing `psi`, and separately reads and
updates `bPrime`. Those streams have independent logical address spaces.

## Gated hardware counters

Seven independent 64-sweep batches per event group and variant ran
with the same binary, compiler flags, 1T execution, CPUs 0–3, and eight
warmup sweeps. Perf counters were enabled for the GS solve, after setup,
validation, warmup and resetting psi. Events use the pinned P-core PMU
in user mode. Groups contain at most four events. The raw CSV records
per-sample counts, values per sweep, and counter coverage. All 504
event records had 100% coverage. Every requested event could be opened.

| Median per sweep unless indicated | Original | RCM | Locality total | Locality balanced |
|---|---:|---:|---:|---:|
| Gated GS, ms/sweep (median ± sample SD) | 2.337 ± 0.004 | 2.133 ± 0.003 | 3.123 ± 0.011 | 3.180 ± 0.006 |
| Instructions, millions | 24.692 | 24.965 | 24.527 | 24.530 |
| Cycles, millions (median ± sample SD) | 9.608 ± 0.018 | 8.830 ± 0.016 | 13.050 ± 0.049 | 13.289 ± 0.022 |
| IPC | 2.570 | 2.827 | 1.879 | 1.846 |
| L1D load misses | 289,449 | 176,201 | 579,746 | 587,363 |
| Retired loads missing L3 | 579 | 507 | 1,412 | 1,473 |
| dTLB load walks completed | 1,148 | 673 | 1,873 | 1,910 |
| `memory_stalls.l1`, millions of cycles | 1.235 | 1.102 | 1.987 | 2.045 |
| `memory_stalls.l2`, millions of cycles | 0.256 | 0.129 | 0.572 | 0.603 |
| `memory_stalls.l3`, millions of cycles | 0.252 | 0.090 | 0.620 | 0.634 |
| `memory_stalls.mem`, millions of cycles | 0.613 | 0.590 | 0.725 | 0.731 |

5. **The counters are consistent with the timing ordering.** Retired
   instructions were within 1.2% of Original. Locality took about 36–38%
   more cycles and recorded roughly twice as many L1D load misses.
   RCM retired slightly more instructions but used 8% fewer cycles, with
   higher IPC and fewer L1D misses and dTLB walks. Generic cache misses
   were similar across variants. The generic LLC-load-miss alias
   returned zero everywhere, so the table uses the named retired L3
   miss event. The named L1, L2 and L3 stall counters also fell with RCM
   and rose with Locality. Full raw counts and per-event variability
   are saved in both PMU summary files.
   These four observations support a working-set and temporal-reuse
   explanation. They do not identify a single causal event or isolate
   the cost of a particular array.

## Reproduction

See the experiment directory at
`experiments/working-set-20261123T235307/MotorBike/` for the raw CSV and JSON,
comparison table, commands, and the log of all 18 passing tests on intelM2. The Intel host directory is
`~/working-set-20261123T235307/experiments/MotorBike/`.
