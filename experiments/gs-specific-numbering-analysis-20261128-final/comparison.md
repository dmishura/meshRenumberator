# GS-specific mesh numbering analysis

Offline analysis of the 29 archived mesh/numbering combinations. No meshes, kernels, numbering algorithms, or benchmark runs were changed or created.

## Definitions and execution model

Owner degree is the actual count of internal LDU faces in each owner row. Neighbour degree separately counts the appearances of a cell in `upperAddr`; neither is replaced by undirected graph degree. All W=2/4/8/16 chunk calculations use `sum(ceil(owner_degree/W))`, with degree-zero rows contributing zero chunks.

M6.4d's ordinary path issues four scalar index loads and four indirect `psi` loads per 4-slot chunk; inactive lanes are masked only after those loads and products, and coefficients are loaded unmasked. The last physical face suffix is diverted to the safe M6 terminal helper: masked index loads supply index zero in inactive lanes, so those lanes still read `psi[0]`; coefficient masked loads suppress inactive elements. M8 does eight scalar index/`psi` reads per 8-slot main chunk, with masked coefficient operands, then uses the same safe 4-wide M6 terminal helper for its final face suffix. Exact execution counts and active-only chunk geometry are stored separately in `analysis.json`.

Cache lines assume `psi` is 8-byte FP64 and 64-byte aligned; 4 KiB and 2 MiB page IDs use the same base. Alignment sensitivity is sampled at base offsets 0, 8, …, 56 bytes for active chunk line counts. Reuse distances are exact distinct-line stack distances from the repository Fenwick backend; cold accesses are kept separate. These are logical locality measures, not hardware miss predictions.

Chunk instruction estimates include only the given hot gather chunk instruction counts (24 for W4; 42/43 for W8). They exclude row/cell loop overhead, gather/scatter outside those chunks, and setup. The ‘inactive-slot equivalent’ prorates chunk-body instructions over unused lanes; it is not a literal count of instructions attributable to inactive lanes.

## Summary: all 29 combinations

| Mesh | Numbering | Cells | Faces | Chunks W4 | Util W4 | Chunks W8 | Util W8 | N4/N8 | Est instr/face W4 | Est instr/face W8 (42) | W4 lines/chunk | W4 XMM pairs same line | M6.4d ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MotorBike | original | 352,253 | 1,053,964 | 396,249 | 0.665 | 333,746 | 0.395 | 1.187 | 9.023 | 13.300 | 2.054 | 42.3% | 1.638 |
| MotorBike | rcm | 352,253 | 1,053,964 | 373,443 | 0.706 | 352,278 | 0.374 | 1.060 | 8.504 | 14.038 | 2.317 | 28.2% | 1.393 |
| MotorBike | C | 352,253 | 1,053,964 | 381,005 | 0.692 | 351,813 | 0.374 | 1.083 | 8.676 | 14.020 | 2.246 | 31.2% | 1.516 |
| MotorBike | D | 352,253 | 1,053,964 | 374,347 | 0.704 | 352,287 | 0.374 | 1.063 | 8.524 | 14.038 | 2.314 | 27.4% | 1.405 |
| AHBody | original | 2,845,652 | 8,592,613 | 3,217,468 | 0.668 | 2,661,847 | 0.404 | 1.209 | 8.987 | 13.011 | 2.052 | 43.4% | 14.909 |
| AHBody | existing_renumbered | 2,845,652 | 8,592,613 | 3,107,769 | 0.691 | 2,841,890 | 0.378 | 1.094 | 8.680 | 13.891 | 2.346 | 32.0% | 11.122 |
| AHBody | rcm | 2,845,652 | 8,592,613 | 3,152,799 | 0.681 | 2,845,652 | 0.377 | 1.108 | 8.806 | 13.909 | 2.319 | 30.2% | 12.740 |
| AHBody | C | 2,845,652 | 8,592,613 | 3,101,640 | 0.693 | 2,842,100 | 0.378 | 1.091 | 8.663 | 13.892 | 2.330 | 30.4% | 11.661 |
| AHBody | D | 2,845,652 | 8,592,613 | 3,127,565 | 0.687 | 2,845,655 | 0.377 | 1.099 | 8.736 | 13.909 | 2.296 | 29.2% | 11.648 |
| Cavity3d | original | 8,000,000 | 23,880,000 | 8,000,004 | 0.746 | 7,999,999 | 0.373 | 1.000 | 8.040 | 14.070 | 2.985 | 0.0% | 57.579 |
| Cavity3d | existing_renumbered | 8,000,000 | 23,880,000 | 8,000,138 | 0.746 | 7,999,913 | 0.373 | 1.000 | 8.040 | 14.070 | 2.124 | 1.4% | 31.095 |
| Cavity3d | rcm | 8,000,000 | 23,880,000 | 7,999,999 | 0.746 | 7,999,999 | 0.373 | 1.000 | 8.040 | 14.070 | 2.118 | 0.5% | 31.528 |
| Cavity3d | C | 8,000,000 | 23,880,000 | 7,999,999 | 0.746 | 7,999,999 | 0.373 | 1.000 | 8.040 | 14.070 | 2.118 | 0.5% | 31.551 |
| Cavity3d | D | 8,000,000 | 23,880,000 | 7,999,999 | 0.746 | 7,999,999 | 0.373 | 1.000 | 8.040 | 14.070 | 2.118 | 0.5% | 31.812 |
| ConicalDiffuser | original | 3,095,547 | 9,256,046 | 3,096,728 | 0.747 | 3,095,543 | 0.374 | 1.000 | 8.030 | 14.046 | 2.989 | 0.0% | 19.847 |
| ConicalDiffuser | existing_renumbered | 3,095,547 | 9,256,046 | 3,096,181 | 0.747 | 3,095,541 | 0.374 | 1.000 | 8.028 | 14.046 | 2.128 | 5.0% | 9.611 |
| ConicalDiffuser | rcm | 3,095,547 | 9,256,046 | 3,095,936 | 0.747 | 3,095,546 | 0.374 | 1.000 | 8.027 | 14.046 | 2.129 | 2.0% | 9.477 |
| ConicalDiffuser | C | 3,095,547 | 9,256,046 | 3,095,538 | 0.748 | 3,095,538 | 0.374 | 1.000 | 8.026 | 14.046 | 2.129 | 2.3% | 9.198 |
| ConicalDiffuser | D | 3,095,547 | 9,256,046 | 3,099,056 | 0.747 | 3,095,546 | 0.374 | 1.001 | 8.036 | 14.046 | 2.128 | 2.5% | 9.445 |
| MTBHPC | original | 8,613,999 | 25,952,973 | 9,725,301 | 0.667 | 8,173,354 | 0.397 | 1.190 | 8.993 | 13.227 | 2.059 | 41.9% | 73.049 |
| MTBHPC | existing_renumbered | 8,613,999 | 25,952,973 | 9,390,508 | 0.691 | 8,634,993 | 0.376 | 1.087 | 8.684 | 13.974 | 2.248 | 32.2% | 45.335 |
| MTBHPC | rcm | 8,613,999 | 25,952,973 | 9,166,717 | 0.708 | 8,614,429 | 0.377 | 1.064 | 8.477 | 13.941 | 2.346 | 27.4% | 42.002 |
| MTBHPC | C | 8,613,999 | 25,952,973 | 9,388,495 | 0.691 | 8,634,144 | 0.376 | 1.087 | 8.682 | 13.973 | 2.251 | 31.9% | 44.564 |
| MTBHPC | D | 8,613,999 | 25,952,973 | 9,166,695 | 0.708 | 8,614,486 | 0.377 | 1.064 | 8.477 | 13.941 | 2.348 | 27.2% | 42.025 |
| WD_DamBreak | original | 9,376,387 | 27,975,312 | 9,376,544 | 0.746 | 9,376,246 | 0.373 | 1.000 | 8.044 | 14.077 | 2.983 | 0.0% | 69.111 |
| WD_DamBreak | existing_renumbered | 9,376,387 | 27,975,312 | 9,376,566 | 0.746 | 9,376,356 | 0.373 | 1.000 | 8.044 | 14.077 | 2.120 | 0.9% | 35.859 |
| WD_DamBreak | rcm | 9,376,387 | 27,975,312 | 9,376,386 | 0.746 | 9,376,386 | 0.373 | 1.000 | 8.044 | 14.077 | 2.120 | 0.3% | 35.288 |
| WD_DamBreak | C | 9,376,387 | 27,975,312 | 9,376,386 | 0.746 | 9,376,386 | 0.373 | 1.000 | 8.044 | 14.077 | 2.116 | 0.6% | 36.401 |
| WD_DamBreak | D | 9,376,387 | 27,975,312 | 9,376,386 | 0.746 | 9,376,386 | 0.373 | 1.000 | 8.044 | 14.077 | 2.116 | 0.6% | 36.188 |

## Performance files and compatibility

M6.4d timing values in the main table come from one matched amdR16 two-round campaign (Scalar/M6/M6.4d, 29 cases, two samples per implementation). AVX-512 A/B timings are included only for the archived matched AVX2/A/B campaigns: other meshes, AHBody, and the separately corrected MotorBike C/D run. The 8-wide M8 timings exist only for MotorBike C/D and ConicalDiffuser C/D from the initial M8-wide campaign. Do not combine those campaign medians as if they were one matched run. Exact sources are retained per case in `analysis.json`.

### M8 versus M6.4d where an M8 timing exists

| Mesh | Numbering | M6.4d ms | M8.4x128 ms | M8.2x256 ms | M8.1x512 ms | N4/N8 | 42-instr model ratio vs W4 |
|---|---|---:|---:|---:|---:|---:|---:|
| MotorBike | C | 1.507 | 1.800 | 1.762 | 1.753 | 1.083 | 0.528 |
| MotorBike | D | 1.405 | 1.719 | 1.662 | 1.661 | 1.063 | 0.538 |
| ConicalDiffuser | C | 9.205 | 12.602 | 11.023 | 11.253 | 1.000 | 0.571 |
| ConicalDiffuser | D | 9.505 | 12.790 | 11.275 | 11.526 | 1.001 | 0.571 |

The rightmost ratio is estimated gather chunk-body instructions W8/W4 = `(42/24)*(N8/N4)`. It is below 1 only when N4/N8 exceeds 1.75. The 43-instruction ZMM kernel's threshold is 43/24 = 1.792. In these four cases, compare the estimated ratio with observed timing; it is not a complete-kernel cost model.

## Within-chunk and stream locality

See `chunk_locality.csv` for mean/median/P95/P99/max active-neighbour cache-line and page counts, distance distributions, same-line consecutive accesses, same-line XMM pairs, and histograms of chunks with 1–W cache lines. `analysis.json` additionally contains exact per-window distributions for 4, 8, 16, 32, 64, and 128 consecutive chunks, for both active and architecturally issued `psi` reads, plus exact line/page stack-distance summaries. `reuse_distance.csv` separates active-only line/page distances from executed line/page distances (including inactive slots). `scatter_locality.csv` reports destination-index locality and exact scatter line reuse distances independently.

## Owner-degree and chunk details

`degree_statistics.csv` contains complete owner and neighbour histograms, including zero-degree cells, mean/median/std/max/P95/P99, requested degree bands, and owner rows needing more than one 4-wide chunk. `analysis.json` carries utilization, active/inactive slots, full/partial chunk counts, and per-chunk active-element histograms for W=2/4/8/16.

## Structural lower bounds and adjacent swaps

For each width, `analysis.json` reports the unconstrained lower bound `ceil(F/W)`, actual chunk count, relative excess, and potential chunk-body instructions saved if that lower bound were attained. `adjacent_swap_candidates.csv` counts unique connected cell pairs and exact single-swap changes in W4 chunk count. These changes are local counterfactuals; improvements from separate swaps cannot be summed because swaps interact.

## Correlations

Only the matched M6.4d campaign is used for the 29-case primary performance correlations. Cross-mesh comparisons use time/cell and time/face; within-mesh statistics are exploratory because each mesh has only four or five numberings. AVX-512 A/B and M8 correlations are exported separately per matched archived campaign in `correlations.csv`, without pooling across binaries. The 8-wide M8 campaign has four MotorBike/ConicalDiffuser C/D points only, so those correlations are descriptive.

| Metric versus M6.4d ms/face | n | Pearson r | Spearman rho |
|---|---:|---:|---:|
| W4 chunks | 29 | 0.337 | 0.302 |
| W4 utilization | 29 | -0.277 | -0.557 |
| estimated W4 instructions/face | 29 | 0.283 | 0.557 |
| W4 active unique lines/chunk | 29 | 0.591 | 0.331 |
| W4 active 4KiB pages/chunk | 29 | 0.719 | 0.775 |
| W4 line reuse distance P90 | 29 | 0.556 | 0.611 |
| active psi page reuse distance P90 | 29 | 0.715 | 0.809 |
| scatter line reuse distance P90 | 29 | 0.556 | 0.611 |
| matrix bandwidth | 29 | 0.679 | 0.613 |
| mean row neighbour span | 29 | 0.640 | 0.749 |
| P99 row neighbour span | 29 | 0.763 | 0.802 |

### Within-mesh correlations (exploratory)

| Mesh | Metric | n | Pearson r (ms/face) | Spearman rho (ms/face) |
|---|---|---:|---:|---:|
| AHBody | W4 chunks | 5 | 0.977 | 0.700 |
| AHBody | W4 utilization | 5 | -0.975 | -0.700 |
| AHBody | estimated W4 instructions/face | 5 | 0.977 | 0.700 |
| AHBody | W4 active unique lines/chunk | 5 | -0.933 | -0.700 |
| AHBody | W4 active 4KiB pages/chunk | 5 | -0.859 | -0.300 |
| AHBody | W4 line reuse distance P90 | 5 | 0.922 | 0.527 |
| AHBody | scatter line reuse distance P90 | 5 | 0.922 | 0.527 |
| AHBody | matrix bandwidth | 5 | 0.461 | 0.359 |
| AHBody | mean row neighbour span | 5 | 0.966 | 0.975 |
| AHBody | P99 row neighbour span | 5 | 0.941 | 0.975 |
| Cavity3d | W4 chunks | 5 | -0.235 | -0.224 |
| Cavity3d | W4 utilization | 5 | 0.235 | 0.224 |
| Cavity3d | estimated W4 instructions/face | 5 | -0.235 | -0.224 |
| Cavity3d | W4 active unique lines/chunk | 5 | 1.000 | 0.224 |
| Cavity3d | W4 active 4KiB pages/chunk | 5 | 1.000 | 0.894 |
| Cavity3d | W4 line reuse distance P90 | 5 | 1.000 | 0.707 |
| Cavity3d | active psi page reuse distance P90 | 5 | 1.000 | 0.707 |
| Cavity3d | scatter line reuse distance P90 | 5 | 1.000 | 0.707 |
| Cavity3d | matrix bandwidth | 5 | 0.414 | -0.224 |
| Cavity3d | mean row neighbour span | 5 | 0.957 | 0.224 |
| Cavity3d | P99 row neighbour span | 5 | 0.994 | 0.224 |
| ConicalDiffuser | W4 chunks | 5 | 0.023 | 0.400 |
| ConicalDiffuser | W4 utilization | 5 | -0.023 | -0.400 |
| ConicalDiffuser | estimated W4 instructions/face | 5 | 0.023 | 0.400 |
| ConicalDiffuser | W4 active unique lines/chunk | 5 | 0.999 | 0.100 |
| ConicalDiffuser | W4 active 4KiB pages/chunk | 5 | 1.000 | 0.600 |
| ConicalDiffuser | W4 line reuse distance P90 | 5 | 0.999 | 0.707 |
| ConicalDiffuser | active psi page reuse distance P90 | 5 | 0.999 | 0.707 |
| ConicalDiffuser | scatter line reuse distance P90 | 5 | 0.999 | 0.707 |
| ConicalDiffuser | matrix bandwidth | 5 | 0.470 | 0.564 |
| ConicalDiffuser | mean row neighbour span | 5 | 0.975 | 0.667 |
| ConicalDiffuser | P99 row neighbour span | 5 | 0.999 | 0.105 |
| MTBHPC | W4 chunks | 5 | 0.922 | 0.900 |
| MTBHPC | W4 utilization | 5 | -0.914 | -0.900 |
| MTBHPC | estimated W4 instructions/face | 5 | 0.922 | 0.900 |
| MTBHPC | W4 active unique lines/chunk | 5 | -0.949 | -0.900 |
| MTBHPC | W4 active 4KiB pages/chunk | 5 | -0.962 | -0.900 |
| MTBHPC | W4 line reuse distance P90 | 5 | 0.994 | 0.894 |
| MTBHPC | active psi page reuse distance P90 | 5 | 0.994 | 0.707 |
| MTBHPC | scatter line reuse distance P90 | 5 | 0.994 | 0.894 |
| MTBHPC | matrix bandwidth | 5 | 0.700 | 0.667 |
| MTBHPC | mean row neighbour span | 5 | 0.984 | 0.051 |
| MTBHPC | P99 row neighbour span | 5 | 0.990 | 0.051 |
| MotorBike | W4 chunks | 4 | 0.984 | 1.000 |
| MotorBike | W4 utilization | 4 | -0.986 | -1.000 |
| MotorBike | estimated W4 instructions/face | 4 | 0.984 | 1.000 |
| MotorBike | W4 active unique lines/chunk | 4 | -0.971 | -1.000 |
| MotorBike | W4 active 4KiB pages/chunk | 4 | -0.936 | -1.000 |
| MotorBike | W4 line reuse distance P90 | 4 | 0.874 | 0.800 |
| MotorBike | active psi page reuse distance P90 | 4 | 0.874 | 0.775 |
| MotorBike | scatter line reuse distance P90 | 4 | 0.874 | 0.800 |
| MotorBike | matrix bandwidth | 4 | 0.869 | 0.316 |
| MotorBike | mean row neighbour span | 4 | 0.851 | 0.316 |
| MotorBike | P99 row neighbour span | 4 | 0.863 | 0.316 |
| WD_DamBreak | W4 chunks | 5 | 0.541 | 0.112 |
| WD_DamBreak | W4 utilization | 5 | -0.541 | -0.112 |
| WD_DamBreak | estimated W4 instructions/face | 5 | 0.541 | 0.112 |
| WD_DamBreak | W4 active unique lines/chunk | 5 | 0.999 | 0.100 |
| WD_DamBreak | W4 active 4KiB pages/chunk | 5 | 0.999 | 0.800 |
| WD_DamBreak | W4 line reuse distance P90 | 5 | 1.000 | 0.707 |
| WD_DamBreak | active psi page reuse distance P90 | 5 | 1.000 | 0.707 |
| WD_DamBreak | scatter line reuse distance P90 | 5 | 1.000 | 0.707 |
| WD_DamBreak | matrix bandwidth | 5 | 0.936 | 0.154 |
| WD_DamBreak | mean row neighbour span | 5 | 0.999 | 0.154 |
| WD_DamBreak | P99 row neighbour span | 5 | 0.999 | 0.359 |

Within-mesh coefficients are descriptive with only four or five orderings each. A high cross-mesh correlation can be driven by mesh size or family and is not evidence of causality.

## Answers and interpretation

1. **Renumbering and chunk utilization:** W4 active-slot utilization ranges from about 66.5–70.8% on irregular meshes and 74.6–74.8% on the regular meshes. Renumbering mainly changes which cell is the LDU owner of an edge, so the internal-face count stays fixed while owner-row chunk counts change. For example, MotorBike W4 chunks range from 373,443 (RCM) to 396,249 (Original), and AHBody from 3,101,640 (C) to 3,217,468 (Original). The complete owner and neighbour histograms, including zeros, are in `degree_statistics.csv`.
2. **Eight-wide kernels:** none of the 29 numberings reaches the W8 instruction break-even ratio `N4/N8 > 42/24 = 1.75` (nor the 43/24 threshold for M8.1x512). Ratios range from approximately 1.000 on regular meshes to 1.209 on AHBody Original. In the four matched M8-wide cases, all W8 kernels were slower than M6.4d: about 16–37% depending on mesh/order/kernel. This agrees with the direction of the chunk-body estimate, but the estimate is not a complete kernel model and does not establish causality.
3. **Locality versus measured time:** across 29 M6.4d cases, active 4KiB pages/chunk (Spearman 0.775), P99 neighbour span (0.802), and active psi page reuse-distance P90 (0.809) have the strongest listed rank associations with ms/face. These are exploratory and confounded by mesh family. Within-mesh samples are only four or five points and are not reliable correlation evidence.
4. **AHBody:** mean owner degree is invariant at 3.020, but the owner tail and ordering vary: Original has P99=6/max=15 and 19.5% of rows require multiple W4 chunks; C has P99=6/max=11 and 9.1% multi-chunk; RCM has P99=5/max=10 and 10.8% multi-chunk. Original also has line reuse P90=287 versus 10–11 for the four reordered layouts, and its M6.4d time is 14.91 ms versus 11.12–12.74 ms for reordered cases. This indicates a large locality change, while degree/chunk properties alone do not rank the reordered timings.
5. **Chunk reduction potential:** the unconstrained lower bound is `ceil(F/W)` and actual W4 counts exceed it by roughly 34–50% across cases. A single adjacent-cell swap reduces W4 chunks for some edges (for example 2.45M of 8.59M AHBody Original candidate edges), but many are neutral or increase the count. These are exact isolated counterfactuals; their deltas cannot be summed because swaps interact and a valid global permutation must preserve LDU face semantics.
6. **Gather/scatter trade-off:** the output keeps the streams separate. With the current owner-grouped LDU face sequence, scatter destination indices and active gather neighbour indices follow the same index sequence, so their line-distance/reuse summaries can coincide; they are still distinct data streams and this does not combine their cache states or costs. Inactive M6.4d main-chunk lanes still issue index and psi loads before product masking; safe terminal lanes read `psi[0]`.
7. **Promising starting numberings:** no single numbering wins across meshes. In the archived M6.4d campaign, RCM is fastest for MotorBike, MTBHPC and WD_DamBreak; the existing renumbering is fastest for AHBody and Cavity3d; BFS C is fastest for ConicalDiffuser. RCM/existing layouts are good starting references on most irregular meshes because they sharply reduce long-range reuse distances, but the ConicalDiffuser result and the modest performance gaps show that a GS-aware objective should keep owner-row chunking, gather locality and scatter behavior separate.

## Files

- `analysis.json` — complete per-case results and timing source records.
- `summary.csv` — requested 29-case primary table.
- `timing_comparison.csv` and `timing_campaign_summary.csv` — archived samples summarized without combining incompatible campaigns.
- `degree_statistics.csv`, `chunk_utilization.csv`, `gather_instruction_estimates.csv`, `structural_metrics.csv`, `chunk_locality.csv`, `reuse_distance.csv`, `scatter_locality.csv`, `scatter_windows.csv`, `stream_transitions.csv`, `adjacent_swap_candidates.csv` — machine-readable detailed tables.
- `config.json` — source paths, code hashes, assumptions, and run timing.
