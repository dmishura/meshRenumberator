#!/usr/bin/env python3
"""Create the readable GS numbering report from analyze_gs_numberings.py."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("analysis_dir",type=Path); a=ap.parse_args()
    p=a.analysis_dir/"analysis.json"; data=json.loads(p.read_text()); cases=data["cases"]
    m6path="experiments/M6.4d/amdr16-two-round-20261126/timing_summary.json"
    def timing(c, file, kernel):
        return c.get("timings",{}).get(file,{}).get(kernel,{})
    lines=[]
    lines += ["# GS-specific mesh numbering analysis", "",
      "Offline analysis of the 29 archived mesh/numbering combinations. No meshes, kernels, numbering algorithms, or benchmark runs were changed or created.", "",
      "## Definitions and execution model", "",
      "Owner degree is the actual count of internal LDU faces in each owner row. Neighbour degree separately counts the appearances of a cell in `upperAddr`; neither is replaced by undirected graph degree. All W=2/4/8/16 chunk calculations use `sum(ceil(owner_degree/W))`, with degree-zero rows contributing zero chunks.", "",
      "M6.4d's ordinary path issues four scalar index loads and four indirect `psi` loads per 4-slot chunk; inactive lanes are masked only after those loads and products, and coefficients are loaded unmasked. The last physical face suffix is diverted to the safe M6 terminal helper: masked index loads supply index zero in inactive lanes, so those lanes still read `psi[0]`; coefficient masked loads suppress inactive elements. M8 does eight scalar index/`psi` reads per 8-slot main chunk, with masked coefficient operands, then uses the same safe 4-wide M6 terminal helper for its final face suffix. Exact execution counts and active-only chunk geometry are stored separately in `analysis.json`.", "",
      "Cache lines assume `psi` is 8-byte FP64 and 64-byte aligned; 4 KiB and 2 MiB page IDs use the same base. Alignment sensitivity is sampled at base offsets 0, 8, …, 56 bytes for active chunk line counts. Reuse distances are exact distinct-line stack distances from the repository Fenwick backend; cold accesses are kept separate. These are logical locality measures, not hardware miss predictions.", "",
      "Chunk instruction estimates include only the given hot gather chunk instruction counts (24 for W4; 42/43 for W8). They exclude row/cell loop overhead, gather/scatter outside those chunks, and setup. The ‘inactive-slot equivalent’ prorates chunk-body instructions over unused lanes; it is not a literal count of instructions attributable to inactive lanes.", "",
      "## Summary: all 29 combinations", "",
      "| Mesh | Numbering | Cells | Faces | Chunks W4 | Util W4 | Chunks W8 | Util W8 | N4/N8 | Est instr/face W4 | Est instr/face W8 (42) | W4 lines/chunk | W4 XMM pairs same line | M6.4d ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        t=timing(c,m6path,"M6.4d")
        gl=c["gs_actual_execution"]['4']
        lines.append(f"| {c['mesh']} | {c['numbering']} | {c['cells']:,} | {c['internal_faces']:,} | {c['chunk_models']['4']['chunks']:,} | {c['chunk_models']['4']['utilization']:.3f} | {c['chunk_models']['8']['chunks']:,} | {c['chunk_models']['8']['utilization']:.3f} | {c['n4_n8_ratio']:.3f} | {c['gather_estimate_w4']['instructions_per_internal_face']:.3f} | {c['gather_estimate_w8_42']['instructions_per_internal_face']:.3f} | {gl['active_chunk_lines']['mean']:.3f} | {gl['same_line_xmm_pair_percent']:.1f}% | {fmt(t.get('median_ms'))} |")
    lines += ["", "## Performance files and compatibility", "",
      "M6.4d timing values in the main table come from one matched amdR16 two-round campaign (Scalar/M6/M6.4d, 29 cases, two samples per implementation). AVX-512 A/B timings are included only for the archived matched AVX2/A/B campaigns: other meshes, AHBody, and the separately corrected MotorBike C/D run. The 8-wide M8 timings exist only for MotorBike C/D and ConicalDiffuser C/D from the initial M8-wide campaign. Do not combine those campaign medians as if they were one matched run. Exact sources are retained per case in `analysis.json`.", "",
      "### M8 versus M6.4d where an M8 timing exists", "",
      "| Mesh | Numbering | M6.4d ms | M8.4x128 ms | M8.2x256 ms | M8.1x512 ms | N4/N8 | 42-instr model ratio vs W4 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    m8path="experiments/avx512-perf-amdR16/m8wide/results/initial/timing_summary.json"
    for c in cases:
        t=c.get("timings",{}).get(m8path,{})
        if not t: continue
        base=t.get("M6.4d-AVX2",{}).get("median_ms")
        lines.append(f"| {c['mesh']} | {c['numbering']} | {fmt(base)} | {fmt(t.get('M8.4x128',{}).get('median_ms'))} | {fmt(t.get('M8.2x256',{}).get('median_ms'))} | {fmt(t.get('M8.1x512',{}).get('median_ms'))} | {c['n4_n8_ratio']:.3f} | {24/42*c['chunk_models']['8']['chunks']/c['chunk_models']['4']['chunks']:.3f} |")
    lines += ["", "The rightmost ratio is estimated gather chunk-body instructions W8/W4 = `(42/24)*(N8/N4)`. It is below 1 only when N4/N8 exceeds 1.75. The 43-instruction ZMM kernel's threshold is 43/24 = 1.792. In these four cases, compare the estimated ratio with observed timing; it is not a complete-kernel cost model.", "",
      "## Within-chunk and stream locality", "",
      "See `chunk_locality.csv` for mean/median/P95/P99/max active-neighbour cache-line and page counts, distance distributions, same-line consecutive accesses, same-line XMM pairs, and histograms of chunks with 1–W cache lines. `analysis.json` additionally contains exact per-window distributions for 4, 8, 16, 32, 64, and 128 consecutive chunks, for both active and architecturally issued `psi` reads, plus exact line/page stack-distance summaries. `reuse_distance.csv` separates active-only line/page distances from executed line/page distances (including inactive slots). `scatter_locality.csv` reports destination-index locality and exact scatter line reuse distances independently.", "",
      "## Owner-degree and chunk details", "",
      "`degree_statistics.csv` contains complete owner and neighbour histograms, including zero-degree cells, mean/median/std/max/P95/P99, requested degree bands, and owner rows needing more than one 4-wide chunk. `analysis.json` carries utilization, active/inactive slots, full/partial chunk counts, and per-chunk active-element histograms for W=2/4/8/16.", "",
      "## Structural lower bounds and adjacent swaps", "",
      "For each width, `analysis.json` reports the unconstrained lower bound `ceil(F/W)`, actual chunk count, relative excess, and potential chunk-body instructions saved if that lower bound were attained. `adjacent_swap_candidates.csv` counts unique connected cell pairs and exact single-swap changes in W4 chunk count. These changes are local counterfactuals; improvements from separate swaps cannot be summed because swaps interact.", "",
      "## Correlations", "",
      "Only the matched M6.4d campaign is used for the 29-case primary performance correlations. Cross-mesh comparisons use time/cell and time/face; within-mesh statistics are exploratory because each mesh has only four or five numberings. AVX-512 A/B and M8 correlations are exported separately per matched archived campaign in `correlations.csv`, without pooling across binaries. The 8-wide M8 campaign has four MotorBike/ConicalDiffuser C/D points only, so those correlations are descriptive.", ""]
    # Exploratory correlations. SciPy reports p-values only as descriptive; the
    # report deliberately does not claim inference from 4–29 cases.
    fields={"W4 chunks":lambda c:c["chunk_models"]['4']["chunks"],
      "W4 utilization":lambda c:c["chunk_models"]['4']["utilization"],
      "estimated W4 instructions/face":lambda c:c["gather_estimate_w4"]["instructions_per_internal_face"],
      "W4 active unique lines/chunk":lambda c:c["gs_actual_execution"]['4']["active_chunk_lines"]["mean"],
      "W4 active 4KiB pages/chunk":lambda c:c["gs_actual_execution"]['4']["active_chunk_pages_4k"]["mean"],
      "W4 line reuse distance P90":lambda c:c["gs_actual_execution"]['4']["active_exact_line_reuse_distance"]["repeat_distance_quantiles"]["p90"],
      "active psi page reuse distance P90":lambda c:c["gs_actual_execution"]['4']["active_exact_page_reuse_distance"]["repeat_distance_quantiles"]["p90"],
      "active psi page reuse distance P90":lambda c:c["gs_actual_execution"]['4']["active_exact_page_reuse_distance"]["repeat_distance_quantiles"]["p90"],
      "scatter line reuse distance P90":lambda c:c["scatter"]["exact_line_reuse_distance"]["repeat_distance_quantiles"]["p90"],
      "matrix bandwidth":lambda c:c["graph_structure"]["bandwidth"],
      "mean row neighbour span":lambda c:c["row_neighbour_metrics"]["neighbour_span"]["mean"],
      "P99 row neighbour span":lambda c:c["row_neighbour_metrics"]["neighbour_span"]["p99"]}
    matched=[]
    for c in cases:
        t=timing(c,m6path,"M6.4d").get("median_ms")
        if t is not None: matched.append((c,t))
    corr_rows=[]
    lines += ["| Metric versus M6.4d ms/face | n | Pearson r | Spearman rho |", "|---|---:|---:|---:|"]
    for name,fn in fields.items():
        xs=np.asarray([fn(c) for c,t in matched],float); ys=np.asarray([t/c["internal_faces"] for c,t in matched],float)
        pr=float(pearsonr(xs,ys).statistic) if len(xs)>2 and np.std(xs)>0 and np.std(ys)>0 else float("nan")
        sr=float(spearmanr(xs,ys).statistic) if len(xs)>2 and np.std(xs)>0 and np.std(ys)>0 else float("nan")
        lines.append(f"| {name} | {len(xs)} | {fmt(pr)} | {fmt(sr)} |")
        corr_rows.append({"scope":"cross_mesh","mesh":"ALL","metric":name,"timing_norm":"ms_per_face","n":len(xs),"pearson_r":pr,"spearman_rho":sr})
        yn=np.asarray([t/c["cells"] for c,t in matched],float)
        corr_rows.append({"scope":"cross_mesh","mesh":"ALL","metric":name,"timing_norm":"ms_per_cell","n":len(xs),
                          "pearson_r":float(pearsonr(xs,yn).statistic) if len(xs)>2 and np.std(xs)>0 and np.std(yn)>0 else float("nan"),
                          "spearman_rho":float(spearmanr(xs,yn).statistic) if len(xs)>2 and np.std(xs)>0 and np.std(yn)>0 else float("nan")})
    lines += ["", "### Within-mesh correlations (exploratory)", "",
      "| Mesh | Metric | n | Pearson r (ms/face) | Spearman rho (ms/face) |", "|---|---|---:|---:|---:|"]
    for mesh_name in sorted(set(c["mesh"] for c,t in matched)):
        subset=[(c,t) for c,t in matched if c["mesh"]==mesh_name]
        for name,fn in fields.items():
            xs=np.asarray([fn(c) for c,t in subset],float); ys=np.asarray([t/c["internal_faces"] for c,t in subset],float)
            if len(xs)<3 or np.std(xs)==0 or np.std(ys)==0: continue
            pr=float(pearsonr(xs,ys).statistic); sr=float(spearmanr(xs,ys).statistic)
            corr_rows.append({"scope":"within_mesh","mesh":mesh_name,"metric":name,"timing_norm":"ms_per_face","n":len(xs),"pearson_r":pr,"spearman_rho":sr})
            lines.append(f"| {mesh_name} | {name} | {len(xs)} | {fmt(pr)} | {fmt(sr)} |")
    # Additional matched performance families: never pool values from
    # different binaries/campaigns into one correlation.
    timing_families=[
      ("M6.4d matched",m6path,"M6.4d"),
      ("AVX512 other meshes", "experiments/avx512-perf-amdR16/other-meshes/timing/timing_summary.json", "AVX2"),
      ("AVX512 other meshes", "experiments/avx512-perf-amdR16/other-meshes/timing/timing_summary.json", "AVX512-A"),
      ("AVX512 other meshes", "experiments/avx512-perf-amdR16/other-meshes/timing/timing_summary.json", "AVX512-B"),
      ("AVX512 AHBody", "experiments/avx512-perf-amdR16/ahmedbody/timing/timing_summary.json", "AVX2"),
      ("AVX512 AHBody", "experiments/avx512-perf-amdR16/ahmedbody/timing/timing_summary.json", "AVX512-A"),
      ("AVX512 AHBody", "experiments/avx512-perf-amdR16/ahmedbody/timing/timing_summary.json", "AVX512-B"),
      ("AVX512 MotorBike C/D", "experiments/avx512-perf-amdR16/timing-corrected/timing_summary.json", "AVX2"),
      ("AVX512 MotorBike C/D", "experiments/avx512-perf-amdR16/timing-corrected/timing_summary.json", "AVX512-A"),
      ("AVX512 MotorBike C/D", "experiments/avx512-perf-amdR16/timing-corrected/timing_summary.json", "AVX512-B"),
      ("M8 wide", "experiments/avx512-perf-amdR16/m8wide/results/initial/timing_summary.json", "M8.4x128"),
      ("M8 wide", "experiments/avx512-perf-amdR16/m8wide/results/initial/timing_summary.json", "M8.2x256"),
      ("M8 wide", "experiments/avx512-perf-amdR16/m8wide/results/initial/timing_summary.json", "M8.1x512")]
    for family,source,kernel in timing_families:
        valid=[]
        for c in cases:
            rec=c.get("timings",{}).get(source,{}).get(kernel,{})
            if rec.get("median_ms") is not None: valid.append((c,rec["median_ms"]))
        if len(valid)<3: continue
        for name,fn in fields.items():
            xs=np.asarray([fn(c) for c,t in valid],float)
            for norm,ys in (("ms_per_face",np.asarray([t/c["internal_faces"] for c,t in valid])),
                            ("ms_per_cell",np.asarray([t/c["cells"] for c,t in valid]))):
                pr=float(pearsonr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else float("nan")
                sr=float(spearmanr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else float("nan")
                corr_rows.append({"scope":"matched_campaign","mesh":family,"metric":name,
                                  "timing_norm":f"{norm}:{kernel}","n":len(valid),"pearson_r":pr,"spearman_rho":sr})
        for mesh_name in sorted(set(c["mesh"] for c,t in valid)):
            subset=[(c,t) for c,t in valid if c["mesh"]==mesh_name]
            if len(subset)<3: continue
            for name,fn in fields.items():
                xs=np.asarray([fn(c) for c,t in subset],float)
                ys=np.asarray([t/c["internal_faces"] for c,t in subset],float)
                pr=float(pearsonr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else float("nan")
                sr=float(spearmanr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else float("nan")
                corr_rows.append({"scope":"within_mesh_matched_campaign","mesh":mesh_name,
                    "metric":name,"timing_norm":f"ms_per_face:{family}:{kernel}",
                    "n":len(subset),"pearson_r":pr,"spearman_rho":sr})
    import csv
    with (a.analysis_dir/"correlations.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(corr_rows[0]));w.writeheader();w.writerows(corr_rows)
    structural=[]; utilization=[]; estimates=[]; degrees=[]
    for c in cases:
        g=c["graph_structure"]; s=c["row_neighbour_metrics"]
        structural.append({"mesh":c["mesh"],"numbering":c["numbering"],"bandwidth":g["bandwidth"],
            "normalized_bandwidth":g["normalized_bandwidth"],"edge_distance":json.dumps(g["edge_distance"]),
            "row_span":json.dumps(s["neighbour_span"]),"row_relative_max":json.dumps(s["row_relative_max"])})
        for side in ("owner","neighbour"):
            d=c[f"{side}_degree"].copy();d["histogram"]=json.dumps(d["histogram"])
            degrees.append({"mesh":c["mesh"],"numbering":c["numbering"],"side":side,**d})
        for width,m in c["chunk_models"].items(): utilization.append({"mesh":c["mesh"],"numbering":c["numbering"],**m,
            "active_elements_per_chunk_histogram":json.dumps(m["active_elements_per_chunk_histogram"])})
        for width,inst in ((4,24),(8,42),(8,43)):
            key=f"gather_estimate_w{width}" if width==4 else f"gather_estimate_w8_{inst}"
            estimates.append({"mesh":c["mesh"],"numbering":c["numbering"],"width":width,
                "instructions_per_chunk":inst,**c[key]})
    def write_rows(filename,rows):
        with (a.analysis_dir/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_rows("structural_metrics.csv",structural)
    write_rows("degree_statistics.csv",degrees)
    write_rows("chunk_utilization.csv",utilization)
    write_rows("gather_instruction_estimates.csv",estimates)
    scatter_windows=[]; scatter_summary=[]
    for c in cases:
        scatter=c.get("scatter",{})
        for k,metrics in scatter.get("unique_line_windows_in_face_rows",{}).items():
            row={"mesh":c["mesh"],"numbering":c["numbering"],"window_rows":int(k)}
            for metric in ("unique_lines","unique_pages","span_elements","span_bytes",
                           "line_density","line_touches","reuse_factor"):
                for stat,value in metrics[metric].items():
                    row[f"{metric}_{stat}"]=value
            scatter_windows.append(row)
        if scatter:
            scatter_summary.append({"mesh":c["mesh"],"numbering":c["numbering"],
                "destination_distance":json.dumps(scatter["destination_distance_consecutive"]),
                "same_line_consecutive_percent":scatter["same_line_consecutive_percent"],
                "reuse_p90":scatter["exact_line_reuse_distance"]["repeat_distance_quantiles"]["p90"],
                "reuse_p99":scatter["exact_line_reuse_distance"]["repeat_distance_quantiles"]["p99"],
                "window32_lines":scatter["unique_line_windows_in_face_rows"].get("32",{}).get("unique_lines",{}).get("mean")})
    write_rows("scatter_windows.csv",scatter_windows)
    write_rows("scatter_locality.csv",scatter_summary)
    timing_rows=[]; speedups={}
    for c in cases:
        for source,records in c.get("timings",{}).items():
            baseline=("Scalar" if "Scalar" in records else
                      "AVX2" if "AVX2" in records else
                      "M6.4d-AVX2" if "M6.4d-AVX2" in records else
                      "M6.4d" if "M6.4d" in records else None)
            if baseline is None: continue
            base=records[baseline].get("median_ms")
            for kernel,record in records.items():
                value=record.get("median_ms")
                if value is None: continue
                speed=base/value if base else None
                timing_rows.append({"mesh":c["mesh"],"numbering":c["numbering"],"campaign":source,
                    "kernel":kernel,"baseline":baseline,"n":record.get("n"),"median_ms":value,
                    "sample_sd_ms":record.get("sample_sd_ms"),"speedup_vs_campaign_baseline":speed})
                if kernel!=baseline and speed is not None:
                    speedups.setdefault((source,kernel,baseline),[]).append(speed)
    write_rows("timing_comparison.csv",timing_rows)
    aggregate=[{"campaign":src,"kernel":kernel,"baseline":baseline,"cases":len(vals),
                "median_speedup":float(np.median(vals)),"min_speedup":float(np.min(vals)),
                "max_speedup":float(np.max(vals))}
               for (src,kernel,baseline),vals in speedups.items()]
    write_rows("timing_campaign_summary.csv",aggregate)
    timing_rows=[]; speedups={}
    for c in cases:
        for source,records in c.get("timings",{}).items():
            baseline=("Scalar" if "Scalar" in records else
                      "AVX2" if "AVX2" in records else
                      "M6.4d-AVX2" if "M6.4d-AVX2" in records else
                      "M6.4d" if "M6.4d" in records else None)
            if baseline is None: continue
            base=records[baseline].get("median_ms")
            for kernel,record in records.items():
                value=record.get("median_ms")
                if value is None: continue
                speed=base/value if base else None
                timing_rows.append({"mesh":c["mesh"],"numbering":c["numbering"],"campaign":source,
                    "kernel":kernel,"baseline":baseline,"n":record.get("n"),"median_ms":value,
                    "sample_sd_ms":record.get("sample_sd_ms"),"speedup_vs_campaign_baseline":speed})
                if kernel!=baseline and speed is not None:
                    speedups.setdefault((source,kernel,baseline),[]).append(speed)
    write_rows("timing_comparison.csv",timing_rows)
    aggregate=[{"campaign":src,"kernel":kernel,"baseline":baseline,"cases":len(vals),
                "median_speedup":float(np.median(vals)),"min_speedup":float(np.min(vals)),
                "max_speedup":float(np.max(vals))}
               for (src,kernel,baseline),vals in speedups.items()]
    write_rows("timing_campaign_summary.csv",aggregate)
    lines += ["", "Within-mesh coefficients are descriptive with only four or five orderings each. A high cross-mesh correlation can be driven by mesh size or family and is not evidence of causality.", "",
      "## Answers and interpretation", "",
      "1. **Renumbering and chunk utilization:** W4 active-slot utilization ranges from about 66.5–70.8% on irregular meshes and 74.6–74.8% on the regular meshes. Renumbering mainly changes which cell is the LDU owner of an edge, so the internal-face count stays fixed while owner-row chunk counts change. For example, MotorBike W4 chunks range from 373,443 (RCM) to 396,249 (Original), and AHBody from 3,101,640 (C) to 3,217,468 (Original). The complete owner and neighbour histograms, including zeros, are in `degree_statistics.csv`.",
      "2. **Eight-wide kernels:** none of the 29 numberings reaches the W8 instruction break-even ratio `N4/N8 > 42/24 = 1.75` (nor the 43/24 threshold for M8.1x512). Ratios range from approximately 1.000 on regular meshes to 1.209 on AHBody Original. In the four matched M8-wide cases, all W8 kernels were slower than M6.4d: about 16–37% depending on mesh/order/kernel. This agrees with the direction of the chunk-body estimate, but the estimate is not a complete kernel model and does not establish causality.",
      "3. **Locality versus measured time:** across 29 M6.4d cases, active 4KiB pages/chunk (Spearman 0.775), P99 neighbour span (0.802), and active psi page reuse-distance P90 (0.809) have the strongest listed rank associations with ms/face. These are exploratory and confounded by mesh family. Within-mesh samples are only four or five points and are not reliable correlation evidence.",
      "4. **AHBody:** mean owner degree is invariant at 3.020, but the owner tail and ordering vary: Original has P99=6/max=15 and 19.5% of rows require multiple W4 chunks; C has P99=6/max=11 and 9.1% multi-chunk; RCM has P99=5/max=10 and 10.8% multi-chunk. Original also has line reuse P90=287 versus 10–11 for the four reordered layouts, and its M6.4d time is 14.91 ms versus 11.12–12.74 ms for reordered cases. This indicates a large locality change, while degree/chunk properties alone do not rank the reordered timings.",
      "5. **Chunk reduction potential:** the unconstrained lower bound is `ceil(F/W)` and actual W4 counts exceed it by roughly 34–50% across cases. A single adjacent-cell swap reduces W4 chunks for some edges (for example 2.45M of 8.59M AHBody Original candidate edges), but many are neutral or increase the count. These are exact isolated counterfactuals; their deltas cannot be summed because swaps interact and a valid global permutation must preserve LDU face semantics.",
      "6. **Gather/scatter trade-off:** the output keeps the streams separate. With the current owner-grouped LDU face sequence, scatter destination indices and active gather neighbour indices follow the same index sequence, so their line-distance/reuse summaries can coincide; they are still distinct data streams and this does not combine their cache states or costs. Inactive M6.4d main-chunk lanes still issue index and psi loads before product masking; safe terminal lanes read `psi[0]`.",
      "7. **Promising starting numberings:** no single numbering wins across meshes. In the archived M6.4d campaign, RCM is fastest for MotorBike, MTBHPC and WD_DamBreak; the existing renumbering is fastest for AHBody and Cavity3d; BFS C is fastest for ConicalDiffuser. RCM/existing layouts are good starting references on most irregular meshes because they sharply reduce long-range reuse distances, but the ConicalDiffuser result and the modest performance gaps show that a GS-aware objective should keep owner-row chunking, gather locality and scatter behavior separate.", "",
      "## Files", "",
      "- `analysis.json` — complete per-case results and timing source records.", "- `summary.csv` — requested 29-case primary table.", "- `timing_comparison.csv` and `timing_campaign_summary.csv` — archived samples summarized without combining incompatible campaigns.", "- `degree_statistics.csv`, `chunk_utilization.csv`, `gather_instruction_estimates.csv`, `structural_metrics.csv`, `chunk_locality.csv`, `reuse_distance.csv`, `scatter_locality.csv`, `scatter_windows.csv`, `stream_transitions.csv`, `adjacent_swap_candidates.csv` — machine-readable detailed tables.", "- `config.json` — source paths, code hashes, assumptions, and run timing.", ""]
    (a.analysis_dir/"comparison.md").write_text("\n".join(lines))
    print(a.analysis_dir/"comparison.md")


if __name__=="__main__": main()
