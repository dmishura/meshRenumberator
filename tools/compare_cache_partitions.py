"""Compare cache-partition candidates and previously measured meshes."""
import argparse
import csv
import json
from pathlib import Path

WINDOWS = (32, 128, 512, 1024, 2048)
KERNELS = ("OriginalGS/1T", "IH1W1024/1T", "IndexKahn1024Blocked64OMP/4T")


def read_json(path):
    return json.loads(path.read_text())


def read_variant(name, experiment, diagnostics):
    summary_path, = experiment.glob("benchmark-*/summary.json")
    summary = read_json(summary_path)
    if not summary["topology"]["valid"] or not all(row["one_sweep_exact"] for row in summary["exact"]):
        raise ValueError(f"{name}: cannot compare a failed solver validation")
    bandwidth = read_json(diagnostics / "bandwidth.json")
    if bandwidth["topology_sha256"] != summary["topology"]["output_topology_sha256"]:
        raise ValueError(f"{name}: topology does not match benchmark summary")
    with (diagnostics / "working_set.csv").open(newline="") as stream:
        windows = {(row["stream"], int(row["window_rows"]), row["metric"]): row
                   for row in csv.DictReader(stream)}
    sizes = read_json(diagnostics / "row_span.json")
    reuse = read_json(diagnostics / "reuse_distance.json")
    coeff = read_json(diagnostics / "coefficient_trace.json")["arrays"]
    stats = read_json(experiment / "locality.json")["LDU_psi_read"]
    partition_file = experiment / "partition.json"
    partition = read_json(partition_file) if partition_file.is_file() else None
    result = {
        "numbering": name,
        "budget_kib": partition["settings"]["budget_kib"] if partition else "",
        "local_order": partition["selected"]["local_order"] if partition else "",
        "coarse_order": partition["selected"]["coarse_order"] if partition else "",
        "ws_objective": sum(float(windows["psi_gs_read_write", k, "unique_lines"]["mean"])
                            / k for k in WINDOWS) / len(WINDOWS),
        "psi_read_lines_k1024": float(windows["psi_gs_read", 1024, "unique_lines"]["mean"]),
        "psi_read_write_lines_k1024": float(windows["psi_gs_read_write", 1024, "unique_lines"]["mean"]),
        "psi_read_write_reuse_factor_k1024": float(
            windows["psi_gs_read_write", 1024, "reuse_factor"]["mean"]),
        "psi_read_write_reuse_p90": reuse["psi_gs_read_write"]["repeat_distance_quantiles"]["p90"],
        "coefficient_transitions": sum(coeff[k]["line_transitions"] for k in ("upper", "upperAddr")),
        "upper_transitions": coeff["upper"]["line_transitions"],
        "upperAddr_transitions": coeff["upperAddr"]["line_transitions"],
        "bandwidth": bandwidth["bandwidth"],
        "mean_neighbour_span": sizes["neighbour_span"]["mean"],
        "psi_total_jump": stats["totalJumpMagnitude"],
        "psi_jump_p99": stats["p99"], "psi_jump_p99_9": stats["p99.9"],
        "psi_jump_p99_99": stats["p99.99"], "psi_jump_max": stats["max"],
        "partition_count": "", "partition_mean_cells": "", "partition_p50_cells": "",
        "crossing_edge_fraction": "", "estimated_p90_bytes": "", "psi_actual_p90_bytes": "",
        "largest_p99_99_component": "", "p99_99_boundary_incident_fraction": "",
        "boundary_incident_fraction_all_cells": "", "renumber_seconds": "",
    }
    if partition:
        groups = partition["partition"]
        seams = partition["seams"]
        result.update({
            "partition_count": groups["count"],
            "partition_mean_cells": groups["size"]["mean"],
            "partition_p50_cells": groups["size"]["p50"],
            "crossing_edge_fraction": groups["crossing_edge_fraction"],
            "estimated_p90_bytes": groups["estimated_psi_and_graph_metadata_bytes"]["p90"],
            "psi_actual_p90_bytes": groups["actual_psi_bytes"]["p90"],
            "largest_p99_99_component": seams["largest_p99.99_component"],
            "p99_99_boundary_incident_fraction": seams["outliers"]["p99.99"]["boundary_incident_fraction"],
            "boundary_incident_fraction_all_cells": seams["boundary_fraction_all_cells"],
            "renumber_seconds": partition["wall_seconds"],
        })
    for key, field in zip(KERNELS, ("gs", "ih1", "index_kahn")):
        measure = summary["timings"][key]
        result[field + "_median_ms"] = measure["median_ms_per_sweep"]
        result[field + "_sample_sd_ms"] = measure["stdev_ms_per_sweep"]
    return result, summary["settings"]


def compare(root, references):
    measured = [read_variant(name, experiment, root / "references" / name)
                for name, experiment in references]
    for directory in sorted(root.glob("cache_*")):
        if (directory / "partition.json").is_file():
            measured.append(read_variant(directory.name, directory, directory / "working_set"))
    if any(settings != measured[0][1] for _, settings in measured[1:]):
        raise ValueError("benchmark settings or probe hashes differ")
    rows = [row for row, _ in measured]
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    content = [
        "| Numbering | GS 1T | IH1 1T | IndexKahn 4T | psi read/write lines K1024 | Reuse P90 | Upper + upperAddr transitions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        timing = " | ".join(f"{row[k + '_median_ms']:.3f} ± {row[k + '_sample_sd_ms']:.3f}"
                           for k in ("gs", "ih1", "index_kahn"))
        content.append(f"| {row['numbering']} | {timing} | {row['psi_read_write_lines_k1024']:.1f} "
                       f"| {row['psi_read_write_reuse_p90']:,} | {row['coefficient_transitions']:,} |")
    (root / "comparison.md").write_text(
        "# Cache-partition comparison\n\n"
        "Timings are milliseconds per sweep, median ± sample standard deviation. "
        "Reuse P90 is exact psi read/write stack distance in cache lines.\n\n" +
        "\n".join(content) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", action="append", required=True, metavar="NAME=EXPERIMENT")
    args = parser.parse_args()
    refs = []
    for value in args.reference:
        name, path = value.split("=", 1)
        refs.append((name, Path(path)))
    compare(args.root, refs)
