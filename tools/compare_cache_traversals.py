"""Summarize eligible cache traversal timings against matched references."""
import argparse
import csv
from pathlib import Path

from compare_cache_partitions import read_json, read_variant


def measured(name, experiment, diagnostic=None):
    diagnostic = diagnostic or experiment / "working_set"
    row, settings = read_variant(name, experiment, diagnostic)
    meta = experiment / "candidate.json"
    if meta.is_file():
        record = read_json(meta)
        if record["reasons"]:
            raise ValueError(f"{name} was rejected by the traversal guard")
        schedule = record["schedule"]
        row["traversal_seconds"] = record["wall_seconds"]
    else:
        saved = diagnostic / "layout.json"
        schedule = (read_json(saved) if saved.is_file()
                    else read_json(experiment / "locality.json")["IH1_packed_arrays"])
        row["traversal_seconds"] = ""
    index = schedule["index_kahn_wavefronts"]
    row["waves"] = index["levels"]
    row["median_wave_width"] = index["median_rows"]
    return row, settings


def compare(directory, references, candidates):
    measured_runs = [measured(*ref) for ref in references]
    measured_runs.extend(measured("traversal_" + path.name, path) for path in candidates)
    if any(settings != measured_runs[0][1] for _, settings in measured_runs[1:]):
        raise ValueError("benchmark settings or probe hashes differ")
    rows = [row for row, _ in measured_runs]
    kernels = ("gs", "ih1", "index_kahn")
    for row in rows:
        for field in kernels:
            for other in rows[:len(references)]:
                row[f"{field}_speedup_vs_{other['numbering']}"] = (
                    other[f"{field}_median_ms"] / row[f"{field}_median_ms"])
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Cache traversal benchmark", "",
             "Timings are ms/sweep, median ± sample SD. The footprint is the GS psi read/write stream.",
             "",
             "| Numbering | GS 1T | IH1W1024 1T | IndexKahn 4T | Psi lines K1024 | Reuse P90 | Waves | Coeff transitions |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        timings = " | ".join(f"{row[k + '_median_ms']:.3f} ± {row[k + '_sample_sd_ms']:.3f}"
                             for k in kernels)
        lines.append(f"| {row['numbering']} | {timings} | {row['psi_read_write_lines_k1024']:.1f} "
                     f"| {row['psi_read_write_reuse_p90']:,} | {row['waves']:,} "
                     f"| {row['coefficient_transitions']:,} |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", action="append", required=True,
                        metavar="NAME=EXPERIMENT[=DIAGNOSTICS]")
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    args = parser.parse_args()
    references = []
    for item in args.reference:
        name, *paths = item.split("=", 2)
        references.append((name, *map(Path, paths)))
    compare(args.output, references, args.candidate)
