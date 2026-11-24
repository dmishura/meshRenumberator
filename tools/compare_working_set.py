"""Table of matched MotorBike working-set and GS counter results."""
import argparse
import csv
import json
import math
from pathlib import Path

VARIANTS = ("original", "rcm", "locality_total", "locality_balanced")


def load_json(path):
    return json.loads(path.read_text())


def read_windows(path):
    with path.open(newline="") as stream:
        return {(row["stream"], int(row["window_rows"]), row["metric"]): row
                for row in csv.DictReader(stream)}


def check_ordering(bandwidth, previous):
    if bandwidth["topology_sha256"] != previous["checksum"]:
        raise ValueError("existing ordering record describes a different topology")
    distance = bandwidth["edge_distance"]
    for key, old in (("mean", "mean_edge_delta"), ("p50", "median_edge_delta"),
                     ("p90", "p90_edge_delta"), ("p99", "p99_edge_delta"),
                     ("max", "max_edge_delta")):
        if not math.isclose(distance[key], previous[old], rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"edge metric {key} does not match previous analysis")


def load_case(root, reference, variant):
    old = reference / variant
    current = root / variant
    benchmark, = old.glob("benchmark-*/summary.json")
    band = load_json(current / "bandwidth.json")
    check_ordering(band, load_json(old / "locality.json")["ordering"])
    old_benchmark = load_json(benchmark)
    if old_benchmark["topology"]["output_topology_sha256"] != band["topology_sha256"]:
        raise ValueError(f"{variant}: benchmark used a different topology")
    return {
        "band": band, "rows": load_json(current / "row_span.json"),
        "working": read_windows(current / "working_set.csv"),
        "coeff": read_windows(current / "coefficient_locality.csv"),
        "timing": old_benchmark["timings"]["OriginalGS/1T"],
        "reuse": load_json(current / "reuse_distance.json"),
    }


def compare(root, reference):
    cases = {variant: load_case(root, reference, variant) for variant in VARIANTS}
    collections = [root / "pmu"]
    if (root / "pmu_memory/summaries.json").is_file():
        collections.append(root / "pmu_memory")
    pmu = load_json(collections[0] / "summaries.json")
    for directory in collections[1:]:
        extra = load_json(directory / "summaries.json")
        keys = ("samples", "cpus", "sweeps", "width", "probe_sha256")
        if any(pmu["settings"][key] != extra["settings"][key] for key in keys):
            raise ValueError("hardware collections did not use the same run conditions")
        for variant in VARIANTS:
            pmu["experiments"][variant]["events"].update(extra["experiments"][variant]["events"])
    table = []

    def row(label, values, decimals=0):
        values = [f"{v:,.{decimals}f}" if v is not None else "unavailable" for v in values]
        table.append(f"| {label} | " + " | ".join(values) + " |")

    row("Original GS prior median, ms/sweep",
        [v["timing"]["median_ms_per_sweep"] for v in cases.values()], 3)
    row("Matrix bandwidth (cells)", [v["band"]["bandwidth"] for v in cases.values()])
    row("Normalized bandwidth B/N", [v["band"]["normalized_bandwidth"] for v in cases.values()], 4)
    for label, field in (("Mean neighbour span", "mean"), ("P99 neighbour span", "p99")):
        row(label, [v["rows"]["neighbour_span"][field] for v in cases.values()])

    for stream in ("psi_adjacency", "psi_gs_read", "bPrime_gs", "upper", "upperAddr"):
        kind = "working" if stream in {"psi_adjacency", "psi_gs_read", "bPrime_gs"} else "coeff"
        for k in (128, 1024):
            for metric, decimals in (("unique_lines", 1), ("reuse_factor", 2)):
                row(f"{stream} mean {metric}, K={k}", [
                    float(v[kind][stream, k, metric]["mean"]) for v in cases.values()], decimals)
    for stream in ("upper", "upperAddr"):
        row(f"{stream} P99 unique lines, K=1024", [
            float(v["coeff"][stream, 1024, "unique_lines"]["p99"]) for v in cases.values()])
    for stream in ("psi_adjacency", "psi_gs_read", "bPrime_gs"):
        row(f"{stream} stack distance P99", [
            v["reuse"][stream]["repeat_distance_quantiles"]["p99"] for v in cases.values()])

    events = ("instructions", "cycles", "cache-references", "cache-misses",
              "L1-dcache-load-misses", "LLC-load-misses", "dTLB-load-misses",
              "cycle_activity.cycles_mem_any", "mem_load_retired.l3_miss",
              "dtlb_load_misses.walk_completed", "cycle_activity.stalls_total",
              "memory_stalls.l1", "memory_stalls.l2", "memory_stalls.l3", "memory_stalls.mem")
    for event in events:
        row(f"{event}/sweep, median", [
            pmu["experiments"][v]["events"].get(event, {}).get("per_sweep", {}).get("median")
            for v in VARIANTS])
    row("IPC, median of paired samples",
        [pmu["experiments"][v]["ipc"]["median"] for v in VARIANTS], 3)
    row("Gated GS median, ms/sweep", [
        pmu["experiments"][v]["ms_per_sweep_cycles_group"]["median"] for v in VARIANTS], 3)

    content = """# MotorBike bandwidth and working-set comparison

The four inputs are the previously measured MotorBike meshes. Bandwidth uses
cell labels, and the working sets use 64-byte logical lines and 4 KiB pages.
Window figures show means unless marked P99. Stack distance is exact per stream.
PMU figures cover only the timed Original GS batches.

| Metric | Original | RCM | Locality total | Locality balanced |
|---|---:|---:|---:|---:|
""" + "\n".join(table) + "\n"
    (root / "comparison.md").write_text(content)
    records = []
    for directory in collections:
        with (directory / "hardware_counters.csv").open(newline="") as stream:
            records.extend({"collection": directory.name, **row} for row in csv.DictReader(stream))
    with (root / "hardware_counters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("previous_campaign", type=Path)
    args = parser.parse_args()
    compare(args.output_root, args.previous_campaign)
