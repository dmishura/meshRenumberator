#!/usr/bin/env python3
"""Collect controlled perf-stat counters around the serial GS solve in gs_probe."""
from __future__ import annotations

import argparse
import csv
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


# Keep groups small, qualify events to the pinned P-core PMU, and verify
# coverage. Cycles and instructions are paired for per-sample IPC.
DEFAULT_GROUPS = [
    ["cycles", "instructions", "cache-references", "cache-misses"],
    ["L1-dcache-loads", "L1-dcache-load-misses", "LLC-loads", "LLC-load-misses"],
    ["dTLB-loads", "dTLB-load-misses", "cycle_activity.cycles_mem_any",
     "cycle_activity.stalls_total"],
    ["mem_load_retired.l3_miss", "dtlb_load_misses.walk_completed"],
]


def event_spec(events):
    qualified = ",".join(f"cpu_core/{event}/u" for event in events)
    return "{" + qualified + "}" if len(events) > 1 else qualified


def event_name(event):
    if event.startswith("cpu_core/"):
        return event.split("/")[1]
    return event


def discover(args, groups, output):
    supported, unavailable = [], []
    for index, events in enumerate(groups):
        path = output / f"event-probe-{index:02d}.csv"
        command = [args.taskset, "-c", args.cpus, args.perf, "stat",
                   "-x", ",", "-o", str(path), "-e", event_spec(events), "--", "/usr/bin/true"]
        probe = subprocess.run(command, capture_output=True, text=True)
        (output / f"event-probe-{index:02d}.log").write_text(probe.stderr)
        measured = perf_rows(path) if path.is_file() else []
        available = {row["event"] for row in measured if not row["unavailable"]}
        if probe.returncode == 0 and available == set(events):
            supported.append(events)
        elif len(events) > 1:
            for event in events:
                one_path = output / f"event-probe-{index:02d}-{event}.csv"
                one_command = command.copy()
                one_command[one_command.index("-o") + 1] = str(one_path)
                one_command[one_command.index("-e") + 1] = event_spec([event])
                one = subprocess.run(one_command, capture_output=True, text=True)
                (output / f"event-probe-{index:02d}-{event}.log").write_text(one.stderr)
                values = perf_rows(one_path) if one_path.is_file() else []
                if one.returncode == 0 and any(not row["unavailable"] for row in values):
                    supported.append([event])
                else:
                    unavailable.append(event)
        else:
            unavailable.extend(events)
    return supported, unavailable


def perf_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", errors="replace") as stream:
        for fields in csv.reader(stream):
            if len(fields) < 3:
                continue
            raw = fields[0].strip()
            if not raw or raw.startswith("#"):
                continue
            event = fields[2].strip()
            if not event:
                continue
            try:
                count = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                count = None
            try:
                coverage = float(fields[4].strip().rstrip("%"))
            except (IndexError, ValueError):
                coverage = None
            rows.append({"event": event_name(event), "count": count, "raw_count": raw,
                         "coverage_percent": coverage, "unavailable": count is None})
    return rows


def experiment_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and
                  (p / "permutation.npz").is_file() and
                  (p / "validation.json").is_file() and
                  (p / "config.json").is_file())


def measure_one(exp: Path, probe: Path, output: Path, samples: int,
                groups: list[list[str]], cpu_set: str, sweeps: int,
                width: int, perf: str, taskset: str) -> list[dict]:
    validation = json.loads((exp / "validation.json").read_text())
    nfaces = validation.get("internal_faces")
    if nfaces is None:
        topology = validation.get("topology", {})
        nfaces = topology.get("internal_faces")
    if not isinstance(nfaces, int) or nfaces < 0:
        raise ValueError(f"{exp}: validation.json lacks integer internal_faces")
    config = json.loads((exp / "config.json").read_text())
    source = config.get("source")
    if not source:
        raise ValueError(f"{exp}: config.json lacks source")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="gs-pmu-") as temp_name:
        temp = Path(temp_name)
        with np.load(exp / "permutation.npz") as maps:
            maps["new_to_old"].tofile(temp / "cells.bin")
            faces = maps["new_face_to_old"]
            if faces.size < nfaces:
                raise ValueError(f"{exp}: face permutation shorter than internal_faces")
            faces[:nfaces].tofile(temp / "faces.bin")
        for sample in range(samples):
            for group_index, events in enumerate(groups):
                prefix = f"sample-{sample:02d}-group-{group_index:02d}"
                logfile = output / f"{prefix}.perf.csv"
                ctl, ack = temp / f"{prefix}.ctl", temp / f"{prefix}.ack"
                os.mkfifo(ctl)
                os.mkfifo(ack)
                env = os.environ.copy()
                env.update({"MESH_PMU_CONTROL": str(ctl), "MESH_PMU_ACK": str(ack),
                            "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
                            "OMP_DYNAMIC": "FALSE"})
                probe_cmd = [str(probe), str(source), str(exp / "polyMesh"),
                             str(temp / "cells.bin"), str(temp / "faces.bin"),
                             "1", str(sweeps), "1", str(width)]
                command = [taskset, "-c", cpu_set, perf, "stat", "--delay=-1",
                           f"--control=fifo:{ctl},{ack}", "-x", ",", "-o", str(logfile),
                           "-e", event_spec(events), "--", *probe_cmd]
                # Probe performs validation and warmup before enabling the counters.
                with (output / f"{prefix}.stdout.log").open("w") as stdout:
                    proc = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT,
                                          env=env, timeout=45)
                try:
                    ctl.unlink()
                    ack.unlink()
                except FileNotFoundError:
                    pass
                if proc.returncode != 0:
                    raise RuntimeError(f"perf/probe failed ({proc.returncode}); see {output / (prefix + '.stdout.log')}")
                parsed = {row["event"]: row for row in perf_rows(logfile)}
                sample_lines = [line for line in (output / f"{prefix}.stdout.log").read_text().splitlines()
                                if line.startswith("sample,OriginalGS,1,")]
                if len(sample_lines) != 1:
                    raise RuntimeError("expected exactly one measured GS batch")
                ms = float(sample_lines[0].split(",")[4])
                for name in events:
                    item = parsed.get(name, {"event": name, "count": None, "raw_count": "missing",
                                             "coverage_percent": None, "unavailable": True})
                    item.update({"experiment": exp.name, "sample": sample,
                                 "group": group_index, "requested_events": ";".join(events),
                                 "log": logfile.name, "ms_per_sweep": ms,
                                 "count_per_sweep": item["count"] / sweeps
                                 if item["count"] is not None else None})
                    rows.append(item)
    return rows


def usable(row):
    return (row["count_per_sweep"] is not None and row["coverage_percent"] is not None
            and row["coverage_percent"] >= 99.0)


def stats(samples):
    return {"samples": len(samples),
            "median": float(np.median(samples)) if samples else None,
            "sample_stdev": float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
            "min": min(samples) if samples else None, "max": max(samples) if samples else None}


def summarize(rows):
    events = {}
    for key in sorted({row["event"] for row in rows}):
        group = [row for row in rows if row["event"] == key]
        samples = [row["count_per_sweep"] for row in group if usable(row)]
        coverage = [row["coverage_percent"] for row in group
                    if row["coverage_percent"] is not None]
        events[key] = {"per_sweep": stats(samples),
                       "unavailable_samples": sum(row["unavailable"] for row in group),
                       "unverified_coverage_samples": len(group) - len(coverage),
                       "minimum_coverage_percent": min(coverage) if coverage else None,
                       "multiplexed_samples": sum(value < 99.0 for value in coverage)}
    pairs = {}
    timed = []
    for row in rows:
        if row["event"] in {"cycles", "instructions"} and usable(row):
            pair = pairs.setdefault((row["sample"], row["group"]), {})
            pair[row["event"]] = row["count_per_sweep"]
            if row["event"] == "cycles":
                timed.append(row["ms_per_sweep"])
    ipc = [p["instructions"] / p["cycles"] for p in pairs.values()
           if "instructions" in p and p.get("cycles")]
    return {"events": events, "ipc": stats(ipc), "ms_per_sweep_cycles_group": stats(timed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="directory containing experiment subdirectories")
    parser.add_argument("probe", type=Path, help="built gs_probe executable")
    parser.add_argument("--output", type=Path, help="output directory (default INPUT_ROOT/pmu)")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--cpus", default="0,1,2,3")
    parser.add_argument("--sweeps", type=int, default=64)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--perf", default="perf")
    parser.add_argument("--taskset", default="taskset")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--events", help="comma separated P-core events measured separately")
    selection.add_argument("--event-group", action="append",
                           help="comma separated P-core events measured as a group; repeatable")
    args = parser.parse_args()
    if args.samples < 1 or args.sweeps < 1 or args.width < 1:
        parser.error("samples, sweeps, and width must be positive")
    if not args.probe.is_file():
        parser.error(f"probe not found: {args.probe}")
    experiments = experiment_dirs(args.input_root.resolve())
    if len(experiments) != 4:
        parser.error(f"expected four experiment directories under {args.input_root}, found {len(experiments)}")
    if args.event_group is not None:
        groups = [[event.strip() for event in group.split(",") if event.strip()]
                  for group in args.event_group]
    elif args.events is not None:
        groups = [[event.strip()] for event in args.events.split(",") if event.strip()]
    else:
        groups = DEFAULT_GROUPS
    if not groups or any(not group or len(group) > 4 for group in groups):
        parser.error("supply one to four events per hardware group")
    output_root = (args.output or args.input_root / "pmu").resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    groups, unavailable = discover(args, groups, output_root)
    if not groups:
        raise RuntimeError(f"no accessible hardware counters; see {output_root}")
    all_rows = []
    summaries = {}
    lock_path = args.input_root.resolve().parent / ".benchmark.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for exp in experiments:
            rows = measure_one(exp, args.probe.resolve(), output_root / exp.name,
                               args.samples, groups, args.cpus, args.sweeps,
                               args.width, args.perf, args.taskset)
            all_rows.extend(rows)
            summaries[exp.name] = summarize(rows)
    fields = ["experiment", "sample", "group", "event", "count", "raw_count",
              "coverage_percent", "unavailable", "requested_events", "log",
              "count_per_sweep", "ms_per_sweep"]
    with (output_root / "hardware_counters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    (output_root / "summaries.json").write_text(json.dumps({
        "settings": {"samples": args.samples, "cpus": args.cpus,
                     "sweeps": args.sweeps, "width": args.width,
                                          "events": groups, "unavailable_events": unavailable,
                     "probe": str(args.probe.resolve()),
                     "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest()},
        "experiments": summaries,
    }, indent=2) + "\n")
    print(output_root)


if __name__ == "__main__":
    main()
