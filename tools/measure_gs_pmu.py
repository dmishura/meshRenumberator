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
    ["cycles", "instructions"],
    ["task-clock", "cpu-migrations"],
    ["l1d_miss.load", "mem_load_retired.l1_hit", "mem_load_retired.l1_miss"],
    ["l2_rqsts.all_demand_data_rd", "l2_rqsts.demand_data_rd_miss"],
    ["mem_load_retired.l3_hit", "mem_load_retired.l3_miss"],
    ["dtlb_load_misses.walk_completed"],
    ["cycle_activity.cycles_mem_any", "cycle_activity.stalls_total"],
    ["l1d_miss.fb_full", "l1d_miss.l2_stalls", "l1d_pending.load_cycles"],
]


def event_spec(events):
    generic = {"task-clock", "cpu-migrations"}
    qualified = ",".join(event if event in generic else f"cpu_core/{event}/u"
                         for event in events)
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
            try:
                running_time_ns = int(fields[3].strip())
            except (IndexError, ValueError):
                running_time_ns = None
            rows.append({"event": event_name(event), "count": count, "raw_count": raw,
                         "coverage_percent": coverage, "unavailable": count is None})
            rows[-1]["perf_running_time_ns"] = running_time_ns
    return rows


def experiment_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and
                  (p / "permutation.npz").is_file() and
                  (p / "validation.json").is_file() and
                  (p / "config.json").is_file())


def measure_one(exp: Path, probe: Path, output: Path, samples: int,
                groups: list[list[str]], cpu_set: str, sweeps: int,
                width: int, perf: str, taskset: str, kernel: str = "OriginalGS",
                sample_offset: int = 0) -> list[dict]:
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
        for local_sample in range(samples):
            sample = sample_offset + local_sample
            for group_index, events in enumerate(groups):
                prefix = f"sample-{sample:02d}-group-{group_index:02d}"
                logfile = output / f"{prefix}.perf.csv"
                ctl, ack = temp / f"{prefix}.ctl", temp / f"{prefix}.ack"
                os.mkfifo(ctl)
                os.mkfifo(ack)
                env = os.environ.copy()
                env.update({"MESH_PMU_CONTROL": str(ctl), "MESH_PMU_ACK": str(ack),
                            "MESH_PMU_KERNEL": kernel,
                            "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
                            "OMP_DYNAMIC": "FALSE"})
                probe_cmd = [str(probe), str(source), str(exp / "polyMesh"),
                             str(temp / "cells.bin"), str(temp / "faces.bin"),
                             "1", str(sweeps), "1", str(width)]
                command = [taskset, "-c", cpu_set, perf, "stat", "--delay=-1",
                           f"--control=fifo:{ctl},{ack}", "-x", ",", "-o", str(logfile),
                           "-e", event_spec(events), "--", *probe_cmd]
                # Probe performs validation and warmup before enabling the counters.
                freq_before = cpu_frequency_khz(cpu_set)
                with (output / f"{prefix}.stdout.log").open("w") as stdout:
                    proc = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT,
                                          env=env, timeout=45)
                freq_after = cpu_frequency_khz(cpu_set)
                try:
                    ctl.unlink()
                    ack.unlink()
                except FileNotFoundError:
                    pass
                if proc.returncode != 0:
                    raise RuntimeError(f"perf/probe failed ({proc.returncode}); see {output / (prefix + '.stdout.log')}")
                parsed = {row["event"]: row for row in perf_rows(logfile)}
                sample_lines = [line for line in (output / f"{prefix}.stdout.log").read_text().splitlines()
                                if line.startswith(f"sample,{kernel},1,")]
                if len(sample_lines) != 1:
                    raise RuntimeError("expected exactly one measured GS batch")
                ms = float(sample_lines[0].split(",")[4])
                for name in events:
                    item = parsed.get(name, {"event": name, "count": None, "raw_count": "missing",
                                             "coverage_percent": None, "unavailable": True})
                    item.update({"experiment": exp.name, "sample": sample,
                                 "group": group_index, "requested_events": ";".join(events),
                                 "log": logfile.name, "ms_per_sweep": ms,
                                 "cpu_freq_before_khz": freq_before,
                                 "cpu_freq_after_khz": freq_after,
                                 "perf_running_time_ns": item.get("perf_running_time_ns"),
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


def cpu_frequency_khz(cpu_set: str):
    """Best-effort instantaneous cpufreq sysfs reading for a pinned CPU."""
    if not cpu_set.isdecimal():
        return None
    path = Path(f"/sys/devices/system/cpu/cpu{cpu_set}/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


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
    parser.add_argument("--kernel", choices=("OriginalGS", "IH1W1024", "SWndReload4", "SWnd4", "MinimalReload4", "Padded4", "OptimizedMinimalReload4"), default="OriginalGS")
    parser.add_argument("--interleave-order",
                        help="comma-separated experiment names; measurement rounds alternate forward and reverse")
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
    if not experiments:
        parser.error(f"no experiment directories found under {args.input_root}")
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
    interleave = ([name.strip() for name in args.interleave_order.split(",")]
                  if args.interleave_order else [exp.name for exp in experiments])
    by_name = {exp.name: exp for exp in experiments}
    if len(interleave) != len(set(interleave)) or set(interleave) != set(by_name):
        parser.error("--interleave-order must list each experiment directory exactly once")
    lock_path = args.input_root.resolve().parent / ".benchmark.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sample in range(args.samples):
            order = interleave if sample % 2 == 0 else list(reversed(interleave))
            for name in order:
                exp = by_name[name]
                rows = measure_one(exp, args.probe.resolve(), output_root / exp.name,
                                   1, groups, args.cpus, args.sweeps,
                                   args.width, args.perf, args.taskset,
                                   args.kernel, sample)
                all_rows.extend(rows)
                summaries.setdefault(exp.name, []).extend(rows)
        summaries = {name: summarize(rows) for name, rows in summaries.items()}
    fields = ["experiment", "sample", "group", "event", "count", "raw_count",
              "coverage_percent", "unavailable", "requested_events", "log",
              "count_per_sweep", "ms_per_sweep", "cpu_freq_before_khz",
              "cpu_freq_after_khz", "perf_running_time_ns"]
    with (output_root / "hardware_counters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    (output_root / "summaries.json").write_text(json.dumps({
        "settings": {"samples": args.samples, "cpus": args.cpus,
                     "sweeps": args.sweeps, "width": args.width,
                     "kernel": args.kernel, "interleave_order": interleave,
                     "events": groups, "unavailable_events": unavailable,
                     "probe": str(args.probe.resolve()),
                     "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest()},
        "experiments": summaries,
    }, indent=2) + "\n")
    print(output_root)


if __name__ == "__main__":
    main()
