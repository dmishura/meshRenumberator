"""Candidate selection for one continuous, cache-aware graph traversal."""

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np


PROFILES = {
    "breadth": {"root": "peripheral", "scan": 1, "weights": (0.0, 0.0, 0.0, 1.0)},
    "reuse": {"root": "centre", "scan": 24, "weights": (1.0, 0.85, 0.40, 0.40)},
    "balanced": {"root": "centre", "scan": 12, "weights": (0.65, 0.50, 0.35, 1.20)},
}
SCHEDULES = ("index_kahn_wavefronts", "packed_ih1_wavefronts")


def build_order(n_cells, owner, neighbour, *, backend, weights, scan, recent, seed, root,
                audit_every=0):
    owner, neighbour = np.asarray(owner), np.asarray(neighbour)
    if (owner.ndim != 1 or owner.shape != neighbour.shape or not 0 < n_cells < 2**31 - 1
            or (owner.size and (owner.dtype.kind not in "iu" or neighbour.dtype.kind not in "iu"
                                or np.any(owner < 0) or np.any(owner >= neighbour)
                                or np.any(neighbour >= n_cells)))):
        raise ValueError("invalid traversal graph")
    if (not 0 <= seed < 2**64 or not 0 < scan < 2**31 or not 0 < recent < 2**31
            or not 0 <= audit_every < 2**63 or root not in {"seed", "peripheral", "centre"}
            or len(weights) != 4 or any(not math.isfinite(w) or not 0 <= w <= 1000 for w in weights)):
        raise ValueError("invalid traversal profile, weights, or parameters")
    executable = Path(backend).resolve()
    if not executable.is_file():
        raise ValueError(f"missing {executable}; build with bash tools/build_cache_traversal.sh")
    with tempfile.TemporaryDirectory(prefix="mesh-cache-traversal-") as directory:
        endpoints, labels = Path(directory) / "endpoints", Path(directory) / "labels"
        with endpoints.open("wb") as stream:
            owner.astype(np.int32, copy=False).tofile(stream)
            neighbour.astype(np.int32, copy=False).tofile(stream)
        options = (n_cells, owner.size, scan, recent, seed, root, audit_every, *weights)
        command = [str(executable), str(endpoints), str(labels), *map(str, options)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip() or "cache traversal failed")
        order = np.fromfile(labels, dtype=np.int32).astype("<i4", copy=False)
    statistics = json.loads(result.stdout)
    if (order.shape != (n_cells,) or np.any((order < 0) | (order >= n_cells))
            or np.unique(order).size != n_cells):
        raise ValueError("backend did not produce a bijective traversal")
    return order, statistics


def schedule_rejections(schedule, original, max_wave_ratio, min_median_wave_width):
    """Reject wave growth and narrowing separately for the two actual schedules."""
    reasons = []
    for name in SCHEDULES:
        current, baseline = schedule[name], original[name]
        if max_wave_ratio and current["levels"] > max_wave_ratio * baseline["levels"]:
            reasons.append(f"{name} wave count {current['levels']} > "
                           f"{max_wave_ratio} × Original {baseline['levels']}")
        floor = min(min_median_wave_width, baseline["median_rows"])
        if floor and current["median_rows"] < floor:
            reasons.append(f"{name} median width {current['median_rows']} < {floor}")
    return reasons


def choose_candidate(rows):
    accepted = [row for row in rows if not row["reasons"]]
    return min(accepted, key=lambda row: (row["score"], row["profile"]), default=None)


def layout(executable, mesh, width):
    command = [str(executable.resolve()), str(mesh), str(width)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "schedule inspection failed")
    result = json.loads(result.stdout)
    if result.get("width") != width or any(result[k]["levels"] < 1 for k in SCHEDULES):
        raise ValueError("schedule inspection returned unexpected wave widths")
    return result


def trace_reuse(mesh, backend, capacities):
    from working_set import gs_psi_trace, summarize_reuse, trace_metrics

    offsets, indices = gs_psi_trace(np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour),
                                   mesh.n_cells)
    _, hist = trace_metrics(offsets, indices, (mesh.n_cells,), None, backend, reuse=True)
    summary = summarize_reuse(hist, capacities)
    summary["policy"] = "fully associative write allocation, one initially cold GS psi sweep"
    for cap in summary["capacities"]:
        cap["cold_misses"] = summary["cold_accesses"]
        hits = int(hist[1:1 + cap["lines"]].sum())
        cap["capacity_misses"] = summary["total_accesses"] - summary["cold_accesses"] - hits
    return summary


def graph_summary(mesh):
    from working_set import distribution, graph_metrics, row_offsets

    owner, neighbour = np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour)
    band, span = graph_metrics(mesh.n_cells, owner, neighbour)
    # The actual GS gather degree and its variation, in natural row order.
    degree = np.bincount(owner, minlength=mesh.n_cells)
    starts = row_offsets(degree)
    # Mean overlap of adjacent nonempty rows' numeric endpoint intervals.
    # It is a range diagnostic, never interpreted as overlap of read sets.
    first = starts[:-1][degree > 0]
    last = starts[1:][degree > 0] - 1
    # Internal faces are grouped by owner. Neighbour order inside each
    # group is allowed to vary in Original, so reduce the actual min/max.
    low = np.minimum.reduceat(neighbour, first) if first.size else np.empty(0, dtype=np.int32)
    high = np.maximum.reduceat(neighbour, first) if first.size else np.empty(0, dtype=np.int32)
    overlap = np.maximum(0, np.minimum(high[:-1], high[1:]) - np.maximum(low[:-1], low[1:]) + 1)
    union = np.maximum(high[:-1], high[1:]) - np.minimum(low[:-1], low[1:]) + 1
    return {"bandwidth": band, "row_span": span, "gs_outgoing_degree": distribution(degree),
            "adjacent_row_degree_change": distribution(np.abs(np.diff(degree))),
            "interval_overlap_adjacent_nonempty_rows": distribution(overlap / union)}


def run(source, destination, mesh, offsets, vertices, args):
    from cache_partition import evaluate
    from locality import coefficient_metrics
    from mesh_renumber import digest, json_write, read_poly_mesh, write_numbered_experiment

    start = time.perf_counter()
    ratio, floor, tolerance = (args.max_wave_ratio, args.min_median_wave_width,
                               args.traversal_coeff_tolerance)
    if (not math.isfinite(ratio) or ratio < 0 or floor < 0 or not math.isfinite(tolerance)
            or tolerance < 0 or args.traversal_schedule_width < 1):
        raise ValueError("invalid traversal guard parameters")
    if not args.traversal_layout_probe.is_file():
        raise ValueError("missing schedule inspector; build with bash tools/build_cache_traversal.sh")
    baseline_coeff = coefficient_metrics(mesh)
    original_schedule = layout(args.traversal_layout_probe, source, args.traversal_schedule_width)
    original_ws = evaluate(mesh, args.traversal_ws_backend, baseline_coeff, tolerance)
    original_graph = graph_summary(mesh)
    profiles = list(PROFILES) if args.traversal_profile == "auto" else [args.traversal_profile]
    settings = {
        "initial": "original", "profiles": profiles, "seed": args.traversal_seed,
        "recent_rows": args.traversal_recent_rows, "verify_every": args.traversal_verify_every,
        "coefficient_tolerance": tolerance, "max_wave_ratio": ratio,
        "min_median_wave_width": floor, "schedule_width": args.traversal_schedule_width,
        "cache_kib": args.traversal_cache_kib,
        "traversal_backend_sha256": digest(args.traversal_backend),
        "working_set_backend_sha256": digest(args.traversal_ws_backend),
        "layout_probe_sha256": digest(args.traversal_layout_probe),
    }
    owner, neighbour = np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".traversal-") as directory:
        campaign = Path(directory)
        rows = []
        for profile in profiles:
            spec = dict(PROFILES[profile])
            if args.traversal_weights is not None:
                spec["weights"] = args.traversal_weights
            if args.traversal_frontier_scan is not None:
                spec["scan"] = args.traversal_frontier_scan
            if args.traversal_root is not None:
                spec["root"] = args.traversal_root
            beginning = time.perf_counter()
            order, frontier = build_order(
                mesh.n_cells, owner, neighbour, backend=args.traversal_backend,
                recent=args.traversal_recent_rows, seed=args.traversal_seed,
                audit_every=args.traversal_verify_every, **spec,
            )
            engine_seconds = time.perf_counter() - beginning
            result_dir = campaign / "candidates" / profile
            write_numbered_experiment(source, result_dir, mesh, offsets, vertices,
                                      "cache-traversal", order, args)
            numbered = read_poly_mesh(result_dir / "polyMesh")
            geometry = graph_summary(numbered)
            schedule = layout(args.traversal_layout_probe, result_dir / "polyMesh",
                              args.traversal_schedule_width)
            reasons = schedule_rejections(schedule, original_schedule, ratio, floor)
            old_span = original_graph["row_span"]["neighbour_span"]
            new_span = geometry["row_span"]["neighbour_span"]
            if all(new_span[k] > 1.5 * old_span[k] for k in ("mean", "p99")):
                reasons.append("both mean and P99 neighbour spans exceed 1.5 × Original")
            footprint = evaluate(numbered, args.traversal_ws_backend, baseline_coeff, tolerance)
            reasons.extend(f"coefficient limit exceeded: {field}" for field in footprint["guard_exceeded"])
            reuse = trace_reuse(numbered, args.traversal_ws_backend, args.traversal_cache_kib)
            row = {
                "profile": profile, "parameters": spec, "score": footprint["score"],
                "frontier": frontier, "reasons": reasons, "graph": geometry,
                "schedule": schedule, "working_set": footprint, "gs_psi_lru": reuse,
                "engine_seconds": engine_seconds, "wall_seconds": time.perf_counter() - beginning,
            }
            json_write(result_dir / "candidate.json", row)
            config = json.loads((result_dir / "config.json").read_text())
            config["traversal"] = {**settings, "profile": profile, "eligible": not reasons,
                                   "parameters": spec}
            json_write(result_dir / "config.json", config)
            rows.append(row)
        selected = choose_candidate(rows)
        if selected:
            best = campaign / "candidates" / selected["profile"]
            shutil.copytree(best / "polyMesh", campaign / "polyMesh")
            for name in ("permutation.npz", "config.json", "validation.json", "candidate.json"):
                shutil.copyfile(best / name, campaign / name)
        json_write(campaign / "traversal.json", {
            "settings": settings, "selected": selected["profile"] if selected else None,
            "ranked_eligible": [row["profile"] for row in sorted(rows, key=lambda r: (
                r["score"], r["profile"])) if not row["reasons"]],
            "objective": "equal weights for mean unique GS psi read/write lines divided by window rows",
            "original": {"working_set": original_ws, "schedule": original_schedule, "graph": original_graph},
            "candidates": rows, "wall_seconds": time.perf_counter() - start,
        })
        destination.mkdir()
        campaign.replace(destination)
    if not selected:
        raise ValueError(f"no eligible cache traversal candidate; see {destination / 'traversal.json'}")
