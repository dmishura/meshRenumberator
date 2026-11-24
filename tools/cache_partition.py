"""Connected partitioning and multi-scale selection in natural GS order."""

from collections import deque
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from locality import coefficient_metrics, full_metrics
from openfoam_mesh_reader import PolyMeshTopology
from working_set import (
    WINDOW_COLUMNS, distribution, gs_psi_trace, row_offsets, summarize_reuse, trace_metrics,
    window_summary,
)

WINDOWS = (32, 128, 512, 1024, 2048)
COEFF_FIELDS = ("coefficient_transitions", "upper_cache_line_transitions",
                "upperAddr_cache_line_transitions", "diag_cache_line_transitions",
                "lower_cache_line_transitions", "ownerStartAddr_cache_line_transitions")


def grow_partitions(n_cells, owner, neighbour, budget_bytes, seed, backend):
    """Return per-cell membership, per-group metadata, and two local orders."""
    owner, neighbour = np.asarray(owner), np.asarray(neighbour)
    if (owner.ndim != 1 or owner.shape != neighbour.shape or not 0 <= seed < 2**64
            or not 0 < n_cells < 2**31 - 1 or not 0 < budget_bytes < 2**63
            or (owner.size and (owner.dtype.kind not in "iu" or neighbour.dtype.kind not in "iu"
                                or np.any(owner < 0) or np.any(owner >= neighbour)
                                or np.any(neighbour >= n_cells)))):
        raise ValueError("invalid partition graph, seed, or budget")
    backend = Path(backend).resolve()
    if not backend.is_file():
        raise ValueError(f"missing {backend}; build with bash tools/build_cache_partition.sh")
    with tempfile.TemporaryDirectory(prefix="mesh-cache-partition-") as directory:
        src = Path(directory) / "endpoints"
        dst = Path(directory) / "groups"
        with src.open("wb") as stream:
            owner.astype(np.int32, copy=False).tofile(stream)
            neighbour.astype(np.int32, copy=False).tofile(stream)
        process = subprocess.run([str(backend), str(src), str(dst), str(n_cells),
                                  str(owner.size), str(budget_bytes), str(seed)],
                                 capture_output=True, text=True)
        if process.returncode:
            raise ValueError(process.stderr.strip() or "partition backend failed")
        with dst.open("rb") as stream:
            n = int(np.fromfile(stream, dtype=np.int32, count=1).item())
            result = {name: np.fromfile(stream, dtype=np.int32, count=n_cells)
                      for name in ("membership", "traversal", "local_rcm")}
            result["offsets"] = np.fromfile(stream, dtype=np.int32, count=n + 1)
            result["footprint_cells"] = np.fromfile(stream, dtype=np.int32, count=n)
            result["estimated_bytes"] = np.fromfile(stream, dtype=np.int64, count=n)
            if (n < 1 or n > n_cells or stream.read(1)
                    or any(result[k].size != n_cells for k in ("membership", "traversal", "local_rcm"))
                    or result["offsets"].size != n + 1 or result["estimated_bytes"].size != n
                    or result["footprint_cells"].size != n):
                raise ValueError("invalid partition backend output")

    offsets = result["offsets"]
    sizes = np.diff(offsets)
    if offsets[0] != 0 or offsets[-1] != n_cells or np.any(sizes <= 0):
        raise ValueError("partitions do not cover all cells")
    for name in ("traversal", "local_rcm"):
        order = result[name]
        if (np.any((order < 0) | (order >= n_cells))
                or np.unique(order).size != n_cells
                or not np.array_equal(result["membership"][order], np.repeat(np.arange(n), sizes))):
            raise ValueError(f"invalid partition membership or local ordering ({name})")
    overflow = result["estimated_bytes"] > budget_bytes
    if np.any(overflow & (sizes != 1)):
        raise ValueError("a non-singleton partition exceeded its graph budget")
    return result


def build_coarse_graph(membership, n_groups, owner, neighbour):
    a = membership[owner].astype(np.int64)
    b = membership[neighbour].astype(np.int64)
    cross = a != b
    lower, upper = np.minimum(a[cross], b[cross]), np.maximum(a[cross], b[cross])
    edges, counts = np.unique(lower * n_groups + upper, return_counts=True)
    return edges // n_groups, edges % n_groups, counts


def order_partitions(n_groups, edge_a, edge_b, weights, traversal):
    if traversal not in {"breadth", "depth"}:
        raise ValueError("unknown coarse traversal")
    rows = np.concatenate((edge_a, edge_b))
    columns = np.concatenate((edge_b, edge_a))
    weight = np.tile(weights, 2)
    degree = np.bincount(rows, minlength=n_groups)
    offsets = row_offsets(degree)
    neighbours = columns[np.lexsort((columns, -weight, rows))]
    visited = np.zeros(n_groups, dtype=bool)
    order = []
    for root in np.argsort(degree, kind="stable").tolist():
        if visited[root]:
            continue
        frontier = deque([root])
        if traversal == "breadth":
            visited[root] = True
        while frontier:
            group = frontier.popleft() if traversal == "breadth" else frontier.pop()
            if traversal == "depth":
                if visited[group]:
                    continue
                visited[group] = True
            order.append(group)
            adjacent = neighbours[offsets[group]:offsets[group + 1]].tolist()
            if traversal == "breadth":
                for other in adjacent:
                    if not visited[other]:
                        visited[other] = True
                        frontier.append(other)
            else:
                frontier.extend(other for other in reversed(adjacent) if not visited[other])
    return np.asarray(order, dtype=np.int32)


def number_cells(order, local_order, offsets):
    starts = row_offsets(np.diff(offsets)[order])
    gather = np.arange(starts[-1]) + np.repeat(offsets[order] - starts[:-1], np.diff(starts))
    return local_order[gather].astype("<i4", copy=False)


def candidate_mesh(mesh, new_to_old):
    from mesh_renumber import inverse, remap

    labels = inverse(new_to_old, mesh.n_cells)
    owner, neighbour, _, _ = remap(mesh, labels, False)
    return PolyMeshTopology(mesh.poly_mesh_dir, owner, neighbour,
                            mesh.n_cells, mesh.n_faces, mesh.n_points)


def evaluate(mesh, backend, baseline, tolerance):
    owner, neighbour = np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour)
    offsets, accesses = gs_psi_trace(owner, neighbour, mesh.n_cells)
    windows = tuple(k for k in WINDOWS if k <= mesh.n_cells) or (mesh.n_cells,)
    measured, _ = trace_metrics(offsets, accesses, windows, None, backend)
    weights = {k: 1 / len(windows) for k in windows}
    index = WINDOW_COLUMNS.index("unique_lines")
    score = sum(weights[k] * measured[k][:, index].mean() / k for k in windows)
    coefficients = coefficient_metrics(mesh)
    over = [name for name in COEFF_FIELDS if coefficients[name] > (1 + tolerance) * baseline[name]]
    return {"score": float(score), "weights": weights, "coefficient_locality": coefficients,
            "guard_exceeded": over, "windows": list(window_summary("gs_psi_read_write", measured, None))}


def footprint_report(source, numbered, result, coarse, group_order, new_to_old):
    n = source.n_cells
    f = source.n_internal_faces
    old_owner, old_neighbour = np.asarray(source.internal_owner), np.asarray(source.neighbour)
    group = result["membership"]
    group_labels = np.empty_like(group_order)
    group_labels[group_order] = np.arange(group_order.size)
    group_new = group_labels[group]
    sizes = np.diff(result["offsets"])[group_order]
    start = row_offsets(sizes)
    labels = np.empty(n, dtype=np.int32)
    labels[new_to_old] = np.arange(n, dtype=np.int32)
    lines = (n + 7) // 8
    cell_lines = labels // 8
    keys = np.concatenate((
        group_new.astype(np.int64) * lines + cell_lines,
        group_new[old_owner].astype(np.int64) * lines + cell_lines[old_neighbour],
        group_new[old_neighbour].astype(np.int64) * lines + cell_lines[old_owner],
    ))
    actual = np.bincount(np.unique(keys) // lines, minlength=group_order.size)
    crosses = group_new[old_owner] != group_new[old_neighbour]
    cross_a, cross_b = old_owner[crosses], old_neighbour[crosses]
    boundary_edges = np.bincount(np.r_[group_new[cross_a], group_new[cross_b]],
                                 minlength=group_order.size)
    boundary_cells = np.zeros(n, dtype=bool)
    boundary_cells[labels[cross_a]] = True
    boundary_cells[labels[cross_b]] = True
    boundary_sizes = np.add.reduceat(boundary_cells.astype(np.int64), start[:-1])
    face_start = row_offsets(np.bincount(np.asarray(numbered.internal_owner), minlength=n))[start]
    faces = np.diff(face_start)
    # A separate aligned interval for each contiguous array, in cache lines.
    def line_counts(first, last, width):
        return np.where(last > first, (last - 1) // width - first // width + 1, 0)

    upper_lines = line_counts(face_start[:-1], face_start[1:], 8)
    address_lines = line_counts(face_start[:-1], face_start[1:], 16)
    diag_lines = line_counts(start[:-1], start[1:], 8)
    offset_lines = line_counts(start[:-1], start[1:] + 1, 16)
    estimated = result["estimated_bytes"][group_order]
    estimated_cells = result["footprint_cells"][group_order]
    a, b, weights = coarse
    summary = {
        "count": group_order.size,
        "crossing_edges": int(crosses.sum()),
        "crossing_edge_fraction": float(crosses.mean()) if f else 0,
        "boundary_cells": int(boundary_cells.sum()),
        "coarse_edges": weights.size,
        "coarse_bandwidth": int(np.abs(group_labels[a] - group_labels[b]).max(initial=0)),
        "size": distribution(sizes),
        "boundary_edges_per_partition": distribution(boundary_edges),
        "boundary_cells_per_partition": distribution(boundary_sizes),
        "estimated_psi_and_graph_metadata_bytes": distribution(estimated),
        "actual_psi_bytes": distribution(actual * 64),
        "actual_upper_bytes": distribution(upper_lines * 64),
        "actual_upperAddr_bytes": distribution(address_lines * 64),
        "actual_psi_plus_ldu_structure_bytes": distribution(
            64 * (actual + upper_lines + address_lines + diag_lines + offset_lines)),
        "crossing_edge_distance": distribution(np.abs(labels[cross_a] - labels[cross_b])),
    }
    return summary, boundary_cells, {
        "partition_old_cell": group, "partition_new_cell": group_new[new_to_old],
        "coarse_order": group_order, "local_offsets": start, "estimated_bytes": estimated,
        "estimated_psi_cells": estimated_cells, "psi_lines": actual,
        "upper_lines": upper_lines, "upperAddr_lines": address_lines,
        "diag_lines": diag_lines, "ownerStartAddr_lines": offset_lines,
        "boundary_edges": boundary_edges, "boundary_cells": boundary_sizes,
        "coarse_edges_a": a, "coarse_edges_b": b, "coarse_weights": weights,
    }


def seam_report(mesh, boundary):
    from analyze_mesh_ordering import _build_ldu_adjacency_stream

    values, _ = full_metrics(mesh)
    stats = distribution(values, (99, 99.9, 99.99))
    owner, neighbour = np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour)
    stream, degree = _build_ldu_adjacency_stream(mesh.n_cells, owner, neighbour)
    starts = row_offsets(degree)
    near = boundary.copy()
    near[owner[boundary[neighbour]]] = True
    near[neighbour[boundary[owner]]] = True
    selected = values > stats["p99.99"]
    seen = np.zeros(mesh.n_cells, dtype=bool)
    sizes = []
    for cell in np.flatnonzero(selected).tolist():
        if seen[cell]:
            continue
        seen[cell] = True
        queue = [cell]
        size = 0
        while queue:
            row = queue.pop()
            size += 1
            for other in stream[starts[row]:starts[row + 1]].tolist():
                if selected[other] and not seen[other]:
                    seen[other] = True
                    queue.append(other)
        sizes.append(size)
    masks = {}
    for quantile in ("p99", "p99.9", "p99.99"):
        mask = values > stats[quantile]
        count = int(mask.sum())
        masks[quantile] = {"count": count,
                           "boundary_incident_fraction": float(boundary[mask].mean()) if count else 0,
                           "within_one_edge_of_boundary_fraction": float(near[mask].mean()) if count else 0}
    return {"psiTotalJump_read": {**stats, "total": int(values.sum())},
            "outliers": masks, "p99.99_component_sizes": sorted(sizes, reverse=True),
            "largest_p99.99_component": max(sizes, default=0),
            "boundary_fraction_all_cells": float(boundary.mean()),
            "within_one_edge_of_boundary_fraction_all_cells": float(near.mean())}


def choose_partition(mesh, args):
    clock = time.perf_counter
    start = clock()
    tolerance = args.partition_coeff_tolerance
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("partition coefficient tolerance must be finite and non-negative")
    owner, neighbour = np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour)
    budget = args.partition_budget_kib * 1024
    groups = grow_partitions(mesh.n_cells, owner, neighbour, budget, args.partition_seed,
                             args.partition_backend)
    growth_seconds = clock() - start
    offsets = groups["offsets"]
    n_groups = offsets.size - 1
    coarse = build_coarse_graph(groups["membership"], n_groups, owner, neighbour)
    locals_ = ("traversal", "rcm") if args.partition_local_order == "auto" else (args.partition_local_order,)
    globals_ = ("breadth", "depth") if args.partition_coarse_order == "auto" else (args.partition_coarse_order,)
    reference = coefficient_metrics(mesh)

    baseline = evaluate(mesh, args.partition_ws_backend, reference, tolerance)
    candidates = []
    best = None
    best_order = None
    best_coarse = None
    for strategy in globals_:
        order = order_partitions(n_groups, *coarse, strategy)
        for local in locals_:
            local_order = groups["local_rcm" if local == "rcm" else "traversal"]
            permutation = number_cells(order, local_order, offsets)
            numbering = candidate_mesh(mesh, permutation)
            trial = evaluate(numbering, args.partition_ws_backend, reference, tolerance)
            trial.update({"local_order": local, "coarse_order": strategy})
            candidates.append(trial)
            if trial["guard_exceeded"]:
                continue
            if best is None or trial["score"] < best["score"]:
                best, best_order, best_coarse = trial, permutation, order
    if best is None:
        brief = [{"local": trial["local_order"], "coarse": trial["coarse_order"],
                  "exceeded": trial["guard_exceeded"],
                  "array_values": trial["coefficient_locality"]} for trial in candidates]
        raise ValueError(f"no cache-partition candidate met the array-wise tolerance: {json.dumps(brief)}")

    stage_seconds = clock() - start
    selected = candidate_mesh(mesh, best_order)
    partition, boundary, arrays = footprint_report(mesh, selected, groups, coarse, best_coarse, best_order)
    partition["singleton_budget_overflows"] = int(np.count_nonzero(groups["estimated_bytes"] > budget))
    report = {
        "settings": {"initial": "original", "budget_kib": args.partition_budget_kib,
                     "seed": args.partition_seed, "local_order": args.partition_local_order,
                     "coarse_order": args.partition_coarse_order, "coefficient_tolerance": tolerance,
                     "lru": args.partition_lru, "lru_cache_kib": list(args.partition_cache_kib)},
        "objective": "equal weights for mean GS psi read/write unique lines divided by row count K",
        "baseline": baseline, "candidates": candidates, "selected": best, "partition": partition,
        "seams": seam_report(selected, boundary), "growth_seconds": growth_seconds,
        "growth_and_candidate_selection_seconds": stage_seconds,
    }
    if args.partition_lru:
        o, i = gs_psi_trace(np.asarray(selected.internal_owner), np.asarray(selected.neighbour),
                            selected.n_cells)
        # A temporary one-window result avoids re-evaluating the full window set.
        _, histogram = trace_metrics(o, i, (selected.n_cells,), None, args.partition_ws_backend,
                                     reuse=True)
        estimate = summarize_reuse(histogram, args.partition_cache_kib)
        total, cold = estimate["total_accesses"], estimate["cold_accesses"]
        estimate["policy"] = "fully associative, write allocate, one cold GS sweep, psi only"
        for capacity in estimate["capacities"]:
            hits = int(histogram[1:1 + capacity["lines"]].sum())
            capacity["cold_misses"] = cold
            capacity["capacity_misses"] = total - cold - hits
        report["gs_psi_lru"] = estimate
    report["wall_seconds"] = clock() - start
    return best_order, report, arrays
