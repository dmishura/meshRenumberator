"""Diagnostics for natural-order LDU adjacency and the benchmark's GS reads."""

import csv
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

WINDOW_COLUMNS = ("unique_lines", "unique_pages", "span_elements", "span_bytes",
                  "line_density", "line_touches", "repeated_line_touches", "reuse_factor")
DEFAULT_WINDOWS = (8, 32, 128, 512, 1024, 4096)
DEFAULT_CAPACITIES = (32, 64, 256, 1024, 4096)


def distribution(values, percentiles=(50, 90, 99, 99.9)):
    if not values.size:
        return {"mean": 0, **{f"p{p}": 0 for p in percentiles}, "max": 0}
    loc = [int(p / 100 * (values.size - 1)) for p in percentiles]
    ordered = np.partition(values, loc)
    return {"mean": float(values.mean()), **{
        f"p{p}": ordered[i].item() for p, i in zip(percentiles, loc)},
        "max": values.max().item()}


def graph_metrics(n_cells, owner, neighbour):
    owner = np.asarray(owner, dtype=np.int64)
    neighbour = np.asarray(neighbour, dtype=np.int64)
    if (n_cells < 0 or owner.ndim != 1 or owner.shape != neighbour.shape or
            np.any(owner < 0) or np.any(neighbour >= n_cells) or np.any(owner >= neighbour)):
        raise ValueError("invalid upper-triangular graph endpoints")
    edge_distance = neighbour - owner
    band = int(edge_distance.max(initial=0))
    low = np.full(n_cells, n_cells, dtype=np.int64)
    high = np.full(n_cells, -1, dtype=np.int64)
    relative = np.zeros(n_cells, dtype=np.int64)
    for row, column in ((owner, neighbour), (neighbour, owner)):
        np.minimum.at(low, row, column)
        np.maximum.at(high, row, column)
        np.maximum.at(relative, row, edge_distance)
    span = np.maximum(0, high - low)
    if int(span.max(initial=0)) > 2 * band or int(relative.max(initial=0)) > band:
        raise ValueError("row distance bound violated")
    edge = distribution(edge_distance)
    # For the reader's signed 32-bit labels and face counts this is bounded
    # by 2**62. Check the wider synthetic API before summing in fixed width.
    if band * edge_distance.size <= np.iinfo(np.int64).max:
        edge["sum"] = int(edge_distance.sum(dtype=np.int64))
    else:
        edge["sum"] = sum(map(int, edge_distance))
    return ({"cells": n_cells, "internal_faces": owner.size, "bandwidth": band,
             "normalized_bandwidth": band / n_cells if n_cells else 0,
             "row_bounds_verified": True, "edge_distance": edge},
            {"neighbour_span": distribution(span), "row_relative_max": distribution(relative)})


def row_offsets(counts):
    return np.r_[0, np.cumsum(counts, dtype=np.int64)]


def gs_psi_trace(owner, neighbour, n_cells):
    """Natural GS reads outgoing neighbours, then writes the row's psi."""
    starts = row_offsets(np.bincount(owner, minlength=n_cells))
    offsets = starts + np.arange(n_cells + 1, dtype=np.int64)
    accesses = np.empty(offsets[-1], dtype=np.int32)
    accesses[np.arange(neighbour.size) + owner] = neighbour
    accesses[offsets[1:] - 1] = np.arange(n_cells, dtype=np.int32)
    return offsets, accesses



def trace_metrics(offsets, indices, windows, stride, backend, reuse=False, entry_bytes=8):
    offsets = np.asarray(offsets)
    indices = np.asarray(indices)
    if (offsets.ndim != 1 or not offsets.size or indices.ndim != 1
            or offsets.dtype.kind not in "iu" or indices.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != indices.size
            or np.any(offsets[1:] < offsets[:-1]) or np.any(indices < 0)):
        raise ValueError("invalid trace arrays")
    if not windows or any(k <= 0 for k in windows) or (stride is not None and stride <= 0):
        raise ValueError("windows and stride must be positive")
    executable = Path(backend).resolve()
    if not executable.is_file():
        raise ValueError(f"missing {executable}; build with bash tools/build_working_set.sh")
    with tempfile.TemporaryDirectory(prefix="mesh-working-set-") as directory:
        path = Path(directory) / "trace"
        prefix = Path(directory) / "result"
        with path.open("wb") as stream:
            offsets.astype(np.uint64, copy=False).tofile(stream)
            indices.astype(np.uint64, copy=False).tofile(stream)
        command = [str(executable), str(path), str(prefix), str(offsets.size - 1),
                   str(indices.size), str(entry_bytes), str(int(reuse))]
        command.extend(f"{k}:{stride or max(1, k // 4)}" for k in windows)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or "working-set backend failed")
        results = {k: np.fromfile(f"{prefix}-{k}", dtype=np.float64)
                   .reshape(-1, len(WINDOW_COLUMNS)) for k in windows}
        histogram = np.fromfile(f"{prefix}-reuse", dtype=np.uint64) if reuse else None
    return results, histogram


def summarize_reuse(histogram, capacities):
    total = int(histogram.sum())
    cold = int(histogram[0])
    repeats = total - cold
    cumulative = np.cumsum(histogram[1:], dtype=np.uint64)
    percentiles = {
        f"p{p}": int(np.searchsorted(cumulative, int(p / 100 * (repeats - 1)) + 1))
        if repeats else None for p in (50, 90, 99)
    }
    return {"cold_accesses": cold, "total_accesses": total,
            "repeat_distance_quantiles": percentiles,
            "definition": "exact distinct-line stack distance, one stream from cold",
            "estimate": "fully associative LRU-style fraction over all accesses, including cold",
            "capacities": [{"kib": kib, "lines": kib * 16,
                            "fraction_below": int(histogram[1:1 + kib * 16].sum()) / total
                            if total else 0} for kib in capacities]}


def contiguous_windows(offsets, windows, stride, entry_bytes, passes):
    """Grouped owner faces form contiguous address ranges in a window."""
    n = len(offsets) - 1
    results = {}
    for k in windows:
        rows = np.arange(0, max(0, n - k + 1), stride or max(1, k // 4))
        first, stop = offsets[rows], offsets[rows + k]
        span = stop - first
        last = np.maximum(first, stop - 1)
        lines = np.where(span, last // (64 // entry_bytes) - first // (64 // entry_bytes) + 1, 0)
        pages = np.where(span, last // (4096 // entry_bytes) - first // (4096 // entry_bytes) + 1, 0)
        touches = span * passes
        results[k] = np.column_stack((lines, pages, span, span * entry_bytes, span > 0,
                                      touches, touches - lines,
                                      touches / np.maximum(lines, 1)))
    return results


def owner_starts_windows(n, windows, stride):
    results = {}
    for k in windows:
        first = np.arange(0, max(0, n - k + 1), stride or max(1, k // 4))
        last = first + k
        lines = last // 16 - first // 16 + 1
        pages = last // 1024 - first // 1024 + 1
        span = np.full(first.size, k + 1)
        touches = np.full(first.size, k * 2)
        results[k] = np.column_stack((lines, pages, span, span * 4, np.ones(first.size),
                                      touches, touches - lines, touches / lines))
    return results


def repeated_face_transitions(offsets, line_entries):
    counts = np.diff(offsets)
    first = offsets[:-1][counts > 0]
    last = offsets[1:][counts > 0] - 1
    crossings = last // line_entries - first // line_entries
    return (max(0, int(offsets[-1]) - 1) // line_entries +
            int(crossings.sum()) + int(np.count_nonzero(crossings)))


def window_summary(stream, results, stride):
    for k, values in results.items():
        for index, metric in enumerate(WINDOW_COLUMNS):
            yield {"stream": stream, "window_rows": k, "stride_rows": stride or max(1, k // 4),
                   "windows": values.shape[0], "metric": metric,
                   **distribution(values[:, index], (50, 90, 99))}


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_locality(args):
    from analyze_mesh_ordering import _build_ldu_adjacency_stream
    from mesh_renumber import json_write, read_poly_mesh

    mesh = read_poly_mesh(args.mesh)
    n = mesh.n_cells
    owner = np.asarray(mesh.internal_owner)
    neighbour = np.asarray(mesh.neighbour)
    graph, rows = graph_metrics(n, owner, neighbour)
    graph["topology_sha256"] = mesh.topology_checksum()
    strides = args.stride
    windows = args.window_sizes
    offsets = row_offsets(np.bincount(owner, minlength=n))
    adjacency, degree = _build_ldu_adjacency_stream(n, owner, neighbour)

    # bPrime[cell] is read before the gather loop. Each subsequent update to
    # bPrime[neighbour] is one logical read-modify-write touch.
    bprime_offsets = offsets + np.arange(n + 1, dtype=np.int64)
    bprime = np.empty(bprime_offsets[-1], dtype=np.int32)
    bprime[bprime_offsets[:-1]] = np.arange(n, dtype=np.int32)
    bprime[np.arange(neighbour.size) + owner.astype(np.int64) + 1] = neighbour
    working = []
    reuse = {}
    combined_fields = ("unique_lines", "unique_pages", "line_touches", "repeated_line_touches")
    combined_columns = [WINDOW_COLUMNS.index(name) for name in combined_fields]
    combined = {}

    def add_combined(result):
        for k, data in result.items():
            fields = data[:, combined_columns]
            if k not in combined:
                combined[k] = fields.copy()
            else:
                combined[k] += fields

    psi_rw = gs_psi_trace(owner, neighbour, n)
    for name, row_starts, indices in (
        ("psi_adjacency", row_offsets(degree), adjacency),
        ("psi_gs_read", offsets, neighbour),
        ("psi_gs_read_write", *psi_rw),
        ("bPrime_gs", bprime_offsets, bprime),
    ):
        measured, histogram = trace_metrics(row_starts, indices, windows, strides,
                                            args.backend, args.reuse_distance)
        working.extend(window_summary(name, measured, strides))
        if name in {"psi_gs_read_write", "bPrime_gs"}:
            add_combined(measured)
        if histogram is not None:
            reuse[name] = summarize_reuse(histogram, args.cache_kib)

    coefficient = []
    for name, starts, widths, passes in (
        ("upper", offsets, 8, 2), ("upperAddr", offsets, 4, 2),
        ("diag", np.arange(n + 1), 8, 1), ("lower", np.zeros(n+1, dtype=np.int64), 8, 1),
    ):
        measured = contiguous_windows(starts, windows, strides, widths, passes)
        coefficient.extend(window_summary(name, measured, strides))
        add_combined(measured)
    starts_windows = owner_starts_windows(n, windows, strides)
    coefficient.extend(window_summary("ownerStartAddr", starts_windows, strides))
    add_combined(starts_windows)
    for k, values in combined.items():
        for index, name in enumerate(combined_fields):
            working.append({"stream": "combined_gs_arrays", "window_rows": k,
                            "stride_rows": strides or max(1, k // 4),
                            "windows": values.shape[0], "metric": name,
                            **distribution(values[:, index], (50, 90, 99))})
    f = neighbour.size
    trace = {
        "model": "Original GS symmetric L0 kernel, base-aligned arrays",
        "arrays": {
            "upper": {"touches": 2*f, "line_transitions": repeated_face_transitions(offsets, 8)},
            "upperAddr": {"touches": 2*f, "line_transitions": repeated_face_transitions(offsets, 16)},
            "lower": {"touches": 0, "line_transitions": 0, "note": "no separate lower storage"},
            "diag": {"touches": n, "line_transitions": max(0, n-1)//8},
            "ownerStartAddr": {"touches": 2*n, "line_transitions": n//16,
                               "note": "one logical read of each of the two row bounds"},
        },
        "face_access_order": "owner rows ascending, their consecutive face IDs twice forward",
        "face_counts_per_owner": distribution(np.diff(offsets)),
    }
    if args.face_permutation:
        with np.load(args.face_permutation) as archive:
            old_ids = archive["new_face_to_old"][:f].astype(np.int64)
            flips = archive["face_flipped"][:f]
        if (old_ids.size != f or flips.size != f or
                not np.array_equal(np.sort(old_ids), np.arange(f))):
            raise ValueError("invalid internal face permutation")
        displacement = np.abs(old_ids - np.arange(f, dtype=np.int64))
        trace["physical_face_order"] = {
            "positions_preserved": int(np.count_nonzero(displacement == 0)),
            "flipped": int(np.count_nonzero(flips)),
            "face_id_displacement": distribution(displacement),
            "adjacent_original_face_id_distance": distribution(np.abs(np.diff(old_ids))),
        }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "bandwidth.json", graph)
    json_write(output / "row_span.json", rows)
    json_write(output / "coefficient_trace.json", trace)
    json_write(output / "config.json", {
        "mesh": str(args.mesh.resolve()), "windows": windows, "stride_override": strides,
        "window_policy": "full windows starting at 0, default stride max(1,K//4)",
        "address_model": "64-byte lines and 4 KiB pages; logical, aligned per array",
        "traces": {
            "psi_adjacency": "incoming then outgoing, in face ID order by group",
            "psi_gs_read": "only outgoing neighbours in face ID order; excludes writes",
            "psi_gs_read_write": "outgoing neighbours in face ID order, then a write to the row label",
            "combined_gs_arrays": "sum across separate psi read/write, bPrime, and LDU address spaces",
            "bPrime_gs": "one self read then outgoing neighbour updates; excludes source copy",
        },
    })
    write_csv(output / "working_set.csv", working)
    write_csv(output / "coefficient_locality.csv", coefficient)
    if reuse:
        json_write(output / "reuse_distance.json", reuse)
    print(output)
