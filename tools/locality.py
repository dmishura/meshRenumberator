"""Objective and verification for the standalone local-search optimizer."""

import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from analyze_mesh_ordering import _build_ldu_adjacency_stream


def coefficient_metrics(mesh):
    """64-byte aligned logical streams of the benchmark's symmetric LDU GS."""
    counts = np.bincount(np.asarray(mesh.internal_owner), minlength=mesh.n_cells)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    nonempty = counts > 0
    first = starts[nonempty]
    last = first + counts[nonempty] - 1
    final_face = max(0, mesh.n_internal_faces - 1)

    def twice_per_row(width):
        span = last // width - first // width
        return final_face // width + int(span.sum()) + int(np.count_nonzero(span))

    upper = twice_per_row(8)
    addressing = twice_per_row(16)
    return {
        "coefficient_transitions": upper + addressing,
        "upper_cache_line_transitions": upper,
        "upperAddr_cache_line_transitions": addressing,
        "lower_cache_line_transitions": 0,
        "diag_cache_line_transitions": (mesh.n_cells - 1) // 8,
        "ownerStartAddr_cache_line_transitions": mesh.n_cells // 16,
        "line_bytes": 64,
    }


def full_metrics(mesh):
    """Use actual face order; attribute inter-row jumps to the next read row."""
    stream, degree = _build_ldu_adjacency_stream(
        mesh.n_cells, np.asarray(mesh.internal_owner), np.asarray(mesh.neighbour)
    )
    starts = np.r_[0, np.cumsum(degree, dtype=np.int64)]
    values = np.zeros(mesh.n_cells, dtype=np.int64)
    nonempty = degree > 0
    if stream.size:
        magnitudes = np.empty(stream.size, dtype=np.int64)
        magnitudes[0] = 0
        magnitudes[1:] = np.abs(np.diff(stream.astype(np.int64, copy=False)))
        magnitudes[magnitudes <= 1] = 0
        values[nonempty] = np.add.reduceat(magnitudes, starts[:-1][nonempty])
    return values, coefficient_metrics(mesh)


def match_metrics(mesh, reported):
    values, coefficients = full_metrics(mesh)
    mean = float(values.mean())
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    if (int(values.sum()) != reported["psi_total"]
            or not math.isclose(mean, reported["psi_mean"], abs_tol=1e-12)
            or not math.isclose(rms, reported["psi_rms"], abs_tol=1e-12)
            or coefficients["coefficient_transitions"] != reported["coefficient_transitions"]):
        raise ValueError("optimizer statistics do not match the actual LDU stream")
    reported["ldu_arrays"] = coefficients
    return values


def optimize(mesh, args):
    owner = np.asarray(mesh.internal_owner)
    neighbour = np.asarray(mesh.neighbour)
    # The mesh writer uses lexicographic (owner, neighbour) sorting. Check
    # the unmodified baseline agrees with that order before using the solver.
    if np.any((owner[1:] == owner[:-1]) & (neighbour[1:] < neighbour[:-1])):
        raise ValueError("locality requires source faces sorted by owner and neighbour")
    if not (0 <= args.seed < 2**64 and args.candidates >= 0 and args.window >= 1
            and args.patience >= 0 and args.verify_every >= 0
            and math.isfinite(args.balance_weight) and args.balance_weight >= 0
            and math.isfinite(args.coeff_tolerance) and args.coeff_tolerance >= 0):
        raise ValueError("invalid locality optimization settings")
    executable = Path(args.optimizer).resolve()
    if not executable.is_file():
        raise ValueError(f"optimizer missing: {executable}; build it with tools/build_locality.sh")
    weight = args.balance_weight if args.objective == "balanced" else 0
    with tempfile.TemporaryDirectory(prefix="mesh-locality-") as directory:
        graph = Path(directory) / "endpoints"
        output = Path(directory) / "new_to_old"
        with graph.open("wb") as stream:
            owner.astype("<i4", copy=False).tofile(stream)
            neighbour.astype("<i4", copy=False).tofile(stream)
        command = [str(executable), str(graph), str(output), *map(str, (
            mesh.n_cells, mesh.n_internal_faces, args.candidates, args.window, args.seed,
            args.patience, args.verify_every, weight, args.coeff_tolerance))]
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or "locality optimizer failed")
        report = json.loads(process.stdout)
        order = np.fromfile(output, dtype="<i4")
    metrics = [report["initial"], report["final"]] + [row["metrics"] for row in report["history"]]
    for entry in metrics:
        entry["objective"] = (entry["psi_total"] if args.objective == "total"
                              else entry["psi_mean"] + weight * entry["psi_rms"])
    match_metrics(mesh, report["initial"])
    report["settings"] = {
        key: getattr(args, key)
        for key in ("objective", "balance_weight", "coeff_tolerance", "candidates",
                    "window", "seed", "patience", "verify_every")
    }
    report["settings"]["initial"] = "original"
    return order, report
