"""Analysis and timing stages for an existing experiment."""
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
from statistics import median, stdev
import subprocess
import tempfile

import numpy as np


def analyze(args):
    from analyze_mesh_ordering import analyse_mesh
    from visualize_ih1_locality import (
        coefficient_locality_records, deterministic_outlier_masks,
        lower_percentile, natural_read_total_jumps, selected_components,
    )
    from ih_permutation import IHPermutation
    from mesh_renumber import json_write, read_poly_mesh, validate

    validate(args.experiment)
    path = args.experiment / "polyMesh"
    mesh = read_poly_mesh(path)
    values, stats, largest, nonseq, stream, row_starts, _ = natural_read_total_jumps(mesh)
    row_ids = np.arange(mesh.n_cells, dtype="<i4")
    perm = IHPermutation(Path("<identity>"), "LDU", 0, None, row_ids, row_ids)
    _, coeffs, _ = coefficient_locality_records(mesh, perm, "LDU")
    indices = stream.astype("i8")
    transitions = max(0, indices.size - 1)
    for size, name in ((8, "cacheLineTransitions"), (512, "pageTransitions")):
        stats[name] = int(np.count_nonzero(np.diff(indices // size)))
    stats["transitions"] = transitions
    for fraction, name in ((0.999, "p99.9"), (0.9999, "p99.99")):
        stats[name] = lower_percentile(values, fraction)
    masks = deterministic_outlier_masks(values)
    np.savez(args.experiment / "locality.npz", psiTotalJump_read=values,
             psiNonSequential=nonseq, psiMaxJump_read=largest,
             **{name: mask for name, (mask, _, _) in masks.items()})
    report = {"LDU_psi_read": stats, "LDU_coefficients": coeffs,
              "ordering": asdict(analyse_mesh(args.experiment.name, path, mesh=mesh))}
    if args.centres_export:
        centres = export_centres(args.centres_export, args.experiment)
        clusters = {}
        for name, (mask, threshold, rule) in masks.items():
            components = selected_components(mask, stream, row_starts, centres, values)
            count = int(mask.sum())
            total = int(values[mask].sum())
            largest_sum = int(values[components[0]["cellIds"]].sum()) if components else 0
            clusters[name] = {"threshold": threshold, "selection": rule,
                              "selected_cells": count, "component_count": len(components),
                              "largest_component_jump_fraction": largest_sum / total if total else 0,
                              "components": [{k: v for k, v in component.items() if k != "cellIds"}
                                             for component in components]}
        report["LDU_outliers"] = clusters
    json_write(args.experiment / "locality.json", report)


def export_centres(executable, experiment):
    from cell_centres import read_cell_centres

    path = experiment / "cell_centres.ihcentres"
    subprocess.run([str(executable.resolve()), str(experiment / "polyMesh"), str(path)], check=True)
    return read_cell_centres(path)


def benchmark(args):
    from mesh_renumber import digest, json_write, require, validate

    experiment = args.experiment.resolve()
    probe = args.probe.resolve()
    topology = validate(experiment)
    require(args.threads > 0 and args.sweeps > 0 and args.repetitions >= 2
            and args.widths and all(width > 0 for width in args.widths)
            and len(set(args.widths)) == len(args.widths), "invalid benchmark settings")
    paths = os.sched_getaffinity(0)
    config = json.loads((experiment / "config.json").read_text())
    result = Path(tempfile.mkdtemp(prefix="benchmark-", dir=experiment))
    env = os.environ | {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
                       "OMP_DYNAMIC": "FALSE"}
    settings = {"widths": args.widths, "sweeps": args.sweeps,
                "threads": args.threads, "warmup_sweeps": 8,
                "repetitions": args.repetitions, "affinity": sorted(paths),
                "probe_sha256": digest(probe), "cpu": os.uname().machine,
                "omp_environment": {k: v for k, v in env.items() if k.startswith("OMP_")}}
    json_write(result / "config.json", settings)
    # All experiments from this checkout share the same timing lock.
    lock = Path(__file__).resolve().parents[1] / "experiments" / ".benchmark.lock"
    lock.parent.mkdir(exist_ok=True)
    n = topology["internal_faces"]
    with tempfile.TemporaryDirectory(dir=result) as tmp, np.load(
        experiment / "permutation.npz"
    ) as maps, lock.open("a") as handle:
        cells = Path(tmp) / "cells"
        faces = Path(tmp) / "faces"
        maps["new_to_old"].tofile(cells)
        maps["new_face_to_old"][:n].tofile(faces)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [str(probe), config["source"], str(experiment / "polyMesh"),
                   str(cells), str(faces), str(args.threads), str(args.sweeps),
                   str(args.repetitions), *map(str, args.widths)]
        with (result / "probe.log").open("w") as stdout:
            completed = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, env=env)
    require(completed.returncode == 0, f"GS/IH1 probe failed: {result / 'probe.log'}")
    rows = []
    exact = []
    numerical = {}
    for record in csv.reader((result / "probe.log").read_text().splitlines()):
        if not record:
            continue
        if record[0] == "sample":
            method, team, index, time, setup = record[1:]
            rows.append({"solver": method, "threads": int(team), "repetition": int(index),
                         "ms_per_sweep": float(time), "setup_ms": float(setup)})
        elif record[0] == "exact":
            exact.append({"solver": record[1], "threads": int(record[2]),
                          "one_sweep_exact": True, "converged_max_abs_delta": float(record[3])})
        elif record[0] == "validation":
            keys = ("cell_centre_max_abs_error", "gs_residual_max_abs",
                    "gs_exact_solution_max_abs_error", "gs_cross_numbering_max_abs_delta",
                    "gs_one_sweep_cross_numbering_max_abs_delta")
            numerical = dict(zip(keys, map(float, record[1:])))
    groups = {}
    for row in rows:
        key = f"{row['solver']}/{row['threads']}T"
        groups.setdefault(key, []).append(row["ms_per_sweep"])
    expected = {"OriginalGS/1T"} | {f"IH1W{width}/1T" for width in args.widths}
    if args.threads > 1:
        expected.update(f"IndexKahn{width}Blocked64OMP/{team}T"
                        for width in args.widths for team in (1, args.threads))
    require(set(groups) == expected and numerical and
            len(exact) == len(expected) - 1 and
            all(len(times) == args.repetitions for times in groups.values()),
            f"incomplete benchmark: {result / 'probe.log'}")
    summary = {key: {"median_ms_per_sweep": median(times), "stdev_ms_per_sweep": stdev(times),
                     "min_ms_per_sweep": min(times), "max_ms_per_sweep": max(times),
                     "speedup_vs_same_numbering_gs": median(groups["OriginalGS/1T"]) / median(times)}
               for key, times in groups.items()}
    with (result / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_write(result / "summary.json", {"settings": settings, "topology": topology,
                                       "algorithm": config["algorithm"],
                                       "numerical": numerical, "exact": exact, "timings": summary})
    print(result)


def compare(args):
    from mesh_renumber import require

    summaries = [json.loads((path / "summary.json").read_text()) for path in args.runs]
    first = summaries[0]
    require(first["algorithm"] == "identity", "pass identity benchmark first")
    columns = list(first["timings"])
    localities = []
    for path, summary in zip(args.runs, summaries):
        require(summary["settings"] == first["settings"] and
                summary["topology"]["source_topology_sha256"] ==
                first["topology"]["source_topology_sha256"], "benchmark inputs differ")
        locality = path.parent / "locality.json"
        localities.append(json.loads(locality.read_text()) if locality.is_file() else None)

    include_locality = all(localities)
    extra = ["LDU psi total", "LDU psi P99", "LDU psi max"] if include_locality else []
    headers = ["Numbering", *columns, *extra]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for summary, locality in zip(summaries, localities):
        items = [summary["algorithm"]]
        for key in columns:
            timing = summary["timings"][key]
            ms = timing["median_ms_per_sweep"]
            speedup = first["timings"][key]["median_ms_per_sweep"] / ms
            items.append(f"{ms:.3f} ms ({speedup:.2f}×)")
        if include_locality:
            stats = locality["LDU_psi_read"]
            items.extend(f"{stats[key]:,.0f}" for key in ("totalJumpMagnitude", "p99", "max"))
        print("| " + " | ".join(items) + " |")

    if args.tomography:
        from visualize_ih1_locality import run_natural_read_comparison
        require(len(args.runs) == 2 and args.centres_export,
                "tomography requires two runs and --centres-export")
        labels = [f"{index}-{summary['algorithm']}" for index, summary in enumerate(summaries)]
        experiments = dict(zip(labels, (path.parent for path in args.runs)))
        for experiment in experiments.values():
            export_centres(args.centres_export, experiment)
        run_natural_read_comparison(
            {name: path / "polyMesh" for name, path in experiments.items()},
            {name: path / "cell_centres.ihcentres" for name, path in experiments.items()},
            args.tomography, 2048)
