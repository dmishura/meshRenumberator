"""Tests for the compiled locality optimizer and its mesh metrics."""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
WFGS_TOOLS = Path(os.environ.get("WFGS_ROOT", ROOT.parent / "WavefrontGaussSeidel")) / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(WFGS_TOOLS))

from mesh_renumber import remap  # noqa: E402
from openfoam_mesh_reader import PolyMeshTopology  # noqa: E402
from locality import full_metrics, optimize  # noqa: E402
from visualize_ih1_locality import natural_read_total_jumps  # noqa: E402


def synthetic_mesh() -> PolyMeshTopology:
    """Lexicographic graph with a disconnected component and isolated cell."""
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
        (6, 7),
    ]
    return PolyMeshTopology(
        poly_mesh_dir=Path("<synthetic>"),
        owner=array("i", (owner for owner, _ in edges)),
        neighbour=array("i", (neighbour for _, neighbour in edges)),
        n_cells=8,
        n_faces=len(edges),
        n_points=0,
    )


def optimizer_path() -> Path | None:
    configured = os.environ.get("LOCALITY_OPTIMIZER")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() and os.access(path, os.X_OK) else None
    default = ROOT / "experiments/build/locality_optimizer"
    return default if default.is_file() and os.access(default, os.X_OK) else None


def make_args(executable: Path, **overrides: object):
    from argparse import Namespace

    values = {
        "optimizer": str(executable),
        "objective": "balanced",
        "balance_weight": 1.0,
        "coeff_tolerance": 0.25,
        "candidates": 24,
        "window": 8,
        "seed": 37,
        "patience": 24,
        "verify_every": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def remapped_mesh(mesh: PolyMeshTopology, new_to_old: np.ndarray) -> PolyMeshTopology:
    old_to_new = np.empty_like(new_to_old, dtype=np.int32)
    old_to_new[new_to_old] = np.arange(mesh.n_cells, dtype=np.int32)
    owner, neighbour, _, _ = remap(mesh, old_to_new, identity=False)
    return PolyMeshTopology(
        poly_mesh_dir=Path("<remapped>"),
        owner=array("i", map(int, owner)),
        neighbour=array("i", map(int, neighbour)),
        n_cells=mesh.n_cells,
        n_faces=mesh.n_faces,
        n_points=mesh.n_points,
    )


class LocalityMetricTest(unittest.TestCase):
    def test_full_psi_metric_matches_existing_tomography(self) -> None:
        mesh = synthetic_mesh()
        jumps, _ = full_metrics(mesh)
        reference, *_ = natural_read_total_jumps(mesh)
        np.testing.assert_array_equal(jumps, reference)
        self.assertEqual(jumps.shape, (mesh.n_cells,))


@unittest.skipUnless(optimizer_path(), "compiled optimizer unavailable; set LOCALITY_OPTIMIZER")
class LocalityOptimizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.executable = optimizer_path()
        assert cls.executable is not None

    def test_reported_metrics_match_full_remapped_mesh(self) -> None:
        mesh = synthetic_mesh()
        args = make_args(self.executable, verify_every=1)
        new_to_old, stats = optimize(mesh, args)
        self.assertEqual(new_to_old.dtype, np.dtype(np.int32))
        self.assertEqual(sorted(new_to_old.tolist()), list(range(mesh.n_cells)))

        initial_psi, initial_coeff = full_metrics(mesh)
        final_mesh = remapped_mesh(mesh, new_to_old)
        final_psi, final_coeff = full_metrics(final_mesh)
        self.assertAlmostEqual(stats["initial"]["psi_total"], float(initial_psi.sum()))
        self.assertAlmostEqual(stats["initial"]["psi_mean"], float(initial_psi.mean()))
        self.assertAlmostEqual(
            stats["initial"]["psi_rms"],
            float(np.sqrt(np.mean(np.square(initial_psi, dtype=np.float64)))),
        )
        self.assertAlmostEqual(stats["initial"]["coefficient_transitions"],
                               initial_coeff["coefficient_transitions"])
        self.assertAlmostEqual(stats["final"]["psi_total"], float(final_psi.sum()))
        self.assertAlmostEqual(stats["final"]["psi_mean"], float(final_psi.mean()))
        self.assertAlmostEqual(
            stats["final"]["psi_rms"],
            float(np.sqrt(np.mean(np.square(final_psi, dtype=np.float64)))),
        )
        self.assertAlmostEqual(stats["final"]["coefficient_transitions"],
                               final_coeff["coefficient_transitions"])
        for key in ("accepted", "rejected", "evaluated", "swaps_accepted", "moves_accepted"):
            self.assertGreaterEqual(stats[key], 0)
        self.assertGreaterEqual(stats["accepted"],
                                stats["swaps_accepted"] + stats["moves_accepted"])

    def test_optimizer_is_deterministic_and_respects_coefficient_constraint(self) -> None:
        mesh = synthetic_mesh()
        args = make_args(self.executable, coeff_tolerance=0.0, seed=19)
        first_order, first_stats = optimize(mesh, args)
        second_order, second_stats = optimize(mesh, args)
        np.testing.assert_array_equal(first_order, second_order)
        first_stats.pop("wall_seconds")
        second_stats.pop("wall_seconds")
        self.assertEqual(first_stats, second_stats)
        initial = first_stats["initial"]["coefficient_transitions"]
        final = first_stats["final"]["coefficient_transitions"]
        self.assertLessEqual(final, initial)


    def test_multiblock_candidates_with_strict_guard(self) -> None:
        rng = np.random.default_rng(387)
        n = 96
        endpoints = rng.integers(0, n, size=(520, 2))
        edges = sorted({(min(a, b), max(a, b)) for a, b in endpoints if a != b})
        mesh = PolyMeshTopology(Path("<multiblock>"), array("i", (a for a, _ in edges)),
                                array("i", (b for _, b in edges)), n, len(edges), 0)
        for objective in ("total", "balanced"):
            args = make_args(self.executable, objective=objective, window=22,
                             coeff_tolerance=0, candidates=300, patience=0)
            order, stats = optimize(mesh, args)
            self.assertGreater(stats["swaps_accepted"], 0)
            self.assertGreater(stats["moves_accepted"], 0)
            self.assertGreater(stats["coefficient_rejected"], 0)
            self.assertEqual(stats["evaluated"], stats["accepted"] + stats["rejected"])
            self.assertLess(stats["final"]["objective"], stats["initial"]["objective"])
            self.assertLessEqual(stats["final"]["coefficient_transitions"],
                                 stats["initial"]["coefficient_transitions"])
            remapped = remapped_mesh(mesh, order)
            row_reads = [[] for _ in range(n)]
            for face, owner in enumerate(remapped.owner):
                row_reads[owner].append(face)
            trace = np.asarray([face for row in row_reads for face in row + row])
            expected = sum(int(np.count_nonzero(np.diff(trace // width))) for width in (8, 16))
            self.assertEqual(stats["final"]["coefficient_transitions"], expected)


if __name__ == "__main__":
    unittest.main()
