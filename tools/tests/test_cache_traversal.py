"""Synthetic tests for cache-aware connected traversal and candidate gates."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from cache_traversal import (  # noqa: E402
    build_order,
    choose_candidate,
    schedule_rejections,
)


def backend_path() -> Path | None:
    configured = os.environ.get("CACHE_TRAVERSAL_BACKEND")
    if configured:
        candidate = Path(configured).expanduser()
    else:
        candidate = ROOT / "experiments" / "build" / "cache_traversal_backend"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def fixture_graph() -> tuple[int, np.ndarray, np.ndarray]:
    # Components: a four-cell cluster, one edge, one isolated cell, and a path.
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (4, 5), (7, 8), (8, 9)]
    return (10, np.asarray([a for a, _ in edges], dtype=np.int32),
            np.asarray([b for _, b in edges], dtype=np.int32))


def components(n_cells: int, owner: np.ndarray, neighbour: np.ndarray) -> list[set[int]]:
    adjacency = [set() for _ in range(n_cells)]
    for a, b in zip(owner.tolist(), neighbour.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    unseen = set(range(n_cells))
    result = []
    while unseen:
        root = min(unseen)
        found = {root}
        pending = [root]
        unseen.remove(root)
        while pending:
            for other in adjacency[pending.pop()]:
                if other in unseen:
                    unseen.remove(other)
                    found.add(other)
                    pending.append(other)
        result.append(found)
    return result


@unittest.skipUnless(backend_path(), "compiled backend unavailable; set CACHE_TRAVERSAL_BACKEND")
class BuildOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = backend_path()
        assert cls.backend is not None

    def make_order(self, n_cells: int, owner: np.ndarray, neighbour: np.ndarray):
        return build_order(
            n_cells, owner, neighbour, backend=self.backend,
            weights=(1.0, 1.0, 1.0, 1.0), scan=16, recent=8, seed=41,
            root="seed", audit_every=1,
        )

    def test_fixed_settings_are_deterministic_and_audit_every_placement(self) -> None:
        n_cells, owner, neighbour = fixture_graph()
        first_order, first_stats = self.make_order(n_cells, owner, neighbour)
        second_order, second_stats = self.make_order(n_cells, owner, neighbour)
        self.assertEqual(first_order.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(first_order, second_order)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(sorted(first_order.tolist()), list(range(n_cells)))
        self.assertEqual(first_stats["audited_steps"], n_cells)

    def test_prefix_is_connected_except_at_component_restarts(self) -> None:
        n_cells, owner, neighbour = fixture_graph()
        order, stats = self.make_order(n_cells, owner, neighbour)
        adjacency = [set() for _ in range(n_cells)]
        for a, b in zip(owner.tolist(), neighbour.tolist()):
            adjacency[a].add(b)
            adjacency[b].add(a)
        ordered: set[int] = set()
        starts = 0
        for cell in order.tolist():
            if not adjacency[cell].intersection(ordered):
                starts += 1
            ordered.add(cell)
        expected_components = len(components(n_cells, owner, neighbour))
        self.assertEqual(starts, expected_components)
        self.assertEqual(stats["disconnected_restarts"], starts - 1)
        self.assertIn(6, order)
        self.assertEqual(stats["audited_steps"], n_cells)


    def test_known_and_predicted_lines_have_distinct_costs(self) -> None:
        order, stats = build_order(
            2, np.array([0]), np.array([1]), backend=self.backend,
            weights=(1.0, 1.0, 1.0, 1.0), scan=2, recent=1, seed=0,
            root="seed", audit_every=1,
        )
        np.testing.assert_array_equal(order, [0, 1])
        # Row 0 introduces its write line and 1/8 of a predicted neighbour
        # line. Row 1 reuses the known line for both labels.
        self.assertAlmostEqual(stats["selected_term_mean"][0], (1 + 1/8) / 8 / 2)
        self.assertAlmostEqual(stats["selected_term_mean"][1], 1 / 8 / 2)
        self.assertEqual(stats["audited_steps"], 2)


def layout(index_levels: int, index_median: int, index_max: int,
           packed_levels: int, packed_median: int, packed_max: int) -> dict:
    return {
        "index_kahn_wavefronts": {
            "levels": index_levels, "median_rows": index_median,
            "maximum_rows": index_max,
        },
        "packed_ih1_wavefronts": {
            "levels": packed_levels, "median_rows": packed_median,
            "maximum_rows": packed_max,
        },
    }


class CandidateGateTest(unittest.TestCase):
    def test_schedule_gate_rejects_large_wave_ratio_and_narrow_median(self) -> None:
        original = layout(8, 8, 16, 10, 6, 12)
        good = layout(9, 5, 20, 10, 5, 14)
        bad = layout(12, 2, 40, 14, 3, 28)
        self.assertEqual(schedule_rejections(good, original, 1.5, 5), [])
        reasons = schedule_rejections(bad, original, 1.5, 5)
        self.assertTrue(reasons)
        rendered = " ".join(reasons).lower()
        self.assertTrue("wave" in rendered or "maximum" in rendered or "ratio" in rendered)
        self.assertTrue("median" in rendered or "width" in rendered)

    def test_minimum_width_is_capped_by_small_original_layout(self) -> None:
        original = layout(7, 3, 8, 8, 2, 7)
        candidate = layout(7, 3, 8, 8, 2, 7)
        self.assertEqual(schedule_rejections(candidate, original, 0.0, 20), [])

    def test_both_zero_thresholds_disable_schedule_gate(self) -> None:
        original = layout(4, 10, 20, 4, 10, 20)
        very_narrow = layout(20, 1, 1000, 30, 1, 1000)
        self.assertEqual(schedule_rejections(very_narrow, original, 0, 0), [])

    def test_candidate_selection_ignores_rejected_rows_and_breaks_ties_by_profile(self) -> None:
        rows = [
            {"profile": "zeta", "score": 6.0, "reasons": []},
            {"profile": "beta", "score": 4.0, "reasons": ["wave ratio too high"]},
            {"profile": "gamma", "score": 5.0, "reasons": []},
            {"profile": "alpha", "score": 5.0, "reasons": []},
        ]
        self.assertEqual(choose_candidate(rows), rows[3])
        self.assertIsNone(choose_candidate([
            {"profile": "beta", "score": 1.0, "reasons": ["rejected"]},
        ]))
        self.assertIsNone(choose_candidate([]))


if __name__ == "__main__":
    unittest.main()
