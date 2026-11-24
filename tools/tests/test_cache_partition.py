"""Synthetic invariant tests for greedy cache-sized mesh partitions."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import json
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import mesh_renumber  # noqa: E402

from working_set import gs_psi_trace  # noqa: E402

from cache_partition import (  # noqa: E402
    build_coarse_graph,
    grow_partitions,
    order_partitions,
)


def backend_path() -> Path | None:
    configured = os.environ.get("CACHE_PARTITION_BACKEND")
    if configured:
        candidate = Path(configured).expanduser()
    else:
        candidate = ROOT / "experiments" / "build" / "cache_partition_backend"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def synthetic_graph() -> tuple[int, np.ndarray, np.ndarray]:
    """Several connected components, branches, and isolated cells."""
    edges = [
        (0, 1), (0, 2), (1, 2), (1, 3),
        (2, 4), (3, 4), (3, 5), (4, 5),
        (6, 7),
        (9, 10),
    ]
    return (12, np.asarray([a for a, _ in edges], dtype=np.int32),
            np.asarray([b for _, b in edges], dtype=np.int32))


def assert_group_connected(test: unittest.TestCase, members: list[int],
                           owner: np.ndarray, neighbour: np.ndarray) -> None:
    if len(members) <= 1:
        return
    allowed = set(members)
    adjacency = {cell: set() for cell in members}
    for a, b in zip(owner.tolist(), neighbour.tolist()):
        if a in allowed and b in allowed:
            adjacency[a].add(b)
            adjacency[b].add(a)
    reached = {members[0]}
    pending = [members[0]]
    while pending:
        for other in adjacency[pending.pop()]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    test.assertEqual(reached, allowed)


@unittest.skipUnless(backend_path(), "compiled backend unavailable; set CACHE_PARTITION_BACKEND")
class GrowPartitionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = backend_path()
        assert cls.backend is not None

    def test_assignments_are_deterministic_connected_and_byte_estimates_exact(self) -> None:
        n_cells, owner, neighbour = synthetic_graph()
        first = grow_partitions(n_cells, owner, neighbour, 512, 271, self.backend)
        second = grow_partitions(n_cells, owner, neighbour, 512, 271, self.backend)
        for key in ("membership", "traversal", "local_rcm", "offsets",
                    "estimated_bytes", "footprint_cells"):
            np.testing.assert_array_equal(first[key], second[key])

        membership = first["membership"]
        traversal = first["traversal"]
        local_rcm = first["local_rcm"]
        offsets = first["offsets"]
        self.assertEqual(membership.dtype, np.dtype(np.int32))
        self.assertEqual(sorted(traversal.tolist()), list(range(n_cells)))
        self.assertEqual(sorted(local_rcm.tolist()), list(range(n_cells)))
        self.assertEqual(int(offsets[0]), 0)
        self.assertEqual(int(offsets[-1]), n_cells)
        self.assertTrue(np.all(offsets[1:] >= offsets[:-1]))
        self.assertEqual(first["estimated_bytes"].shape, (offsets.size - 1,))
        self.assertEqual(first["footprint_cells"].shape, (offsets.size - 1,))
        np.testing.assert_array_equal(
            membership[traversal],
            np.repeat(np.arange(offsets.size - 1), np.diff(offsets)),
        )

        degree = np.bincount(np.concatenate((owner, neighbour)), minlength=n_cells)
        for group, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            cells = traversal[int(begin):int(end)].astype(int).tolist()
            self.assertTrue(cells)
            self.assertTrue(np.all(membership[cells] == group))
            assert_group_connected(self, cells, owner, neighbour)
            footprint = set(cells)
            for a, b in zip(owner.tolist(), neighbour.tolist()):
                if a in cells:
                    footprint.add(b)
                if b in cells:
                    footprint.add(a)
            self.assertEqual(int(first["footprint_cells"][group]), len(footprint))
            expected_bytes = 8 * len(footprint) + 4 * int(degree[cells].sum()) + 4 * (len(cells) + 1)
            self.assertEqual(int(first["estimated_bytes"][group]), expected_bytes)
            local_cells = local_rcm[int(begin):int(end)].astype(int).tolist()
            self.assertCountEqual(local_cells, cells)
            self.assertTrue(np.all(membership[local_cells] == group))

        # Cells with no incident faces remain present as valid singleton groups.
        self.assertTrue(all(membership[cell] >= 0 for cell in (8, 11)))
        self.assertEqual(int(np.count_nonzero(membership == membership[8])), 1)
        self.assertEqual(int(np.count_nonzero(membership == membership[11])), 1)

    def test_a_singleton_may_exceed_the_soft_budget(self) -> None:
        # The isolated vertex has a nonzero minimum estimate even though its
        # requested budget is smaller; the builder must preserve the vertex.
        membership = grow_partitions(
            1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
            1, 4, self.backend,
        )
        self.assertEqual(membership["membership"].tolist(), [0])
        self.assertEqual(membership["traversal"].tolist(), [0])
        self.assertEqual(membership["footprint_cells"].tolist(), [1])
        self.assertEqual(membership["estimated_bytes"].tolist(), [16])
        self.assertGreater(membership["estimated_bytes"][0], 1)


    def test_writes_valid_polymesh_and_partition_metadata(self) -> None:
        from test_mesh_renumber import write_fixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = write_fixture(root)
            output = root / "cache-partition"
            source = (mesh / "owner").read_bytes()
            command = [
                sys.executable, str(TOOLS / "mesh_renumber.py"), "renumber",
                "--algorithm", "cache-partition", "--initial", "original",
                "--mesh", str(mesh), "--output-dir", str(output),
                "--partition-budget-kib", "1", "--partition-backend", str(self.backend),
                "--partition-ws-backend", os.environ.get(
                    "WORKING_SET_BACKEND", str(ROOT / "experiments/build/working_set_backend")),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((mesh / "owner").read_bytes(), source)
            report = json.loads((output / "validation.json").read_text())
            self.assertTrue(report["valid"])
            self.assertEqual(report["cells"], 2)
            with np.load(output / "partitions.npz") as partitions:
                np.testing.assert_array_equal(partitions["partition_new_cell"], [0, 0])
                np.testing.assert_array_equal(partitions["local_offsets"], [0, 2])


class CoarseGraphTest(unittest.TestCase):
    def test_cross_group_face_weights_are_sorted_and_unique(self) -> None:
        owner = np.asarray([0, 1, 0, 2, 3, 2, 4, 0], dtype=np.int32)
        neighbour = np.asarray([1, 2, 3, 3, 4, 5, 5, 5], dtype=np.int32)
        membership = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
        edge_a, edge_b, weights = build_coarse_graph(membership, 3, owner, neighbour)
        np.testing.assert_array_equal(edge_a, [0, 0, 1])
        np.testing.assert_array_equal(edge_b, [1, 2, 2])
        np.testing.assert_array_equal(weights, [2, 1, 2])
        self.assertTrue(np.all(edge_a < edge_b))
        self.assertEqual(list(zip(edge_a.tolist(), edge_b.tolist())),
                         sorted(set(zip(edge_a.tolist(), edge_b.tolist()))))

    def test_strongest_neighbor_drives_breadth_and_depth_order(self) -> None:
        # From group 0, the strongest choice is group 1. Depth continues from
        # that choice through 3 before returning to the other frontier at 2.
        edge_a = np.asarray([0, 0, 1, 2], dtype=np.int32)
        edge_b = np.asarray([1, 2, 3, 3], dtype=np.int32)
        weights = np.asarray([10, 9, 8, 4], dtype=np.int64)
        breadth = order_partitions(4, edge_a, edge_b, weights, "breadth")
        depth = order_partitions(4, edge_a, edge_b, weights, "depth")
        np.testing.assert_array_equal(breadth, [0, 1, 2, 3])
        np.testing.assert_array_equal(depth, [0, 1, 3, 2])
        np.testing.assert_array_equal(np.sort(breadth), np.arange(4))
        np.testing.assert_array_equal(np.sort(depth), np.arange(4))


class GsTraceTest(unittest.TestCase):
    def test_reads_follow_face_order_then_write_for_each_row(self) -> None:
        offsets, indices = gs_psi_trace(np.array([0, 0, 2]), np.array([3, 1, 3]), 4)
        np.testing.assert_array_equal(offsets, [0, 3, 4, 6, 7])
        np.testing.assert_array_equal(indices, [3, 1, 0, 1, 3, 2, 3])


if __name__ == "__main__":
    unittest.main()
