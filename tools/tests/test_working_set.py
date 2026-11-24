"""Reference checks for graph and sliding-window working-set metrics."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from working_set import (  # noqa: E402
    WINDOW_COLUMNS, contiguous_windows, graph_metrics, repeated_face_transitions,
    row_offsets, trace_metrics,
)


def stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {key: 0.0 for key in ("mean", "p50", "p90", "p99", "p99.9", "max", "sum")}
    ordered = np.sort(data)
    return {"mean": float(np.mean(data)),
            **{f"p{p}": float(ordered[int(p / 100 * (data.size - 1))])
               for p in (50, 90, 99, 99.9)},
            "max": float(np.max(data)), "sum": float(np.sum(data))}


def backend_path() -> Path | None:
    configured = os.environ.get("WORKING_SET_BACKEND")
    if configured:
        candidate = Path(configured).expanduser()
    else:
        candidate = ROOT / "experiments" / "build" / "working_set_backend"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def brute_window(indices: np.ndarray) -> np.ndarray:
    """Calculate specified window columns directly from element indices."""
    lines = indices // 8
    pages = indices // 512
    unique_lines = np.unique(lines).size
    unique_pages = np.unique(pages).size
    if indices.size == 0:
        return np.zeros(len(WINDOW_COLUMNS), dtype=np.float64)
    line_span = int(lines.max() - lines.min() + 1)
    element_span = int(indices.max() - indices.min() + 1)
    line_touches = int(indices.size)
    return np.asarray([
        unique_lines,
        unique_pages,
        element_span,
        element_span * 8,
        unique_lines / line_span,
        line_touches,
        line_touches - unique_lines,
        line_touches / unique_lines,
    ], dtype=np.float64)


def brute_histogram(indices: np.ndarray) -> np.ndarray:
    """Exact LRU stack distance histogram: slot zero is cold, d+1 is d."""
    lines = (indices // 8).tolist()
    seen: list[int] = []
    distances: list[int] = []
    for line in lines:
        if line in seen:
            distance = seen.index(line)
            distances.append(distance + 1)
            seen.pop(distance)
        else:
            distances.append(0)
        seen.insert(0, line)
    histogram = np.zeros(max(distances, default=0) + 1, dtype=np.int64)
    for distance in distances:
        histogram[distance] += 1
    return histogram


class GraphMetricsTest(unittest.TestCase):
    def test_known_graph_bandwidth_and_row_spans(self) -> None:
        owner = np.asarray([0, 0, 1, 2, 3], dtype=np.int32)
        neighbour = np.asarray([1, 4, 3, 5, 4], dtype=np.int32)
        bandwidth, row_span = graph_metrics(7, owner, neighbour)
        edge_distances = [1, 4, 2, 3, 1]
        self.assertEqual(bandwidth["bandwidth"], 4)
        self.assertAlmostEqual(bandwidth["normalized_bandwidth"], 4 / 7)
        for key, expected in stats(edge_distances).items():
            self.assertAlmostEqual(bandwidth["edge_distance"][key], expected)
        self.assertEqual(set(row_span), {"neighbour_span", "row_relative_max"})
        self.assertEqual(set(row_span["neighbour_span"]),
                         {"mean", "p50", "p90", "p99", "p99.9", "max"})
        self.assertEqual(set(row_span["row_relative_max"]),
                         {"mean", "p50", "p90", "p99", "p99.9", "max"})

        for key in row_span["neighbour_span"]:
            self.assertAlmostEqual(row_span["neighbour_span"][key],
                                   stats([3, 3, 0, 3, 3, 0, 0])[key])
            self.assertAlmostEqual(row_span["row_relative_max"][key],
                                   stats([4, 2, 3, 2, 4, 3, 0])[key])

    def test_edge_distance_sum_exceeds_32_bits(self) -> None:
        count = 50000
        owner = np.arange(count, dtype=np.int32)
        neighbour = owner + count
        bandwidth, _ = graph_metrics(2 * count, owner, neighbour)
        self.assertEqual(bandwidth["edge_distance"]["sum"], 2500000000)

    def test_empty_graph_and_isolated_cells_have_zero_statistics(self) -> None:
        bandwidth, row_span = graph_metrics(
            6, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        )
        self.assertEqual(bandwidth["bandwidth"], 0)
        self.assertEqual(bandwidth["normalized_bandwidth"], 0)
        self.assertEqual(bandwidth["edge_distance"]["sum"], 0)
        self.assertEqual(bandwidth["edge_distance"]["max"], 0)
        for group in row_span.values():
            self.assertEqual(group["max"], 0)
            self.assertEqual(group["mean"], 0)


@unittest.skipUnless(backend_path(), "compiled backend unavailable; set WORKING_SET_BACKEND")
class TraceMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = backend_path()
        assert cls.backend is not None

    def check_trace(self, indices: np.ndarray, windows: list[int], stride: int | None,
                    reuse: bool = False):
        # One row per access lets K be interpreted as a K-access window.
        offsets = np.arange(indices.size + 1, dtype=np.int64)
        return trace_metrics(offsets, indices, windows, stride, self.backend, reuse=reuse)

    def test_random_traces_match_brute_force_for_overlapping_and_disjoint_windows(self) -> None:
        rng = np.random.default_rng(183)
        # Include large page jumps and repeated line references in each trace.
        for length in (13, 29, 41):
            indices = rng.integers(0, 4096, size=length, dtype=np.int64)
            indices[::4] += (2**35) * 8
            for width in (1, 4, 7):
                for stride in (1, max(1, width // 2), width, width + 3, None):
                    metrics, histogram = self.check_trace(indices, [width], stride)
                    expected_starts = list(range(0, length - width + 1, stride or max(1, width // 4)))
                    actual = metrics[width]
                    self.assertEqual(actual.shape, (len(expected_starts), len(WINDOW_COLUMNS)))
                    for row, begin in zip(actual, expected_starts):
                        np.testing.assert_allclose(
                            row, brute_window(indices[begin:begin + width]),
                            rtol=0, atol=1e-12,
                        )
                    self.assertIsNone(histogram)

    def test_reuse_histogram_uses_cold_zero_and_exact_stack_distance_slots(self) -> None:
        indices = np.asarray([0, 8, 0, 16, 8, 0, 4096, 8], dtype=np.int64)
        metrics, histogram = self.check_trace(indices, [3, 8], stride=2, reuse=True)
        self.assertIsNotNone(histogram)
        np.testing.assert_array_equal(histogram, brute_histogram(indices))
        self.assertEqual(int(histogram[0]), 4)
        self.assertEqual(int(histogram[2]), 1)  # stack distance one
        self.assertEqual(int(histogram[3]), 3)  # stack distance two
        self.assertEqual(metrics[8].shape, (1, len(WINDOW_COLUMNS)))

    def test_large_indices_do_not_overflow_and_empty_windows_are_zero(self) -> None:
        base = 2**40
        indices = np.asarray([base, base + 8, base, base + 4096], dtype=np.int64)
        metrics, histogram = self.check_trace(indices, [2, 5], stride=1, reuse=True)
        np.testing.assert_allclose(metrics[2][0], brute_window(indices[:2]), rtol=0, atol=0)
        self.assertEqual(metrics[5].shape, (0, len(WINDOW_COLUMNS)))
        np.testing.assert_array_equal(histogram, brute_histogram(indices))

    def test_empty_trace_produces_empty_window_arrays_and_cold_histogram(self) -> None:
        metrics, histogram = self.check_trace(np.empty(0, dtype=np.int64), [1, 4], 1, reuse=True)
        for values in metrics.values():
            self.assertEqual(values.shape[1], len(WINDOW_COLUMNS))
            self.assertEqual(values.shape[0], 0)
        np.testing.assert_array_equal(histogram, np.zeros(1, dtype=np.int64))

    def test_windows_containing_only_empty_rows_are_zero(self) -> None:
        metrics, histogram = trace_metrics(
            np.asarray([0, 0, 0, 0], dtype=np.int64),
            np.empty(0, dtype=np.int64), [1, 2], 1, self.backend, reuse=True,
        )
        for width, count in ((1, 3), (2, 2)):
            self.assertEqual(metrics[width].shape, (count, len(WINDOW_COLUMNS)))
            np.testing.assert_array_equal(metrics[width], np.zeros_like(metrics[width]))
        np.testing.assert_array_equal(histogram, np.zeros(1, dtype=np.int64))


    def test_coefficient_windows_match_explicit_two_pass_trace(self) -> None:
        offsets = row_offsets([0, 2, 0, 19, 1, 8, 0, 27])
        reads = np.array([face for first, stop in zip(offsets[:-1], offsets[1:])
                          for _ in range(2) for face in range(first, stop)], dtype=np.int64)
        for width in (4, 8):
            expected, _ = trace_metrics(2 * offsets, reads, [1, 3, 8], 1,
                                        self.backend, entry_bytes=width)
            actual = contiguous_windows(offsets, [1, 3, 8], 1, width, 2)
            for size in expected:
                np.testing.assert_allclose(actual[size], expected[size], rtol=0, atol=0)
            lines = reads // (64 // width)
            self.assertEqual(repeated_face_transitions(offsets, 64 // width),
                             np.count_nonzero(lines[1:] != lines[:-1]))


if __name__ == "__main__":
    unittest.main()
