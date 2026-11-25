import tempfile
import unittest
from pathlib import Path

from measure_gs_pmu import event_spec, perf_rows, summarize, usable


class PmuInfrastructureTests(unittest.TestCase):
    def test_cpu_core_events_are_qualified_and_software_events_are_not(self):
        self.assertEqual(event_spec(["cycles", "instructions"]),
                         "{cpu_core/cycles/u,cpu_core/instructions/u}")
        self.assertEqual(event_spec(["task-clock", "cpu-migrations"]),
                         "{task-clock,cpu-migrations}")

    def test_perf_csv_preserves_count_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf.csv"
            path.write_text("12345,,cpu_core/cycles/u,67890,99.5,,\n")
            rows = perf_rows(path)
        self.assertEqual(rows[0]["event"], "cycles")
        self.assertEqual(rows[0]["count"], 12345)
        self.assertEqual(rows[0]["coverage_percent"], 99.5)
        self.assertEqual(rows[0]["perf_running_time_ns"], 67890)
        self.assertTrue(usable({"count_per_sweep": 4, "coverage_percent": 100}))
        self.assertFalse(usable({"count_per_sweep": 4, "coverage_percent": 90}))

    def test_per_sweep_and_ipc_summary_uses_only_verified_samples(self):
        rows = []
        for event, value in (("cycles", 1000), ("instructions", 2000)):
            rows.append({"event": event, "count_per_sweep": value,
                         "coverage_percent": 100, "unavailable": False,
                         "ms_per_sweep": 1.25, "sample": 0, "group": 0})
        rows.append({"event": "cycles", "count_per_sweep": 900,
                     "coverage_percent": 80, "unavailable": False,
                     "ms_per_sweep": 1.20, "sample": 1, "group": 0})
        report = summarize(rows)
        self.assertEqual(report["events"]["cycles"]["per_sweep"]["median"], 1000)
        self.assertEqual(report["ipc"]["median"], 2.0)
        self.assertEqual(report["events"]["cycles"]["multiplexed_samples"], 1)


if __name__ == "__main__":
    unittest.main()
