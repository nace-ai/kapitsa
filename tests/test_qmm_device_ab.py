#!/usr/bin/env python3
"""CPU-only fault injection for the device A/B validity gate; no model or GPU."""
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import qmm_device_ab as ab


class DeviceConditions(unittest.TestCase):
    def setUp(self):
        self.before = dict(valid=True, ac=True, power_source="AC Power", low_power=False,
                           thermal=0, load1=1.0, memory_pressure_level=1,
                           pageouts=100, swapins=200, swapouts=300, monotonic_ns=1000000000)
        self.after = dict(self.before, monotonic_ns=6000000000)
        self.samples = [dict(self.before, time=1+i*.25, interval_s=.25, cpu_idle_pct=96, background_cpu_pct=4,
                             foreign_gpu_ns=0, own_gpu_visible=True) for i in range(20)]

    def errors(self):
        return ab.validate_leg(self.before, self.after, self.samples)

    def test_valid_control(self):
        self.assertEqual(self.errors(), [])

    def test_61_pageouts_are_invalid_even_without_swap(self):
        self.after["pageouts"] += 61
        self.assertIn("pageouts_changed", self.errors())

    def test_ac_required_at_both_endpoints_and_during_run(self):
        for point in (self.before, self.after, self.samples[5]):
            with self.subTest(point=point):
                point["ac"] = False
                self.assertTrue(any("not_AC" in x for x in self.errors()))
                point["ac"] = True

    def test_system_load_at_both_endpoints(self):
        for point in (self.before, self.after):
            point["load1"] = 2.01
            self.assertTrue(any("load1_over_limit" in x for x in self.errors()))
            point["load1"] = 1

    def test_missing_ac_and_nan_load_fail_closed(self):
        del self.before["ac"]
        self.after["load1"] = float("nan")
        errors = self.errors()
        self.assertIn("before:not_AC", errors)
        self.assertIn("after:load1_over_limit", errors)

    def test_paging_or_counter_reset_anywhere(self):
        for key in ab.PAGING:
            for change in (-1, 1):
                with self.subTest(key=key, change=change):
                    self.samples[10][key] += change
                    self.assertIn(key+"_changed", self.errors())
                    self.samples[10][key] -= change

    def test_thermal_power_pressure_and_cpu(self):
        cases = [("thermal", 1, "thermal_not_nominal"), ("low_power", True, "low_power_mode"),
                 ("memory_pressure_level", -1, "memory_pressure"), ("background_cpu_pct", 5.01, "background_CPU_over_limit")]
        for key, value, error in cases:
            with self.subTest(key=key):
                item = copy.deepcopy(self.samples[4])
                self.samples[4][key] = value
                self.assertIn(error, self.errors())
                self.samples[4] = item

    def test_any_foreign_gpu_activity(self):
        self.samples[2]["foreign_gpu_ns"] = 1
        self.assertIn("foreign_GPU_activity", self.errors())

    def test_cpu_gate_uses_request_intervals_not_postamble(self):
        self.samples[-1]["background_cpu_pct"] = 20
        valid_requests = [dict(invalid=[])]
        errors, flags = ab.arm_errors(self.before, self.after, self.samples, valid_requests, 2, 5)
        self.assertEqual(errors, [])
        self.assertIn("background_CPU_over_limit", flags)
        invalid_requests = [dict(invalid=["background_CPU_over_limit"])]
        errors, _ = ab.arm_errors(self.before, self.after, self.samples, invalid_requests, 2, 5)
        self.assertIn("invalid_request", errors)
        self.after["pageouts"] += 61
        self.samples[-1]["foreign_gpu_ns"] = 1
        errors, _ = ab.arm_errors(self.before, self.after, self.samples, valid_requests, 2, 5)
        self.assertIn("pageouts_changed", errors)
        self.assertIn("foreign_GPU_activity", errors)

    def test_monitor_failure_missing_counter_and_gaps(self):
        self.samples[2]["error"] = "ioreg timeout"
        self.samples[3]["interval_s"] = 1.01
        self.samples[4]["gpu_counter_reset"] = True
        self.samples[5]["own_gpu_visible"] = False
        errors = self.errors()
        for error in ("monitor_error", "monitor_gap", "GPU_counter_reset", "missing_GPU_counters"):
            self.assertIn(error, errors)

    def test_three_samples_not_enough(self):
        self.samples = self.samples[:3]
        self.assertIn("insufficient_monitor_samples", self.errors())

    def test_bad_time_order(self):
        self.after["monotonic_ns"] = self.before["monotonic_ns"]
        self.samples[3]["time"] = self.samples[2]["time"]
        self.assertIn("snapshot_time_not_increasing", self.errors())
        self.assertIn("monitor_time_not_increasing", self.errors())

    def test_gpu_query_failure_is_not_quiet(self):
        with patch.object(ab.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ioreg")):
            with self.assertRaises(subprocess.CalledProcessError):
                ab.gpu_snapshot()
        with patch.object(ab.subprocess, "run", return_value=SimpleNamespace(stdout="")):
            with self.assertRaises(RuntimeError):
                ab.gpu_snapshot()

    def test_cached_cpu_ticks_use_a_full_window(self):
        history = []
        for i in range(4):
            value = ab.cpu_usage(history, dict(monotonic_ns=i*250000000, cpu_ticks=[10, 10, 180, 0], owner_cpu_ns=0, cpu_count=10))
            self.assertEqual(value["cpu_idle_pct"], -1)
        value = ab.cpu_usage(history, dict(monotonic_ns=1000000000, cpu_ticks=[15, 15, 370, 0], owner_cpu_ns=0, cpu_count=10))
        self.assertEqual(value["cpu_idle_pct"], 95)
        value = ab.cpu_usage(history, dict(monotonic_ns=1250000000, cpu_ticks=[15, 15, 370, 0], owner_cpu_ns=0, cpu_count=10))
        self.assertEqual(value["cpu_idle_pct"], 95)

    def test_owned_cpu_is_recorded_separately_from_background(self):
        history = [dict(monotonic_ns=0, cpu_ticks=[0, 0, 0, 0], owner_cpu_ns=0, cpu_count=10)]
        value = ab.cpu_usage(history, dict(monotonic_ns=1000000000, cpu_ticks=[140, 10, 850, 0],
                                           owner_cpu_ns=1000000000, cpu_count=10))
        self.assertEqual(value["cpu_idle_pct"], 85)
        self.assertEqual(value["owner_cpu_pct"], 10)
        self.assertEqual(value["background_cpu_pct"], 5)

    def test_missing_or_reset_cpu_accounting_is_invalid(self):
        del self.samples[2]["background_cpu_pct"]
        self.assertIn("background_CPU_over_limit", self.errors())
        history = [dict(monotonic_ns=0, cpu_ticks=[0, 0, 0, 0], owner_cpu_ns=2, cpu_count=10)]
        value = ab.cpu_usage(history, dict(monotonic_ns=1000000000, cpu_ticks=[1, 1, 98, 0], owner_cpu_ns=1, cpu_count=10))
        self.assertIsNone(value["background_cpu_pct"])

    def test_tail_interval_is_retained(self):
        monitor = SimpleNamespace(read=lambda: self.samples)
        args = SimpleNamespace(deadline=None)
        observed = ab.window_samples(monitor, 2.1, 3.1, args)
        self.assertEqual(observed[0]["time"], 2.25)
        self.assertEqual(observed[-1]["time"], 3.25)

    def test_deadline_bounds_work(self):
        with patch.object(ab.time, "time", return_value=100):
            self.assertEqual(ab.remaining(SimpleNamespace(deadline=110), 300), 10)
            with self.assertRaises(TimeoutError):
                ab.remaining(SimpleNamespace(deadline=99), 300)

    def test_start_window_avoids_minute_boundaries(self):
        for epoch in (9.9, 35.1, 59.9, 60, 120):
            self.assertFalse(ab.in_start_window([10, 35], epoch))
        for epoch in (10, 20, 35, 70, 95, 140):
            self.assertTrue(ab.in_start_window([10, 35], epoch))
        self.assertTrue(ab.in_start_window(None, 0))

    def test_campaign_cannot_drop_a_bad_or_missing_leg(self):
        requests = [dict(case=c, text_sha256="same") for c in ("short_greedy", "4k_sampled")]
        legs = [dict(invalid=[], requests=copy.deepcopy(requests)) for _ in range(4)]
        self.assertEqual(ab.campaign_errors(legs, 4, 1), [])
        legs[2]["invalid"] = ["pageouts_changed"]
        self.assertIn("invalid_leg", ab.campaign_errors(legs, 4, 1))
        del legs[2]
        self.assertIn("incomplete_campaign", ab.campaign_errors(legs, 4, 1))

    def test_different_output_cannot_pass_as_same_work(self):
        requests = [dict(case=c, text_sha256="same") for c in ("short_greedy", "4k_sampled")]
        legs = [dict(invalid=[], requests=copy.deepcopy(requests)) for _ in range(4)]
        legs[1]["requests"][0]["text_sha256"] = "different"
        self.assertIn("short_greedy:outputs_differ", ab.campaign_errors(legs, 4, 1))


if __name__ == "__main__":
    unittest.main()
