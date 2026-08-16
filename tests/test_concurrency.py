"""Tests for async_api_ops/concurrency.py.

Two things here are worth pinning down. One is the arithmetic that turns "how much memory is
free" into "how many requests may be in flight", which is easy to change in a way that looks
reasonable and quietly produces a limit of 4 or of 4000. The other is the gate itself, whose
awkward case is a limit that shrinks while tasks are queued behind it - the point at which a
lost wakeup would hang a run for good.

Memory probing is stubbed throughout: a test that asked the machine how much RAM it had would
assert different things on different machines.

Not covered here: the connection pool being sized from the gate's ceiling before that ceiling
has been measured. That lives in base_ops, which needs aqt, so it is out of reach of these
tests.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from addon_modules import load_addon_module  # type: ignore

conc = load_addon_module("concurrency")

MB = conc.MB
GB = 1024 * MB


class StubMemory:
    """Stands in for the platform memory probes.

    `total`/`available` answer system_memory; `rss` answers process_memory and can be moved
    during a test to simulate the process growing or shrinking.
    """

    def __init__(self, total: int = 32 * GB, available: int = 16 * GB, rss: int = 500 * MB):
        self.total = total
        self.available = available
        self.rss = rss
        self.probe_failed = False

    def system_memory(self):
        return None if self.probe_failed else (self.total, self.available)

    def process_memory(self):
        return None if self.probe_failed else self.rss


class MemoryStubTestCase(unittest.TestCase):
    """Replaces the memory probes and the estimates file for the duration of a test."""

    def setUp(self):
        self.memory = StubMemory()
        self._real_system_memory = conc.system_memory
        self._real_process_memory = conc.process_memory
        self._real_load = conc.load_per_task_estimates
        conc.system_memory = self.memory.system_memory
        conc.process_memory = self.memory.process_memory
        # Never read the real user_files estimates: they differ per machine and per run
        conc.load_per_task_estimates = lambda: {}

    def tearDown(self):
        conc.system_memory = self._real_system_memory
        conc.process_memory = self._real_process_memory
        conc.load_per_task_estimates = self._real_load


# --- Reading the machine -------------------------------------------------------------------


class FormatBytesTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(conc.format_bytes(512 * MB), "512 MB")
        self.assertEqual(conc.format_bytes(2 * GB), "2.0 GB")
        self.assertEqual(conc.format_bytes(None), "?")


class MemoryProbeTests(unittest.TestCase):
    def test_the_reserve_is_a_share_of_total_memory_with_a_floor(self):
        # A tenth of a big machine
        self.assertEqual(conc.memory_reserve(32 * GB), int(32 * GB * conc.RESERVE_TOTAL_FRACTION))
        # ...but never less than the floor, or a small machine would be left with nothing
        self.assertEqual(conc.memory_reserve(1 * GB), conc.MIN_RESERVE_BYTES)

    def test_the_probes_report_this_machine(self):
        # Whatever platform the suite runs on, the probes must work: everything below is
        # stubbed, so this is the one check that the real ones are wired up
        memory = conc.system_memory()
        self.assertIsNotNone(memory, "system memory probe failed on this platform")
        total, available = memory
        self.assertGreater(total, 0)
        self.assertGreater(available, 0)
        self.assertLessEqual(available, total)
        rss = conc.process_memory()
        self.assertIsNotNone(rss, "process memory probe failed on this platform")
        self.assertGreater(rss, 0)


class MemoryBudgetTests(MemoryStubTestCase):
    def test_the_reserve_comes_off_before_anything_is_budgeted(self):
        # Budgeting from raw available memory would plan a limit that immediately triggers the
        # pressure response it is supposed to stay clear of
        self.memory.total = 32 * GB
        self.memory.available = 16 * GB
        reserve = conc.memory_reserve(32 * GB)
        spendable = (16 * GB - reserve) * conc.MEMORY_TARGET_FRACTION
        self.assertEqual(conc.memory_budget(), min(spendable, 32 * GB * conc.MEMORY_TOTAL_FRACTION))

    def test_the_budget_is_capped_as_a_share_of_total_memory(self):
        # Free memory right after a reboot is not an invitation to use all of it
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        self.assertEqual(conc.memory_budget(), 8 * GB * conc.MEMORY_TOTAL_FRACTION)

    def test_no_budget_when_less_is_free_than_the_reserve(self):
        self.memory.total = 32 * GB
        self.memory.available = 1 * GB
        self.assertEqual(conc.memory_budget(), 0)

    def test_no_budget_when_the_probe_is_unavailable(self):
        self.memory.probe_failed = True
        self.assertIsNone(conc.memory_budget())


class MaxConcurrencyTests(MemoryStubTestCase):
    def test_the_ceiling_is_the_budget_divided_by_what_a_task_costs(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB
        self.assertEqual(conc.max_concurrency_for(16 * MB), 128)

    def test_a_cheap_task_does_not_lift_the_ceiling_without_limit(self):
        self.assertEqual(conc.max_concurrency_for(1), conc.MAX_AUTO_CONCURRENCY)

    def test_an_expensive_task_still_leaves_a_workable_floor(self):
        self.assertEqual(conc.max_concurrency_for(100 * GB), conc.MIN_AUTO_CONCURRENCY)

    def test_a_conservative_ceiling_when_memory_cannot_be_probed(self):
        self.memory.probe_failed = True
        self.assertEqual(conc.max_concurrency_for(8 * MB), conc.NO_PROBE_CONCURRENCY)

    def test_the_thread_pool_is_sized_for_the_highest_reachable_limit(self):
        # The pool is created before the gate, so it has to cover a ceiling that may be raised
        # later once the op has been measured
        self.assertEqual(conc.max_possible_concurrency({}), conc.MAX_AUTO_CONCURRENCY)
        self.assertEqual(conc.max_possible_concurrency({"max_concurrent_requests": 400}), 400)
        self.assertEqual(conc.max_possible_concurrency(None), conc.MAX_AUTO_CONCURRENCY)


class ConcurrencyLimitsTests(MemoryStubTestCase):
    def test_a_run_starts_below_its_ceiling_and_adapts_upward(self):
        start, ceiling, adaptive = conc.concurrency_limits({}, 8 * MB)
        self.assertTrue(adaptive)
        self.assertEqual(start, conc.ADAPTIVE_START_CONCURRENCY)
        self.assertGreater(ceiling, start)

    def test_a_low_ceiling_also_lowers_the_starting_limit(self):
        self.memory.total = 2 * GB
        self.memory.available = 2 * GB
        start, ceiling, adaptive = conc.concurrency_limits({}, 64 * MB)
        self.assertTrue(adaptive)
        self.assertEqual(start, ceiling)

    def test_a_configured_maximum_lowers_the_ceiling_but_keeps_adapting(self):
        # The memory-pressure response underneath is what protects the machine, so a
        # configured limit must not switch it off
        start, ceiling, adaptive = conc.concurrency_limits({"max_concurrent_requests": 6}, 8 * MB)
        self.assertTrue(adaptive)
        self.assertEqual(ceiling, 6)
        self.assertEqual(start, 6)

    def test_without_a_probe_the_limit_is_static(self):
        self.memory.probe_failed = True
        self.assertEqual(
            conc.concurrency_limits({}, 8 * MB),
            (conc.NO_PROBE_CONCURRENCY, conc.NO_PROBE_CONCURRENCY, False),
        )
        self.assertEqual(
            conc.concurrency_limits({"max_concurrent_requests": 3}, 8 * MB), (3, 3, False)
        )


# --- Learning what an op costs ---------------------------------------------------------------


class MemoryEstimatorTests(MemoryStubTestCase):
    def measure(self, baseline: int, peak: int, in_flight: int):
        """Run one window: baseline at the start, `peak` reached with `in_flight` tasks."""
        estimator = conc.MemoryEstimator("op")
        self.memory.rss = baseline
        estimator.begin_window()
        self.memory.rss = peak
        estimator.sample(self.memory.rss, in_flight)
        return estimator, estimator.end_window()

    def test_growth_is_attributed_across_the_tasks_that_caused_it(self):
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=10)
        self.assertEqual(per_task, 4 * MB)

    def test_a_window_that_never_ran_enough_at_once_teaches_nothing(self):
        # One task's growth says nothing about what running fifty would cost
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=1)
        self.assertIsNone(per_task)

    def test_reused_memory_teaches_nothing(self):
        # The allocator already held the pages, so RSS did not move
        _, per_task = self.measure(baseline=500 * MB, peak=500 * MB, in_flight=10)
        self.assertIsNone(per_task)

    def test_an_implausibly_large_measurement_is_clamped(self):
        _, per_task = self.measure(baseline=0, peak=8 * GB, in_flight=2)
        self.assertEqual(per_task, conc.MAX_PER_TASK_MEMORY)

    def test_an_implausibly_small_measurement_is_clamped(self):
        # Otherwise a near-zero cost would divide the budget into a limit of millions
        _, per_task = self.measure(baseline=500 * MB, peak=500 * MB + 100, in_flight=50)
        self.assertEqual(per_task, conc.MIN_PER_TASK_MEMORY)

    def test_within_a_run_the_largest_window_wins(self):
        # What has to fit in RAM is the peak, not the average
        estimator = conc.MemoryEstimator("op")
        self.memory.rss = 500 * MB
        estimator.begin_window()
        self.memory.rss = 540 * MB
        estimator.sample(self.memory.rss, 10)
        self.assertEqual(estimator.end_window(), 4 * MB)

        self.memory.rss = 500 * MB
        estimator.begin_window()
        self.memory.rss = 510 * MB
        estimator.sample(self.memory.rss, 10)
        self.assertIsNone(estimator.end_window(), "a cheaper window must not lower the estimate")
        self.assertEqual(estimator.measured, 4 * MB)

    def test_a_stored_estimate_is_used_until_something_is_measured(self):
        estimator = conc.MemoryEstimator("op", stored=3 * MB)
        self.assertEqual(estimator.estimate, 3 * MB)
        self.assertTrue(estimator.from_measurement)

    def test_the_default_guess_is_used_when_nothing_was_stored(self):
        estimator = conc.MemoryEstimator("op")
        self.assertEqual(estimator.estimate, conc.DEFAULT_PER_TASK_MEMORY)
        self.assertFalse(estimator.from_measurement)

    def test_end_window_without_a_baseline_teaches_nothing(self):
        estimator = conc.MemoryEstimator("op")
        self.assertIsNone(estimator.end_window())


class EstimatesFileTests(MemoryStubTestCase):
    def setUp(self):
        super().setUp()
        self._tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tempdir.name) / "memory_estimates.json"
        self._real_path = conc._estimates_path
        self._real_load = conc.load_per_task_estimates
        conc._estimates_path = lambda: self.path
        # Undo MemoryStubTestCase's stub: these tests exercise the real reader
        conc.load_per_task_estimates = self._real_load

    def tearDown(self):
        conc._estimates_path = self._real_path
        self._tempdir.cleanup()
        super().tearDown()

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_a_corrupt_file_is_not_an_error(self):
        # Rather than breaking every run until someone deletes it by hand
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_a_file_holding_something_other_than_a_mapping_is_ignored(self):
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_the_first_measurement_is_stored_as_is(self):
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")), {"Making meanings": 2 * MB}
        )

    def test_later_measurements_are_blended_so_one_odd_run_cannot_skew_it(self):
        conc.save_per_task_estimate("Making meanings", 1 * MB)
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        expected = int(1 * MB * (1 - conc.ESTIMATE_BLEND) + 2 * MB * conc.ESTIMATE_BLEND)
        self.assertEqual(conc.load_per_task_estimates()["Making meanings"], expected)

    def test_estimates_are_kept_per_op(self):
        conc.save_per_task_estimate("Making meanings", 1 * MB)
        conc.save_per_task_estimate("Translating sentences", 4 * MB)
        stored = conc.load_per_task_estimates()
        self.assertEqual(stored["Making meanings"], 1 * MB)
        self.assertEqual(stored["Translating sentences"], 4 * MB)

    def test_nothing_is_persisted_when_nothing_was_measured(self):
        conc.MemoryEstimator("Making meanings").persist()
        self.assertFalse(self.path.exists())

    def test_an_unwritable_location_is_not_an_error(self):
        # Losing a measurement is not worth failing a run over
        conc._estimates_path = lambda: Path(self._tempdir.name) / "nope" / "\0bad" / "x.json"
        conc.save_per_task_estimate("Making meanings", 1 * MB)


# --- The gate ---------------------------------------------------------------------------------


class GateTestCase(unittest.IsolatedAsyncioTestCase):
    """Async tests against a gate with stubbed memory."""

    def setUp(self):
        self.memory = StubMemory()
        self._real_system_memory = conc.system_memory
        self._real_process_memory = conc.process_memory
        self._real_load = conc.load_per_task_estimates
        conc.system_memory = self.memory.system_memory
        conc.process_memory = self.memory.process_memory
        conc.load_per_task_estimates = lambda: {}

    def tearDown(self):
        conc.system_memory = self._real_system_memory
        conc.process_memory = self._real_process_memory
        conc.load_per_task_estimates = self._real_load

    def make_gate(self, limit=None, max_limit=None, config=None):
        gate = conc.ConcurrencyGate(config or {}, op_key="test op")
        if limit is not None:
            gate.limit = limit
        if max_limit is not None:
            gate.max_limit = max_limit
        return gate

    async def settle(self):
        """Let queued tasks reach their next await."""
        for _ in range(3):
            await asyncio.sleep(0)


class GateAcquireReleaseTests(GateTestCase):
    async def test_slots_are_handed_out_up_to_the_limit(self):
        gate = self.make_gate(limit=2)
        await gate.acquire()
        await gate.acquire()
        self.assertEqual(gate.in_flight, 2)

    async def test_a_full_gate_makes_the_next_task_wait(self):
        gate = self.make_gate(limit=1)
        await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await self.settle()
        self.assertFalse(waiter.done())

        gate.release()
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(gate.in_flight, 1)

    async def test_releasing_wakes_exactly_one_waiter(self):
        gate = self.make_gate(limit=1)
        await gate.acquire()
        first = asyncio.ensure_future(gate.acquire())
        second = asyncio.ensure_future(gate.acquire())
        await self.settle()

        gate.release()
        await asyncio.wait_for(first, timeout=1)
        await self.settle()
        self.assertFalse(second.done())
        self.assertEqual(gate.in_flight, 1)

        gate.release()
        await asyncio.wait_for(second, timeout=1)

    async def test_a_limit_that_shrinks_holds_waiters_until_enough_have_drained(self):
        """The awkward case: the adapt loop halves the limit while tasks are queued.

        Tasks already running keep their slot, so the gate is over its limit until enough of
        them finish. Every release still wakes a waiter, and that waiter has to put itself back
        in the queue rather than slipping through - without losing the wakeup for the others.
        """
        gate = self.make_gate(limit=4)
        for _ in range(4):
            await gate.acquire()

        gate.limit = 2  # memory pressure
        waiter = asyncio.ensure_future(gate.acquire())
        await self.settle()

        gate.release()  # 3 in flight, still over the limit
        await self.settle()
        self.assertFalse(waiter.done())

        gate.release()  # 2 in flight, at the limit
        await self.settle()
        self.assertFalse(waiter.done())

        gate.release()  # 1 in flight, room at last
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(gate.in_flight, 2)

    async def test_a_raised_limit_lets_waiting_tasks_through(self):
        gate = self.make_gate(limit=1, max_limit=8)
        await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await self.settle()

        gate.limit = 2
        gate._wake_waiters(1)
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(gate.in_flight, 2)

    async def test_a_cancelled_waiter_passes_its_wakeup_on(self):
        """A task cancelled at the moment it is handed a slot must not swallow it.

        Losing that wakeup would leave the remaining tasks queued behind a gate with a free
        slot and nothing left to trigger it.
        """
        gate = self.make_gate(limit=1)
        await gate.acquire()
        first = asyncio.ensure_future(gate.acquire())
        second = asyncio.ensure_future(gate.acquire())
        await self.settle()

        # Hand a slot to `first` and cancel it in the same tick, before it can run
        gate.release()
        first.cancel()
        await self.settle()

        self.assertTrue(first.cancelled())
        await asyncio.wait_for(second, timeout=1)

    async def test_abort_releases_everything_queued_at_once(self):
        # Cancelling a run with hundreds of queued tasks must unwind in one step rather than
        # relying on each one being cancelled individually
        gate = self.make_gate(limit=1)
        await gate.acquire()
        waiters = [asyncio.ensure_future(gate.acquire()) for _ in range(5)]
        await self.settle()

        gate.abort()
        await self.settle()
        for waiter in waiters:
            with self.assertRaises(asyncio.CancelledError):
                await waiter

    async def test_nothing_gets_through_a_gate_that_has_been_aborted(self):
        gate = self.make_gate(limit=8)
        gate.abort()
        with self.assertRaises(asyncio.CancelledError):
            await gate.acquire()

    async def test_status_text_reports_the_gate_and_what_it_knows(self):
        gate = self.make_gate(limit=4)
        await gate.acquire()
        self.assertIn("1/4", gate.status_text())


class GateAdaptationTests(GateTestCase):
    async def test_a_saturated_gate_grows_while_memory_is_comfortable(self):
        gate = self.make_gate(limit=16, max_limit=256)
        gate.in_flight = gate.limit
        await gate._adapt_once()
        self.assertGreater(gate.limit, 16)

    async def test_an_idle_gate_does_not_grow(self):
        # If tasks are not queueing up, a bigger limit would not be used anyway
        gate = self.make_gate(limit=16, max_limit=256)
        gate.in_flight = 2
        await gate._adapt_once()
        self.assertEqual(gate.limit, 16)

    async def test_the_gate_never_grows_past_its_ceiling(self):
        gate = self.make_gate(limit=8, max_limit=9)
        gate.in_flight = gate.limit
        await gate._adapt_once()
        self.assertEqual(gate.limit, 9)

    async def test_the_limit_halves_when_free_memory_drops_below_the_reserve(self):
        gate = self.make_gate(limit=16, max_limit=256)
        gate.in_flight = gate.limit
        self.memory.available = 100 * MB
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

    async def test_the_limit_halves_when_the_process_passes_a_configured_memory_limit(self):
        gate = self.make_gate(config={"memory_limit_mb": 900})
        gate.limit, gate.max_limit = 16, 256
        gate.in_flight = gate.limit
        self.memory.rss = 1000 * MB
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

    async def test_the_limit_never_falls_below_one(self):
        gate = self.make_gate(limit=1, max_limit=256)
        self.memory.available = 100 * MB
        for _ in range(4):
            await gate._adapt_once()
        self.assertEqual(gate.limit, conc.MIN_CONCURRENCY)

    async def test_the_gate_recovers_once_the_pressure_passes(self):
        """Backing off has to be temporary, which is why the probe must report current usage.

        The whole pressure response rests on process_memory() falling again when the run's
        memory is freed. A probe that reports a high-water mark instead - which is what
        resource.ru_maxrss gives on macOS - can never fall, so the halving here would repeat
        every two seconds down to a limit of 1 and stay there for the rest of the session.
        See test_process_memory_reports_current_usage_not_a_high_water_mark.
        """
        gate = self.make_gate(config={"memory_limit_mb": 900})
        gate.limit, gate.max_limit = 16, 256
        gate.in_flight = gate.limit

        self.memory.rss = 1000 * MB
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

        self.memory.rss = 400 * MB
        gate.in_flight = gate.limit
        await gate._adapt_once()
        self.assertGreater(gate.limit, 8)

    async def test_a_measured_window_can_raise_the_ceiling(self):
        # The op turned out cheaper than the default guess, so more of it fits
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB, so 8MB/task gives a ceiling of 256
        gate = self.make_gate()
        gate.max_limit = 32

        self.memory.rss = 500 * MB
        gate.begin_window()
        self.memory.rss = 504 * MB
        gate.estimator.sample(self.memory.rss, 4)  # 1MB per task
        gate.end_window()

        self.assertEqual(gate.max_limit, conc.MAX_AUTO_CONCURRENCY)

    async def test_a_measured_window_can_lower_the_ceiling_and_the_limit_with_it(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        gate = self.make_gate(limit=200, max_limit=256)

        self.memory.rss = 500 * MB
        gate.begin_window()
        self.memory.rss = 500 * MB + 128 * MB
        gate.estimator.sample(self.memory.rss, 2)  # 64MB per task
        gate.end_window()

        self.assertEqual(gate.max_limit, 32)
        self.assertLessEqual(gate.limit, gate.max_limit)

    async def test_a_configured_maximum_survives_re_measuring(self):
        gate = self.make_gate(config={"max_concurrent_requests": 10})
        self.memory.rss = 500 * MB
        gate.begin_window()
        self.memory.rss = 501 * MB
        gate.estimator.sample(self.memory.rss, 50)  # very cheap
        gate.end_window()
        self.assertEqual(gate.max_limit, 10)

    async def test_a_window_that_measured_nothing_leaves_the_ceiling_alone(self):
        gate = self.make_gate(limit=16, max_limit=64)
        gate.begin_window()
        gate.end_window()
        self.assertEqual(gate.max_limit, 64)


class MacOSMemoryProbeTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "the peak-RSS probe is macOS-only")
    def test_process_memory_reports_current_usage_not_a_high_water_mark(self):
        """The macOS probe returns resource.ru_maxrss, which never falls once it has risen.

        Everything that reads process_memory() assumes it tracks what the process is using
        now: the gate backs off while it is above the configured limit, and the estimator
        measures per-task cost as the growth over a window's baseline. A high-water mark makes
        the first latch on forever and the second measure zero growth for the rest of the
        session.
        """
        before = conc.process_memory()
        blob = bytearray(400 * MB)
        blob[::4096] = bytes(len(blob[::4096]))  # touch every page so it is resident
        peak = conc.process_memory()
        del blob
        after = conc.process_memory()

        self.assertGreater(peak, before, "allocating did not move the probe")
        self.assertLess(after, peak, "the probe reports a peak, not current usage")


if __name__ == "__main__":
    unittest.main()
