"""Memory-aware limiting of how much work is in flight at once.

The number of tasks waiting on API responses is what actually drives this addon's memory use,
and how many a device can afford differs a lot between a desktop and a tablet. So instead of a
fixed per-model request rate, a gate caps concurrent operations and resizes itself based on how
much memory is still free.

How much one operation costs also differs a lot between ops — a match-words task fans out per
word and can create notes and call several other ops, while translating a field is a single
request — so the cost is measured while running rather than assumed, and remembered per op for
next time.

Memory probing is stdlib-only (ctypes on Windows, /proc on Linux) because Anki's bundled Python
has no psutil.
"""

import asyncio
import ctypes
import json
import logging
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MB = 1024 * 1024

# Starting guess for what one in-flight operation costs: a worker thread's committed stack, the
# connection, the request/response buffers and the note objects the task holds onto. Only used
# until the op has been measured once; see MemoryEstimator.
DEFAULT_PER_TASK_MEMORY = 8 * MB
# Measured values are clamped to this range. Anything outside it is measurement noise rather
# than a real per-task cost.
MIN_PER_TASK_MEMORY = 512 * 1024
MAX_PER_TASK_MEMORY = 128 * MB

MIN_CONCURRENCY = 1
MIN_AUTO_CONCURRENCY = 4
# Backstop only. Memory is meant to be what limits concurrency; this just stops a very cheap
# op on a very empty machine from opening an absurd number of connections at once.
MAX_AUTO_CONCURRENCY = 256
# Where an adaptive run starts before it has grown into its ceiling
ADAPTIVE_START_CONCURRENCY = 16
# Used when memory can't be probed at all, so there is nothing to adapt against
NO_PROBE_CONCURRENCY = 8

# Fraction of the memory we're allowed to touch that goes to in-flight tasks. The rest is
# headroom, so a run doesn't drive the machine to the point where the adapt loop has to
# start backing off.
MEMORY_TARGET_FRACTION = 0.5
# ...but never more than this fraction of total RAM, so a machine that happens to be idle
# doesn't get us a limit it can't sustain once other apps want memory back
MEMORY_TOTAL_FRACTION = 0.25

# Memory to leave for everything else on the machine - the browser and editor the user has
# open alongside Anki. Both the budget and the runtime pressure check work off this, so the
# ceiling we plan for and the point we back off at agree with each other.
MIN_RESERVE_BYTES = 512 * MB
RESERVE_TOTAL_FRACTION = 0.1

ADAPT_INTERVAL_SECONDS = 2.0
# Fraction of the current limit to add per adapt tick while memory stays comfortable. Growing
# by a fixed step instead would take minutes to reach a high ceiling, so a run that is short
# or whose notes are quick would finish having never used the concurrency it could afford.
GROWTH_RATE = 0.25

# How many tasks to have queued behind the gate, as a multiple of the current limit. Some
# queue is needed: a slot must be claimed the instant one frees, and the gate can only tell
# it is the bottleneck (and so may grow) while tasks are waiting on it. Kept small because
# queued tasks hold their note and prompt in memory just like running ones.
TASK_QUEUE_DEPTH = 4


# --- Memory probes -----------------------------------------------------------------------

_probe_warning_logged = False


def _warn_probe_unavailable(what: str, error: Exception) -> None:
    global _probe_warning_logged
    if not _probe_warning_logged:
        _probe_warning_logged = True
        logger.warning(
            "Memory probing unavailable (%s: %s); concurrency will use a static limit",
            what,
            error,
        )


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_api():
    """Bind the two memory calls with explicit prototypes.

    The prototypes matter: left untyped, GetCurrentProcess's return value is truncated to a
    32-bit int and the handle it yields is rejected.
    """
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    return kernel32, psapi


def _windows_system_memory() -> Optional[tuple[int, int]]:
    kernel32, _ = _windows_api()
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error())
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _windows_process_memory() -> Optional[int]:
    kernel32, psapi = _windows_api()
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)


def _linux_system_memory() -> Optional[tuple[int, int]]:
    total = available = None
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
            if total is not None and available is not None:
                break
    if total is None or available is None:
        return None
    return total, available


def _linux_process_memory() -> Optional[int]:
    with open("/proc/self/statm", encoding="utf-8") as f:
        resident_pages = int(f.read().split()[1])
    # os.sysconf and the resource module below are POSIX-only; these branches only run when
    # sys.platform says so, but a type check on Windows can't see that
    return resident_pages * os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]


def _macos_system_memory() -> Optional[tuple[int, int]]:
    sysconf = os.sysconf  # type: ignore[attr-defined]
    total = sysconf("SC_PHYS_PAGES") * sysconf("SC_PAGE_SIZE")
    page_size = sysconf("SC_PAGE_SIZE")
    output = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
    ).stdout
    free_pages = 0
    for line in output.splitlines():
        # Free plus inactive is the closest analogue to Linux's MemAvailable
        if line.startswith("Pages free:") or line.startswith("Pages inactive:"):
            free_pages += int(line.split(":")[1].strip().rstrip("."))
    return total, free_pages * page_size


def _macos_process_memory() -> Optional[int]:
    import resource

    # ru_maxrss is bytes on macOS (kilobytes on Linux). High-water mark rather than current
    # usage, which is good enough for spotting a run that is growing.
    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
    return int(usage.ru_maxrss)


def system_memory() -> Optional[tuple[int, int]]:
    """Total and available physical memory in bytes, or None if it can't be determined."""
    try:
        if sys.platform == "win32":
            return _windows_system_memory()
        if sys.platform == "darwin":
            return _macos_system_memory()
        if sys.platform.startswith("linux"):
            return _linux_system_memory()
    except Exception as e:
        _warn_probe_unavailable("system memory", e)
        return None
    return None


def process_memory() -> Optional[int]:
    """This process's resident memory in bytes, or None if it can't be determined."""
    try:
        if sys.platform == "win32":
            return _windows_process_memory()
        if sys.platform == "darwin":
            return _macos_process_memory()
        if sys.platform.startswith("linux"):
            return _linux_process_memory()
    except Exception as e:
        _warn_probe_unavailable("process memory", e)
        return None
    return None


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "?"
    if value >= 1024 * MB:
        return f"{value / (1024 * MB):.1f} GB"
    return f"{value / MB:.0f} MB"


# --- The gate ----------------------------------------------------------------------------


def memory_reserve(total: int) -> int:
    """Memory to keep free for the rest of the machine."""
    return max(MIN_RESERVE_BYTES, int(total * RESERVE_TOTAL_FRACTION))


def memory_budget() -> Optional[float]:
    """Bytes we're willing to spend on in-flight tasks, or None if memory can't be probed.

    The reserve comes off the top before anything is budgeted, so the ceiling we size for and
    the level the adapt loop backs off at are consistent. Budgeting from raw available memory
    would happily plan a limit that immediately triggers the pressure response.
    """
    memory = system_memory()
    if not memory or not memory[0] or not memory[1]:
        return None
    total, available = memory
    spendable = max(0, available - memory_reserve(total))
    return min(spendable * MEMORY_TARGET_FRACTION, total * MEMORY_TOTAL_FRACTION)


def max_possible_concurrency(config: Optional[dict] = None) -> int:
    """The highest limit the gate could ever reach for this config.

    Used to size the shared thread pool, which is created before the gate and must still be
    big enough if a cheap op turns out to warrant a lot of concurrency. Threads are spawned
    lazily, so an over-generous ceiling costs nothing.
    """
    config = config or {}
    configured = int(config.get("max_concurrent_requests", 0) or 0)
    return max(configured, MAX_AUTO_CONCURRENCY)


def max_concurrency_for(per_task_memory: float) -> int:
    budget = memory_budget()
    if budget is None:
        return NO_PROBE_CONCURRENCY
    return int(
        min(
            MAX_AUTO_CONCURRENCY,
            max(MIN_AUTO_CONCURRENCY, budget // max(per_task_memory, MIN_PER_TASK_MEMORY)),
        )
    )


def concurrency_limits(
    config: Optional[dict] = None, per_task_memory: Optional[float] = None
) -> tuple[int, int, bool]:
    """Work out (starting limit, ceiling, adaptive) for this device.

    The ceiling comes from how much memory is free divided by what one task costs, so a tablet
    gets a lower one than a desktop, and a heavy op a lower one than a light op, without anyone
    having to configure it. A configured max_concurrent_requests lowers that ceiling further
    but does not switch adaptation off - the memory-pressure response still applies underneath
    it, which is what actually protects the machine.
    """
    config = config or {}
    configured = int(config.get("max_concurrent_requests", 0) or 0)

    if memory_budget() is None:
        # Nothing to adapt against: a conservative static limit
        static = configured if configured > 0 else NO_PROBE_CONCURRENCY
        return static, static, False

    max_limit = max_concurrency_for(per_task_memory or DEFAULT_PER_TASK_MEMORY)
    if configured > 0:
        max_limit = min(max_limit, configured)
    return min(max_limit, ADAPTIVE_START_CONCURRENCY), max_limit, True


# --- Learning what an op actually costs ---------------------------------------------------

ESTIMATES_FILE = "memory_estimates.json"
# Weight given to a new run's measurement when blending it into the stored value. Low enough
# that one unusual run doesn't throw the estimate off, high enough to track real changes.
ESTIMATE_BLEND = 0.4
# A measurement needs at least this many tasks running concurrently to mean anything
MIN_SAMPLE_IN_FLIGHT = 2


def _estimates_path() -> Path:
    # user_files is the add-on's own data directory: it survives add-on updates and is not
    # synced, which suits a per-device measurement.
    return Path(__file__).resolve().parent.parent / "user_files" / ESTIMATES_FILE


def load_per_task_estimates() -> dict:
    path = _estimates_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("Could not read memory estimates: %s", e)
        return {}


def save_per_task_estimate(op_key: str, value: float) -> None:
    path = _estimates_path()
    try:
        estimates = load_per_task_estimates()
        previous = estimates.get(op_key)
        if isinstance(previous, (int, float)) and previous > 0:
            value = previous * (1 - ESTIMATE_BLEND) + value * ESTIMATE_BLEND
        estimates[op_key] = int(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(estimates.items())), f, indent=2)
        logger.debug("Saved per-task memory estimate for %r: %s", op_key, format_bytes(int(value)))
    except Exception as e:
        logger.debug("Could not save memory estimate for %r: %s", op_key, e)


class MemoryEstimator:
    """Measures what one in-flight operation of a given op actually costs.

    Measured per window of tasks. The bulk ops process notes in windows, and between windows
    nothing is in flight, which gives a clean baseline: memory that has accumulated over the
    run so far (results dicts and the like) is absorbed into the new baseline, so only the
    growth caused by the window's concurrent tasks is attributed to per-task cost.

    Within a run the largest measurement wins, because what has to fit in RAM is the peak, not
    the average. Across runs the value is blended into the stored one so a single odd run
    doesn't skew it.
    """

    def __init__(self, op_key: Optional[str], stored: Optional[float] = None):
        self.op_key = op_key
        self.measured: Optional[float] = None
        self.estimate: float = float(stored) if stored else DEFAULT_PER_TASK_MEMORY
        self.from_measurement = bool(stored)
        self._baseline: Optional[int] = None
        self._peak_rss = 0
        self._peak_in_flight = 0

    def begin_window(self) -> None:
        """Called when nothing is in flight, to re-baseline."""
        rss = process_memory()
        self._baseline = rss
        self._peak_rss = rss or 0
        self._peak_in_flight = 0

    def sample(self, rss: Optional[int], in_flight: int) -> None:
        if rss is not None:
            self._peak_rss = max(self._peak_rss, rss)
        self._peak_in_flight = max(self._peak_in_flight, in_flight)

    def end_window(self) -> Optional[float]:
        """Fold this window into the estimate. Returns the new estimate if it changed."""
        rss = process_memory()
        self.sample(rss, 0)
        baseline = self._baseline
        self._baseline = None
        if baseline is None or self._peak_in_flight < MIN_SAMPLE_IN_FLIGHT:
            return None
        growth = self._peak_rss - baseline
        if growth <= 0:
            # Memory the allocator already held was reused; that tells us nothing new
            return None
        per_task = growth / self._peak_in_flight
        per_task = min(MAX_PER_TASK_MEMORY, max(MIN_PER_TASK_MEMORY, per_task))
        if self.measured is not None and per_task <= self.measured:
            return None
        self.measured = per_task
        previous = self.estimate
        self.estimate = per_task
        self.from_measurement = True
        logger.debug(
            "Measured per-task memory for %r: %s over %d concurrent tasks (was %s)",
            self.op_key,
            format_bytes(int(per_task)),
            self._peak_in_flight,
            format_bytes(int(previous)),
        )
        return per_task

    def persist(self) -> None:
        if self.op_key and self.measured:
            save_per_task_estimate(self.op_key, self.measured)


class ConcurrencyGate:
    """Caps how many operations run at once, resizing itself as memory allows.

    Built on a counter plus a queue of waiters rather than an asyncio.Semaphore because the
    limit has to be able to shrink: lowering it simply makes new acquires wait until enough
    in-flight work has drained. Everything runs on the event loop thread, so no locking is
    needed, and `release` is synchronous so it stays safe to call from a `finally` while the
    task is being cancelled.

    One gate per bulk run; create it inside the running event loop.
    """

    def __init__(self, config: Optional[dict] = None, op_key: Optional[str] = None):
        config = config or {}
        memory_limit_mb = int(config.get("memory_limit_mb", 0) or 0)

        total, available = system_memory() or (0, 0)
        self.total_memory = total
        self.memory_limit = memory_limit_mb * MB if memory_limit_mb > 0 else 0
        self.reserve = memory_reserve(total) if total else 0
        # A user-set ceiling the gate must never grow past, even after re-measuring
        self.configured_max = int(config.get("max_concurrent_requests", 0) or 0)

        # What this particular op cost last time it ran, if we've measured it before
        stored = load_per_task_estimates().get(op_key) if op_key else None
        self.estimator = MemoryEstimator(
            op_key, stored if isinstance(stored, (int, float)) else None
        )

        self.limit, self.max_limit, self.adaptive = concurrency_limits(
            config, self.estimator.estimate
        )

        self.in_flight = 0
        self.available_memory = available
        self._waiters: deque[asyncio.Future] = deque()
        self._adapt_task: Optional[asyncio.Task] = None
        self._aborted = False

        logger.debug(
            "ConcurrencyGate(%r): limit=%d max=%d adaptive=%s per_task=%s (%s)"
            " total_mem=%s avail_mem=%s reserve=%s hard_cap=%s",
            op_key,
            self.limit,
            self.max_limit,
            self.adaptive,
            format_bytes(int(self.estimator.estimate)),
            "measured previously" if self.estimator.from_measurement else "default guess",
            format_bytes(total or None),
            format_bytes(available or None),
            format_bytes(self.reserve or None),
            format_bytes(self.memory_limit or None),
        )

    def abort(self) -> None:
        """Stop letting anything through, and release everything queued up.

        Used on cancellation: rather than relying on each waiting task being cancelled
        individually, the gate turns every pending and future acquire into a CancelledError at
        once, so a run with hundreds of queued tasks unwinds in one step.
        """
        self._aborted = True
        queued = len(self._waiters)
        while self._waiters:
            future = self._waiters.popleft()
            if not future.done():
                future.cancel()
        logger.debug(
            "Gate abort: released %d queued, %d still holding a slot",
            queued,
            self.in_flight,
        )

    async def acquire(self) -> None:
        """Wait for a free slot. Raises CancelledError if the task is cancelled while waiting."""
        if self._aborted:
            raise asyncio.CancelledError("concurrency gate aborted")
        while self.in_flight >= self.limit:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._waiters.append(future)
            try:
                await future
            except asyncio.CancelledError:
                try:
                    self._waiters.remove(future)
                except ValueError:
                    # Already popped: we were handed a slot just as we got cancelled, so pass
                    # the wakeup on rather than losing it
                    if future.done() and not future.cancelled():
                        self._wake_waiters(1)
                raise
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1
        self._wake_waiters(1)

    def _wake_waiters(self, count: int) -> None:
        """Wake up to `count` waiters that are still waiting."""
        woken = 0
        while self._waiters and woken < count:
            future = self._waiters.popleft()
            if not future.done():
                future.set_result(None)
                woken += 1

    def start_adapting(self) -> None:
        """Begin watching memory and resizing the limit. No-op when not adaptive."""
        if not self.adaptive and not self.memory_limit:
            return
        if self._adapt_task and not self._adapt_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._adapt_task = loop.create_task(self._adapt_loop())

    def stop_adapting(self) -> None:
        if self._adapt_task:
            self._adapt_task.cancel()
            self._adapt_task = None

    def finish(self) -> None:
        """End of the run: stop adapting and remember what this op cost."""
        self.stop_adapting()
        self.estimator.persist()

    def begin_window(self) -> None:
        """Called by the bulk ops before starting a window of tasks, with nothing in flight."""
        self.estimator.begin_window()

    def end_window(self) -> None:
        """Called once a window's tasks have all finished."""
        if self.estimator.end_window() is None:
            return
        if not self.adaptive:
            # Nothing to adapt against; we still learn the cost for next time
            return
        new_max = max_concurrency_for(self.estimator.estimate)
        if self.configured_max > 0:
            new_max = min(new_max, self.configured_max)
        if new_max == self.max_limit:
            return
        logger.debug(
            "Per-task memory now %s, adjusting concurrency ceiling %d -> %d",
            format_bytes(int(self.estimator.estimate)),
            self.max_limit,
            new_max,
        )
        self.max_limit = new_max
        if self.limit > self.max_limit:
            self.limit = self.max_limit
        # A raised ceiling isn't applied at once: the adapt loop grows the limit towards it
        # only while memory stays comfortable.

    async def _adapt_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(ADAPT_INTERVAL_SECONDS)
                await self._adapt_once()
        except asyncio.CancelledError:
            pass

    async def _adapt_once(self) -> None:
        memory = system_memory()
        available = memory[1] if memory else None
        self.available_memory = available or 0
        # Sampled every tick both for the hard cap and to learn what a task costs
        rss = process_memory()
        self.estimator.sample(rss, self.in_flight)

        under_pressure = (available is not None and self.reserve and available < self.reserve) or (
            rss is not None and self.memory_limit and rss > self.memory_limit
        )

        if under_pressure:
            new_limit = max(MIN_CONCURRENCY, self.limit // 2)
            if new_limit != self.limit:
                logger.debug(
                    "Memory pressure (avail=%s rss=%s), lowering concurrency %d -> %d",
                    format_bytes(available),
                    format_bytes(rss),
                    self.limit,
                    new_limit,
                )
                # Tasks already running keep their slot; the lower limit takes effect as they
                # finish and waiting tasks stay blocked until enough have drained.
                self.limit = new_limit
            return

        # Only grow when the gate itself is the bottleneck; if tasks aren't queueing up, a
        # bigger limit wouldn't be used anyway.
        if self.adaptive and self.limit < self.max_limit and self.in_flight >= self.limit:
            previous = self.limit
            self.limit = min(self.max_limit, self.limit + max(1, int(self.limit * GROWTH_RATE)))
            self._wake_waiters(self.limit - self.in_flight)
            logger.debug(
                "Memory comfortable (avail=%s), raising concurrency %d -> %d (ceiling %d)",
                format_bytes(available),
                previous,
                self.limit,
                self.max_limit,
            )

    def status_text(self) -> str:
        """Short description of the gate's state, for the progress dialog."""
        text = f"{self.in_flight}/{self.limit}"
        if self.available_memory:
            text += f" | Free memory: {format_bytes(self.available_memory)}"
        if self.estimator.measured:
            text += f" | {format_bytes(int(self.estimator.measured))}/task"
        return text
