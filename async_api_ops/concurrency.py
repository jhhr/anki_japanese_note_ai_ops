"""Memory-aware limiting of how much work is in flight at once.

The number of tasks waiting on API responses is what actually drives this addon's memory use,
and how many a device can afford differs a lot between a desktop and a tablet. So instead of a
fixed per-model request rate, a gate caps concurrent operations and resizes itself based on how
much memory is still free.

Memory probing is stdlib-only (ctypes on Windows, /proc on Linux) because Anki's bundled Python
has no psutil.
"""

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

MB = 1024 * 1024

# Rough memory cost of one in-flight operation: a worker thread's committed stack, the
# connection, the request/response buffers and the note objects the task holds onto.
PER_TASK_MEMORY_ESTIMATE = 8 * MB

MIN_CONCURRENCY = 1
MIN_AUTO_CONCURRENCY = 4
MAX_AUTO_CONCURRENCY = 64
INITIAL_AUTO_CONCURRENCY = 8

# Fraction of currently-available memory we're willing to spend on in-flight tasks
MEMORY_TARGET_FRACTION = 0.5
# ...but never more than this fraction of total RAM, so a machine that happens to be idle
# doesn't get us a limit it can't sustain once other apps want memory back
MEMORY_TOTAL_FRACTION = 0.25

MIN_RESERVE_BYTES = 512 * MB
RESERVE_TOTAL_FRACTION = 0.1

ADAPT_INTERVAL_SECONDS = 2.0

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


def concurrency_limits(config: Optional[dict] = None) -> tuple[int, int, bool]:
    """Work out (starting limit, ceiling, adaptive) for this device.

    A configured max_concurrent_requests is used verbatim and turns adaptation off. Otherwise
    the ceiling comes from how much memory is free, so a tablet gets a lower one than a
    desktop without anyone having to configure it.
    """
    config = config or {}
    configured = int(config.get("max_concurrent_requests", 0) or 0)
    if configured > 0:
        return configured, configured, False

    memory = system_memory()
    if not memory or not memory[0] or not memory[1]:
        # No probe available: a conservative static limit
        return INITIAL_AUTO_CONCURRENCY, INITIAL_AUTO_CONCURRENCY, False

    total, available = memory
    budget = min(available * MEMORY_TARGET_FRACTION, total * MEMORY_TOTAL_FRACTION)
    max_limit = int(
        min(
            MAX_AUTO_CONCURRENCY,
            max(MIN_AUTO_CONCURRENCY, budget // PER_TASK_MEMORY_ESTIMATE),
        )
    )
    return min(max_limit, INITIAL_AUTO_CONCURRENCY), max_limit, True


class ConcurrencyGate:
    """Caps how many operations run at once, resizing itself as memory allows.

    Built on a counter plus a queue of waiters rather than an asyncio.Semaphore because the
    limit has to be able to shrink: lowering it simply makes new acquires wait until enough
    in-flight work has drained. Everything runs on the event loop thread, so no locking is
    needed, and `release` is synchronous so it stays safe to call from a `finally` while the
    task is being cancelled.

    One gate per bulk run; create it inside the running event loop.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        memory_limit_mb = int(config.get("memory_limit_mb", 0) or 0)

        total, available = system_memory() or (0, 0)
        self.total_memory = total
        self.memory_limit = memory_limit_mb * MB if memory_limit_mb > 0 else 0
        self.reserve = max(MIN_RESERVE_BYTES, int(total * RESERVE_TOTAL_FRACTION)) if total else 0

        self.limit, self.max_limit, self.adaptive = concurrency_limits(config)

        self.in_flight = 0
        self.available_memory = available
        self._waiters: deque[asyncio.Future] = deque()
        self._adapt_task: Optional[asyncio.Task] = None

        logger.debug(
            "ConcurrencyGate: limit=%d max=%d adaptive=%s total_mem=%s avail_mem=%s"
            " reserve=%s hard_cap=%s",
            self.limit,
            self.max_limit,
            self.adaptive,
            format_bytes(total or None),
            format_bytes(available or None),
            format_bytes(self.reserve or None),
            format_bytes(self.memory_limit or None),
        )

    async def acquire(self) -> None:
        """Wait for a free slot. Raises CancelledError if the task is cancelled while waiting."""
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
        rss = process_memory() if self.memory_limit else None

        under_pressure = (
            available is not None and self.reserve and available < self.reserve
        ) or (rss is not None and self.memory_limit and rss > self.memory_limit)

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
            self.limit += 1
            self._wake_waiters(self.limit - self.in_flight)
            logger.debug(
                "Memory comfortable (avail=%s), raising concurrency to %d",
                format_bytes(available),
                self.limit,
            )

    def status_text(self) -> str:
        """Short description of the gate's state, for the progress dialog."""
        text = f"{self.in_flight}/{self.limit}"
        if self.available_memory:
            text += f" | Free memory: {format_bytes(self.available_memory)}"
        return text
