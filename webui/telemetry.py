"""System telemetry sampling for the moss_trainer web UI.

Exposes :class:`SystemMonitor`, a dependency-light sampler built on ``psutil``
(required, imported unconditionally) and ``pynvml`` / nvidia-ml-py (optional:
a missing module or a missing NVIDIA driver is reported, never fatal).

Every value returned by :meth:`SystemMonitor.sample` is a real reading of the
local machine.  Anything the hardware, OS or driver does not expose is reported
as ``null`` - nothing is simulated or defaulted.

Payload shape (JSON-serialisable)::

    {
      "timestamp": "2026-10-19T12:00:00Z",                  # ISO 8601, UTC
      "cpu": {"percent": 7.1, "logical_cores": 16},
      "memory": {"used_bytes": 21000000000, "total_bytes": 34000000000, "percent": 62.3},
      "disk": {"used_bytes": 2800000000000, "total_bytes": 4096804589568, "percent": 68.4},
      "gpus": [
        {"index": 0, "name": "NVIDIA GeForce RTX 5080",
         "utilization_percent": 5, "memory_used_bytes": 1287929856,
         "memory_total_bytes": 17094934528, "memory_percent": 7.5,
         "temperature_c": 60, "power_w": 58.71, "fan_percent": 0}
      ],
      "gpu_error": null                                     # null == NVIDIA telemetry OK
    }

Semantics and caveats:

* ``cpu.percent`` is the busy share of all logical cores since the previous
  call - non-blocking, with a persistent baseline primed in the constructor and
  updated on every sample.  It is therefore correct regardless of which
  threadpool worker calls :meth:`sample` (``psutil.cpu_percent(interval=None)``
  keeps a *per-thread* baseline and would return 0.0 on freshly spawned
  workers).  The first sample after a failed prime returns ``null`` instead of
  a meaningless ``0.0``.
* ``memory.percent`` / ``disk.percent`` are psutil's OS-level usage percentages
  and ``used_bytes`` is psutil's ``used``, so ``percent`` is not necessarily
  ``used_bytes / total_bytes * 100`` (psutil accounts cached/standby memory
  differently).  GPU ``memory_percent`` is ``memory_used_bytes /
  memory_total_bytes * 100`` because NVML offers no alternative definition.
* ``disk`` describes the volume holding the trainer workspace.  If the given
  path does not exist yet, the nearest existing ancestor is measured.
* ``gpu_error`` is ``null`` whenever NVML works - including a working driver
  with zero NVIDIA devices, in which case ``gpus`` is ``[]``.  It carries a
  human-readable reason when the NVIDIA stack is unavailable (pynvml not
  installed, driver or NVML library missing) or when a device could not be read
  at all; individual unsupported readings are simply ``null`` inside ``gpus``.
* ``fan_percent`` is the raw NVML value, passed through unchanged.  Modern cards
  report ``0`` while their zero-RPM idle mode is active (verified on an RTX
  5080: ``0`` at idle, ``32`` under load), so ``0`` means "fans currently
  stopped", not "unsupported".
* NVML initialisation is lazy (first :meth:`sample`) and a failure is cached, so
  machines without NVIDIA hardware pay no per-sample cost and importing this
  module is always safe.  ``close()`` shuts NVML down and clears that cache.
* Thread-safe: :meth:`sample` and :meth:`close` serialise NVML access and the
  CPU baseline behind a single lock.  No background thread, worker or
  subprocess is started, and no shell command is executed.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

try:  # nvidia-ml-py is optional: its absence must not break the web UI.
    import pynvml
except ImportError:  # pragma: no cover - depends on the deployment environment
    pynvml = None  # type: ignore[assignment]

__all__ = ["SystemMonitor"]


def _device_name(raw: Any) -> str:
    """NVML hands device names back as a bytes buffer; decode defensively."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


class SystemMonitor:
    """Samples CPU, RAM, workspace-disk and NVIDIA GPU state on demand.

    ``disk_path`` is the trainer workspace directory whose volume is reported.
    """

    def __init__(self, disk_path: Path) -> None:
        self._disk_path = Path(disk_path)
        self._lock = threading.Lock()
        self._logical_cores = psutil.cpu_count(logical=True)
        self._prev_cpu_times = self._read_cpu_times()  # prime the baseline once
        self._nvml_ready = False
        self._nvml_failed = False
        self._nvml_error: str | None = None
        self._gpus: list[tuple[int, str, Any]] = []

    # ------------------------------------------------------------------ public

    def sample(self) -> dict[str, Any]:
        """Return one JSON-serialisable telemetry snapshot (real values only)."""
        timestamp = (
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        with self._lock:
            cpu_percent = self._cpu_percent()
            self._ensure_nvml()
            gpus, gpu_error = self._sample_gpus()
        return {
            "timestamp": timestamp,
            "cpu": {"percent": cpu_percent, "logical_cores": self._logical_cores},
            "memory": self._memory(),
            "disk": self._disk(),
            "gpus": gpus,
            "gpu_error": gpu_error,
        }

    def close(self) -> None:
        """Shut NVML down; safe to call repeatedly, a later ``sample()`` re-inits."""
        with self._lock:
            if self._nvml_ready:
                self._shutdown_nvml()
            self._nvml_ready = False
            self._nvml_failed = False
            self._nvml_error = None
            self._gpus = []

    # ------------------------------------------------------------------- cpu

    @staticmethod
    def _read_cpu_times() -> Any | None:
        try:
            return psutil.cpu_times()
        except Exception:  # noqa: BLE001 - psutil can fail on exotic platforms
            return None

    def _cpu_percent(self) -> float | None:
        """Busy share of all logical cores since the previous call (non-blocking)."""
        current = self._read_cpu_times()
        if current is None:
            return None
        previous = self._prev_cpu_times
        self._prev_cpu_times = current
        if previous is None:
            return None  # no baseline yet: unknown, not a fake 0.0

        total = 0.0
        not_busy = 0.0
        for field in current._fields:
            delta = getattr(current, field, 0.0) - getattr(previous, field, 0.0)
            # CPU time may regress on Windows; clamp like psutil/top do.
            delta = max(0.0, float(delta))
            if field in ("guest", "guest_nice"):
                continue  # Linux: already counted inside user/nice
            total += delta
            if field == "idle" or field == "iowait":
                not_busy += delta
        if total <= 0.0:
            return None
        busy = total - not_busy
        return round(min(100.0, max(0.0, busy / total * 100.0)), 1)

    # ------------------------------------------------------------ memory/disk

    @staticmethod
    def _memory() -> dict[str, Any]:
        try:
            usage = psutil.virtual_memory()
        except Exception:  # noqa: BLE001 - report unknown rather than crash the API
            return {"used_bytes": None, "total_bytes": None, "percent": None}
        return {
            "used_bytes": int(usage.used),
            "total_bytes": int(usage.total),
            "percent": round(float(usage.percent), 1),
        }

    def _usage_path(self) -> Path | None:
        if self._disk_path.exists():
            return self._disk_path
        for parent in self._disk_path.parents:
            if parent.exists():
                return parent
        return None

    def _disk(self) -> dict[str, Any]:
        target = self._usage_path()
        if target is None:
            return {"used_bytes": None, "total_bytes": None, "percent": None}
        try:
            usage = psutil.disk_usage(str(target))
        except OSError:
            return {"used_bytes": None, "total_bytes": None, "percent": None}
        return {
            "used_bytes": int(usage.used),
            "total_bytes": int(usage.total),
            "percent": round(float(usage.percent), 1),
        }

    # ------------------------------------------------------------------- gpu

    def _ensure_nvml(self) -> None:
        """Initialise NVML once, caching any failure. Caller holds ``self._lock``."""
        if self._nvml_ready or self._nvml_failed:
            return
        if pynvml is None:
            self._nvml_failed = True
            self._nvml_error = (
                "pynvml (nvidia-ml-py) is not installed; NVIDIA telemetry unavailable"
            )
            return
        try:
            pynvml.nvmlInit()
            devices = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                devices.append((index, _device_name(pynvml.nvmlDeviceGetName(handle)), handle))
        except Exception as exc:  # noqa: BLE001 - no NVIDIA stack / no driver
            self._nvml_failed = True
            self._nvml_error = f"NVML unavailable: {type(exc).__name__}: {exc}"
            self._shutdown_nvml()
            return
        self._gpus = devices
        self._nvml_ready = True

    @staticmethod
    def _shutdown_nvml() -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - never let teardown fail the caller
            pass

    @staticmethod
    def _gpu_read(call: Callable[..., Any], *args: Any) -> Any | None:
        """Single NVML query; ``None`` means this reading is unsupported/unavailable."""
        try:
            return call(*args)
        except Exception:  # noqa: BLE001 - NVMLError_NotSupported et al.
            return None

    def _sample_gpus(self) -> tuple[list[dict[str, Any]], str | None]:
        """Sample every enumerated device. Caller holds ``self._lock``."""
        gpus: list[dict[str, Any]] = []
        unreadable: list[str] = []
        for index, name, handle in self._gpus:
            utilization = self._gpu_read(pynvml.nvmlDeviceGetUtilizationRates, handle)
            memory = self._gpu_read(pynvml.nvmlDeviceGetMemoryInfo, handle)
            temperature = self._gpu_read(
                pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power_mw = self._gpu_read(pynvml.nvmlDeviceGetPowerUsage, handle)
            fan = self._gpu_read(pynvml.nvmlDeviceGetFanSpeed, handle)

            used_bytes: int | None = None
            total_bytes: int | None = None
            memory_percent: float | None = None
            if memory is not None:
                used_bytes = int(memory.used)
                total_bytes = int(memory.total)
                if total_bytes > 0:
                    memory_percent = round(used_bytes / total_bytes * 100.0, 1)

            if all(
                reading is None
                for reading in (utilization, memory, temperature, power_mw, fan)
            ):
                unreadable.append(str(index))

            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "utilization_percent": (
                        int(utilization.gpu) if utilization is not None else None
                    ),
                    "memory_used_bytes": used_bytes,
                    "memory_total_bytes": total_bytes,
                    "memory_percent": memory_percent,
                    "temperature_c": int(temperature) if temperature is not None else None,
                    "power_w": round(power_mw / 1000.0, 2) if power_mw is not None else None,
                    "fan_percent": int(fan) if fan is not None else None,
                }
            )

        error = self._nvml_error
        if error is None and unreadable:
            error = (
                "NVML device query failed for GPU index "
                + ", ".join(unreadable)
                + "; readings are null"
            )
        return gpus, error
