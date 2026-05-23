"""Communication profiler for MoE dispatch operations.

Provides:
  - CommProfiler: hooks into dist.all_to_all calls to record timing/sizes.
  - export_chrome_trace: exports a JSON file loadable in chrome://tracing.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommEvent:
    """A single communication event."""

    name: str
    start_ms: float
    end_ms: float
    sent_bytes: int = 0
    recv_bytes: int = 0
    src_rank: int = 0
    dst_rank: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class CommProfiler:
    """Lightweight profiler that records communication timeline."""

    def __init__(self):
        self.events: list[CommEvent] = []
        self._start_time = time.perf_counter()
        self._pending: dict[str, float] = {}

    def reset(self):
        self.events.clear()
        self._pending.clear()
        self._start_time = time.perf_counter()

    def start_event(self, name: str) -> None:
        self._pending[name] = time.perf_counter()

    def end_event(self, name: str, **extra) -> CommEvent:
        now = time.perf_counter()
        start = self._pending.pop(name, now)
        event = CommEvent(
            name=name,
            start_ms=(start - self._start_time) * 1000,
            end_ms=(now - self._start_time) * 1000,
            **extra,
        )
        self.events.append(event)
        return event

    @contextmanager
    def profile(self, name: str, **extra):
        self.start_event(name)
        try:
            yield
        finally:
            self.end_event(name, **extra)

    def summary(self) -> dict[str, Any]:
        if not self.events:
            return {}
        total = sum(e.duration_ms for e in self.events)
        comm_events = [e for e in self.events if "comm" in e.name.lower() or "all_to_all" in e.name.lower()]
        comm_total = sum(e.duration_ms for e in comm_events)
        comp_total = total - comm_total
        return {
            "total_time_ms": total,
            "comm_time_ms": comm_total,
            "comp_time_ms": comp_total,
            "comm_ratio": comm_total / total if total > 0 else 0,
            "num_events": len(self.events),
            "num_comm_events": len(comm_events),
        }

    def to_chrome_trace(self) -> list[dict]:
        """Convert events to Chrome Trace Event format."""
        traces = []
        for e in self.events:
            traces.append({
                "name": e.name,
                "cat": "comm" if "all_to_all" in e.name.lower() else "compute",
                "ph": "X",
                "ts": e.start_ms * 1000,  # microseconds
                "dur": e.duration_ms * 1000,
                "pid": e.src_rank,
                "tid": e.dst_rank,
                "args": {
                    "sent_bytes": e.sent_bytes,
                    "recv_bytes": e.recv_bytes,
                },
            })
        return traces


def export_chrome_trace(profiler: CommProfiler, filepath: str) -> None:
    """Export profiler events as a Chrome Trace JSON file."""
    traces = profiler.to_chrome_trace()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": traces, "displayTimeUnit": "ms"}, f, indent=2)
