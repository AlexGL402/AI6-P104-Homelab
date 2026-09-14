"""
title: AI6 Token Stats
author: AlexGL402
version: 0.1.0
"""

import time
from typing import Optional
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0)
        context_window: int = Field(default=8192)
        show_context: bool = Field(default=False)

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __metadata__: Optional[dict] = None, **kwargs) -> dict:
        if body.get("stream", True):
            body.setdefault("stream_options", {})["include_usage"] = True
        if __metadata__ is not None:
            __metadata__["ai6_stats_start"] = time.perf_counter()
        return body

    async def outlet(self, body: dict, __event_emitter__=None, __metadata__: Optional[dict] = None, **kwargs) -> dict:
        usage = self._find_dict(body, "usage")
        timings = self._find_dict(body, "timings")

        prompt = self._num(usage, "prompt_tokens", "input_tokens")
        output = self._num(usage, "completion_tokens", "output_tokens")
        total = self._num(usage, "total_tokens")

        if prompt is None:
            prompt = self._num(timings, "prompt_n")
        if output is None:
            output = self._num(timings, "predicted_n")
        if total is None and (prompt is not None or output is not None):
            total = (prompt or 0) + (output or 0)

        seconds = None
        ms = self._num(timings, "predicted_ms", "eval_ms")
        if ms:
            seconds = ms / 1000.0
        elif __metadata__ and __metadata__.get("ai6_stats_start"):
            seconds = time.perf_counter() - __metadata__["ai6_stats_start"]

        speed = self._num(timings, "predicted_per_second")
        if speed is None and output and seconds:
            speed = output / seconds

        parts = []
        if prompt is not None:
            parts.append(f"↑ {int(prompt)}")
        if output is not None:
            parts.append(f"↓ {int(output)}")
        if total is not None:
            parts.append(f"Σ {int(total)}")
        if self.valves.show_context and total is not None and self.valves.context_window:
            pct = total / self.valves.context_window * 100
            parts.append(f"ctx {int(total)}/{self.valves.context_window} ({pct:.0f}%)")
        if seconds is not None:
            parts.append(f"⏱ {seconds:.1f}s")
        if speed is not None:
            parts.append(f"⚡ {float(speed):.2f} t/s")

        if parts and __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": " · ".join(parts),
                    "done": True,
                    "hidden": False,
                    "action": "ai6-token-stats"
                }
            })
        return body

    def _find_dict(self, obj, key):
        found = {}
        if isinstance(obj, dict):
            value = obj.get(key)
            if isinstance(value, dict):
                found = value
            for child in obj.values():
                nested = self._find_dict(child, key)
                if len(nested) > len(found):
                    found = nested
        elif isinstance(obj, list):
            for child in obj:
                nested = self._find_dict(child, key)
                if len(nested) > len(found):
                    found = nested
        return found

    @staticmethod
    def _num(source, *keys):
        if not isinstance(source, dict):
            return None
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        return None
