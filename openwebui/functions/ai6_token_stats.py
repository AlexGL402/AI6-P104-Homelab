"""
title: AI6 Token Stats
author: AlexGL402
version: 0.2.0
"""

import time
from typing import Optional
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0)
        context_window: int = Field(default=8192)
        show_context: bool = Field(default=True)

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __metadata__: Optional[dict] = None, **kwargs) -> dict:
        if body.get("stream", True):
            body.setdefault("stream_options", {})["include_usage"] = True
        if __metadata__ is not None:
            __metadata__["ai6_stats_start"] = time.perf_counter()
        return body

    async def outlet(self, body: dict, __event_emitter__=None, __metadata__: Optional[dict] = None, **kwargs) -> dict:
        flat = self._find_metrics_dict(body)
        usage = self._find_dict(body, "usage")
        timings = self._find_dict(body, "timings")

        prompt = self._pick(flat, usage, timings, keys=("input_tokens", "prompt_tokens", "prompt_n"))
        output = self._pick(flat, usage, timings, keys=("output_tokens", "completion_tokens", "predicted_n"))
        total = self._pick(flat, usage, timings, keys=("total_tokens",))
        if total is None and (prompt is not None or output is not None):
            total = (prompt or 0) + (output or 0)

        speed = self._pick(flat, usage, timings, keys=("predicted_per_second",))
        ms = self._pick(flat, usage, timings, keys=("predicted_ms", "eval_ms"))
        seconds = ms / 1000.0 if ms else None
        if seconds is None and __metadata__ and __metadata__.get("ai6_stats_start"):
            seconds = time.perf_counter() - __metadata__["ai6_stats_start"]
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
            parts.append(f"⏱ {seconds:.2f}s")
        if speed is not None:
            parts.append(f"⚡ {float(speed):.2f} t/s")

        if parts and __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": " · ".join(parts),
                    "done": True,
                    "hidden": False
                }
            })
        return body

    def _find_metrics_dict(self, obj):
        metric_keys = {
            "cache_n", "prompt_n", "prompt_ms", "prompt_per_second",
            "predicted_n", "predicted_ms", "predicted_per_second",
            "input_tokens", "output_tokens", "total_tokens"
        }
        best = {}
        if isinstance(obj, dict):
            if any(k in obj for k in metric_keys):
                best = obj
            for child in obj.values():
                nested = self._find_metrics_dict(child)
                if sum(k in nested for k in metric_keys) > sum(k in best for k in metric_keys):
                    best = nested
        elif isinstance(obj, list):
            for child in obj:
                nested = self._find_metrics_dict(child)
                if sum(k in nested for k in metric_keys) > sum(k in best for k in metric_keys):
                    best = nested
        return best

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
    def _pick(*sources, keys):
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return value
        return None
