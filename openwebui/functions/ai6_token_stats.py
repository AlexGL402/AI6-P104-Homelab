"""
title: AI6 Token Stats
author: AlexGL402
version: 0.5.0
"""

import json
import time
import urllib.request
from typing import Optional
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0)
        context_window: int = Field(default=8192)
        show_context: bool = Field(default=False)
        tokenizer_url: str = Field(
            default="http://host.docker.internal:8012/tokenize"
        )
        model: str = Field(
            default="/home/ai6/models/vllm/Qwen3-14B-AWQ"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._starts = {}

    def _key(self, __user__):
        if isinstance(__user__, dict):
            return str(__user__.get("id") or __user__.get("email") or "default")
        return "default"

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if body.get("stream", True):
            body.setdefault("stream_options", {})["include_usage"] = True

        self._starts[self._key(__user__)] = time.perf_counter()
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        key = self._key(__user__)
        started = self._starts.pop(key, None)

        usage = self._find_dict(body, "usage")
        timings = self._find_dict(body, "timings")

        prompt = self._pick(
            usage, timings,
            keys=("prompt_tokens", "input_tokens", "prompt_n")
        )
        output = self._pick(
            usage, timings,
            keys=("completion_tokens", "output_tokens", "predicted_n")
        )
        total = self._pick(
            usage, timings,
            keys=("total_tokens",)
        )

        ms = self._pick(
            timings,
            keys=("predicted_ms", "eval_ms")
        )
        seconds = ms / 1000.0 if ms is not None else None

        speed = self._pick(
            timings,
            keys=("predicted_per_second",)
        )

        messages = body.get("messages", [])
        assistant_text = self._assistant_text(messages)

        if output is None and assistant_text:
            output = self._token_count(assistant_text)

        if seconds is None and started is not None:
            seconds = time.perf_counter() - started

        if total is None and prompt is not None and output is not None:
            total = prompt + output

        if speed is None and output is not None and seconds and seconds > 0:
            speed = output / seconds

        parts = []

        if prompt is not None:
            parts.append(f"↑ {int(prompt)}")

        if output is not None:
            parts.append(f"↓ {int(output)}")

        if total is not None:
            parts.append(f"Σ {int(total)}")

        if (
            self.valves.show_context
            and total is not None
            and self.valves.context_window > 0
        ):
            pct = total / self.valves.context_window * 100.0
            parts.append(
                f"ctx {int(total)}/{self.valves.context_window} ({pct:.0f}%)"
            )

        if seconds is not None:
            parts.append(f"⏱ {seconds:.2f}s")

        if speed is not None:
            parts.append(f"⚡ {float(speed):.2f} tok/s")

        if parts and isinstance(messages, list):
            stats = " · ".join(parts)

            for msg in reversed(messages):
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and isinstance(msg.get("content"), str)
                ):
                    if "⚡" not in msg["content"][-200:]:
                        msg["content"] += f"\n\n---\n`{stats}`"
                    break

            return {"messages": messages}

        return body

    def _token_count(self, text):
        try:
            payload = json.dumps({
                "model": self.valves.model,
                "prompt": text
            }).encode("utf-8")

            req = urllib.request.Request(
                self.valves.tokenizer_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))

            count = data.get("count")
            if isinstance(count, int):
                return count
        except Exception:
            pass

        return None

    @staticmethod
    def _assistant_text(messages):
        if not isinstance(messages, list):
            return ""

        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and isinstance(msg.get("content"), str)
            ):
                return msg["content"]

        return ""

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
