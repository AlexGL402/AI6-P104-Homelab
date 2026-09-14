"""
title: AI6 Token Stats Event
author: AlexGL402
version: 0.2.0
"""

from pydantic import BaseModel, Field


class Event:
    class Valves(BaseModel):
        context_window: int = Field(default=8192)
        show_context: bool = Field(default=True)
        debug: bool = Field(default=False)

    def __init__(self):
        self.valves = self.Valves()

    async def event(
        self,
        event: dict,
        __event_id__: str = None,
        __event_name__: str = None,
        __id__: str = None,
        __app__=None,
        __request__=None,
    ):
        if __event_name__ != "chat.finished":
            return

        if self.valves.debug:
            print("AI6 Token Stats Event:", event)

        chat_id = self._find_value(event, "chat_id")
        message_id = self._find_value(event, "message_id")
        user_id = self._find_value(event, "user_id")

        if not chat_id or not message_id:
            if self.valves.debug:
                print("AI6 Token Stats: missing chat_id/message_id")
            return

        # chat.finished itself only contains IDs and a short message. The actual
        # usage/timing data is stored on the finished assistant message.
        message = None
        try:
            from open_webui.models.chats import Chats

            message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
            if self.valves.debug:
                print("AI6 Token Stats Message:", message)
        except Exception as exc:
            if self.valves.debug:
                print("AI6 Token Stats message load error:", repr(exc))

        metrics = self._find_metrics_dict(message or {})
        if not metrics:
            if self.valves.debug:
                print("AI6 Token Stats: no metrics on finished message")
            return

        prompt = self._num(metrics, "input_tokens", "prompt_tokens", "prompt_n")
        output = self._num(metrics, "output_tokens", "completion_tokens", "predicted_n")
        total = self._num(metrics, "total_tokens")
        if total is None and (prompt is not None or output is not None):
            total = (prompt or 0) + (output or 0)

        speed = self._num(metrics, "predicted_per_second")
        ms = self._num(metrics, "predicted_ms", "eval_ms")
        seconds = ms / 1000.0 if ms is not None else None
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
            pct = total / self.valves.context_window * 100.0
            parts.append(f"ctx {int(total)}/{self.valves.context_window} ({pct:.0f}%)")
        if seconds is not None:
            parts.append(f"⏱ {seconds:.2f}s")
        if speed is not None:
            parts.append(f"⚡ {float(speed):.2f} tok/s")

        if not parts:
            return

        stats = " · ".join(parts)
        if self.valves.debug:
            print("AI6 Token Stats Final:", stats)

        try:
            from open_webui.socket.main import get_event_emitter

            metadata = {"chat_id": chat_id, "message_id": message_id}
            if user_id:
                metadata["user_id"] = user_id

            emitter = await get_event_emitter(metadata, update_db=False)
            if emitter:
                await emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": stats,
                            "done": True,
                            "hidden": False,
                            "action": "ai6-token-stats",
                        },
                    }
                )
        except Exception as exc:
            if self.valves.debug:
                print("AI6 Token Stats emitter error:", repr(exc))

    def _find_metrics_dict(self, obj):
        metric_keys = {
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_second",
            "input_tokens",
            "output_tokens",
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
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

    def _find_value(self, obj, key):
        if isinstance(obj, dict):
            if key in obj and obj[key] not in (None, ""):
                return obj[key]
            for child in obj.values():
                found = self._find_value(child, key)
                if found not in (None, ""):
                    return found
        elif isinstance(obj, list):
            for child in obj:
                found = self._find_value(child, key)
                if found not in (None, ""):
                    return found
        return None

    @staticmethod
    def _num(source, *keys):
        if not isinstance(source, dict):
            return None
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        return None
