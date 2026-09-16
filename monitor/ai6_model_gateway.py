#!/usr/bin/env python3
"""OpenAI-compatible gateway for dynamically discovered AI6 llama.cpp workers.

The gateway reads worker state from the AI6 monitor, groups ready workers by
GGUF model path, exposes one stable model id per loaded GGUF, and load-balances
requests across workers that currently host the same model.

It also exposes two friendly dynamic aliases when matching workers exist:
- coder-fast: ready workers using exactly 2 GPUs
- coder-large-context: ready workers using exactly 3 GPUs

Each alias is pinned to one underlying model group so a single alias never
mixes different model weights.  If several model groups match, the group with
the most matching workers wins; context size is used as the next tie-breaker.
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

MONITOR_URL = os.environ.get("AI6_MONITOR_URL", "http://127.0.0.1:8090").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("AI6_GATEWAY_TIMEOUT", "1800"))
# Open WebUI may attach native function tools such as ask_user. The current
# local llama.cpp workers are used as plain chat/completion backends, so strip
# tool-control fields by default. Set AI6_GATEWAY_PASS_TOOLS=1 later if a
# model/template is deliberately configured for native tool calling.
PASS_TOOLS = os.environ.get("AI6_GATEWAY_PASS_TOOLS", "0").lower() in ("1", "true", "yes", "on")

app = FastAPI(title="AI6 Model Gateway", version="1.2.0")

_inflight: dict[int, int] = defaultdict(int)
_rr: dict[str, int] = defaultdict(int)
_lock = asyncio.Lock()


def _public_model_id(worker: dict[str, Any]) -> str:
    path = worker.get("model_path")
    if path:
        name = Path(path).name
        if name.lower().endswith(".gguf"):
            name = name[:-5]
        return name
    model = str(worker.get("model") or "llama-model")
    return model.replace("/", "_")


def _gpu_count(worker: dict[str, Any]) -> int:
    devices = worker.get("gpu_devices") or []
    return len(devices)


def _ctx(worker: dict[str, Any]) -> int:
    try:
        return int(worker.get("ctx") or 0)
    except (TypeError, ValueError):
        return 0


async def _stats() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{MONITOR_URL}/api/stats")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI6 monitor unavailable: {e}")


async def _ready_groups() -> dict[str, list[dict[str, Any]]]:
    stats = await _stats()
    workers = (stats.get("llama") or {}).get("workers") or {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p, raw in workers.items():
        w = dict(raw or {})
        if w.get("state") != "ready":
            continue
        try:
            port = int(w.get("port") or p)
        except (TypeError, ValueError):
            continue
        w["port"] = port
        w["public_model"] = _public_model_id(w)
        groups[w["public_model"]].append(w)
    return dict(groups)


def _best_group_for_gpu_count(
    groups: dict[str, list[dict[str, Any]]], gpu_count: int
) -> tuple[str, list[dict[str, Any]]] | None:
    """Pick one model group for a friendly alias.

    Never mix weights behind one alias. Prefer the model with the largest
    matching worker pool, then the largest context, then a stable model name.
    """
    choices: list[tuple[int, int, str, list[dict[str, Any]]]] = []
    for model_id, workers in groups.items():
        matching = [w for w in workers if _gpu_count(w) == gpu_count]
        if not matching:
            continue
        max_ctx = max((_ctx(w) for w in matching), default=0)
        choices.append((len(matching), max_ctx, model_id, matching))
    if not choices:
        return None
    choices.sort(key=lambda x: (-x[0], -x[1], x[2].lower()))
    _, _, model_id, workers = choices[0]
    return model_id, workers


def _virtual_aliases(
    groups: dict[str, list[dict[str, Any]]]
) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    aliases: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    fast = _best_group_for_gpu_count(groups, 2)
    if fast:
        aliases["coder-fast"] = fast
    large = _best_group_for_gpu_count(groups, 3)
    if large:
        aliases["coder-large-context"] = large
    return aliases


async def _pick_from_candidates(route_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No ready worker for model '{route_id}'")
    async with _lock:
        minimum = min(_inflight[w["port"]] for w in candidates)
        tied = [w for w in candidates if _inflight[w["port"]] == minimum]
        idx = _rr[route_id] % len(tied)
        _rr[route_id] += 1
        return tied[idx]


async def _pick_worker(model_id: str) -> dict[str, Any]:
    groups = await _ready_groups()

    # Friendly aliases such as coder-fast and coder-large-context.
    aliases = _virtual_aliases(groups)
    if model_id in aliases:
        _, candidates = aliases[model_id]
        return await _pick_from_candidates(model_id, candidates)

    # Debug/direct addressing: <model>@<port>
    if "@" in model_id:
        base, port_text = model_id.rsplit("@", 1)
        try:
            wanted_port = int(port_text)
        except ValueError:
            wanted_port = -1
        for w in groups.get(base, []):
            if w["port"] == wanted_port:
                return w

    candidates = groups.get(model_id) or []
    return await _pick_from_candidates(model_id, candidates)


def _sanitize_openwebui_body(body: dict[str, Any]) -> dict[str, Any]:
    if PASS_TOOLS:
        return body
    # Open WebUI can auto-attach built-in tool schemas. Without a deliberately
    # configured native-tool chat template the model may emit a tool call
    # (often ask_user) instead of a normal assistant answer. Remove those
    # controls for the plain-chat gateway while leaving messages untouched.
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        body.pop(key, None)
    return body


@app.get("/health")
async def health():
    groups = await _ready_groups()
    aliases = _virtual_aliases(groups)
    return {
        "status": "ok",
        "models": len(groups),
        "workers": sum(map(len, groups.values())),
        "aliases": sorted(aliases),
        "pass_tools": PASS_TOOLS,
    }


@app.get("/v1/models")
async def models():
    groups = await _ready_groups()
    data = []

    # Put friendly aliases first so Open WebUI shows the useful choices first.
    for alias, (target_model, workers) in _virtual_aliases(groups).items():
        data.append({
            "id": alias,
            "object": "model",
            "owned_by": "ai6-alias",
            "ai6_target_model": target_model,
            "ai6_workers": [w["port"] for w in workers],
            "ai6_gpus": [w.get("gpu_devices") or [] for w in workers],
            "ai6_context": max((_ctx(w) for w in workers), default=0),
        })

    for model_id, workers in sorted(groups.items()):
        data.append({
            "id": model_id,
            "object": "model",
            "owned_by": "ai6",
            "ai6_workers": [w["port"] for w in workers],
            "ai6_gpus": [w.get("gpu_devices") or [] for w in workers],
        })
        for w in workers:
            data.append({
                "id": f"{model_id}@{w['port']}",
                "object": "model",
                "owned_by": "ai6-worker",
                "ai6_workers": [w["port"]],
                "ai6_gpus": [w.get("gpu_devices") or []],
            })
    return {"object": "list", "data": data}


async def _proxy_openai(request: Request, suffix: str):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")

    model_id = str(body.get("model") or "")
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")

    worker = await _pick_worker(model_id)
    port = worker["port"]
    body = _sanitize_openwebui_body(body)
    # Use the worker's live llama.cpp model/alias, not the friendly gateway id.
    body["model"] = worker.get("model") or worker.get("public_model") or model_id.split("@", 1)[0]
    stream = bool(body.get("stream"))
    url = f"http://127.0.0.1:{port}/v1/{suffix}"

    async with _lock:
        _inflight[port] += 1

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=body)
            try:
                payload = r.json()
            except Exception:
                payload = {"error": {"message": r.text or f"HTTP {r.status_code}"}}
            return JSONResponse(payload, status_code=r.status_code)
        finally:
            async with _lock:
                _inflight[port] = max(0, _inflight[port] - 1)

    async def event_stream():
        try:
            timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=body) as r:
                    if r.status_code >= 400:
                        text = (await r.aread()).decode("utf-8", "replace")
                        yield ("data: " + json.dumps({"error": {"message": text or f"HTTP {r.status_code}"}}) + "\n\n").encode()
                        return
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            yield chunk
        finally:
            async with _lock:
                _inflight[port] = max(0, _inflight[port] - 1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy_openai(request, "chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy_openai(request, "completions")
