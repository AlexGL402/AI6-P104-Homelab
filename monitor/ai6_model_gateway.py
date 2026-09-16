#!/usr/bin/env python3
"""OpenAI-compatible gateway for dynamically discovered AI6 llama.cpp workers.

The gateway reads worker state from the AI6 monitor, groups ready workers by
GGUF model path, exposes one stable model id per loaded GGUF, and load-balances
requests across workers that currently host the same model.
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
# Open WebUI may attach native function tools such as ask_user.  The current
# local llama.cpp workers are used as plain chat/completion backends, so strip
# tool-control fields by default.  Set AI6_GATEWAY_PASS_TOOLS=1 later if a
# model/template is deliberately configured for native tool calling.
PASS_TOOLS = os.environ.get("AI6_GATEWAY_PASS_TOOLS", "0").lower() in ("1", "true", "yes", "on")

app = FastAPI(title="AI6 Model Gateway", version="1.1.0")

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


async def _pick_worker(model_id: str) -> dict[str, Any]:
    groups = await _ready_groups()

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
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No ready worker for model '{model_id}'")

    async with _lock:
        minimum = min(_inflight[w["port"]] for w in candidates)
        tied = [w for w in candidates if _inflight[w["port"]] == minimum]
        idx = _rr[model_id] % len(tied)
        _rr[model_id] += 1
        return tied[idx]


def _sanitize_openwebui_body(body: dict[str, Any]) -> dict[str, Any]:
    if PASS_TOOLS:
        return body
    # Open WebUI can auto-attach built-in tool schemas.  Without a deliberately
    # configured native-tool chat template the model may emit a tool call
    # (often ask_user) instead of a normal assistant answer.  Remove those
    # controls for the plain-chat gateway while leaving messages untouched.
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        body.pop(key, None)
    return body


@app.get("/health")
async def health():
    groups = await _ready_groups()
    return {
        "status": "ok",
        "models": len(groups),
        "workers": sum(map(len, groups.values())),
        "pass_tools": PASS_TOOLS,
    }


@app.get("/v1/models")
async def models():
    groups = await _ready_groups()
    data = []
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
    body["model"] = worker.get("model") or model_id.split("@", 1)[0]
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
