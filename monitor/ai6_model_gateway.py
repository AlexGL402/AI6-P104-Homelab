#!/usr/bin/env python3
"""OpenAI-compatible gateway for dynamically discovered AI6 llama.cpp workers.

The gateway reads worker state from the AI6 monitor, groups ready workers by
GGUF model path, exposes stable model ids and friendly aliases, and
load-balances requests across ready workers that host the same model.

Friendly aliases include:
- coder-fast: ready workers using exactly 2 GPUs
- coder-large-context: ready workers using exactly 3 GPUs
- coding / reasoning / chat: role-oriented aliases chosen from loaded models

Open WebUI tool handling is conservative by default: only explicitly allowed
web-search tools are forwarded to llama.cpp. This prevents unrelated built-in
tools such as ask_user from hijacking plain chat responses while still allowing
agentic web search when Open WebUI attaches search_web/fetch_url tools.
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
PASS_TOOLS = os.environ.get("AI6_GATEWAY_PASS_TOOLS", "0").lower() in ("1", "true", "yes", "on")

# Default Open WebUI web-related function names. Override with a comma-separated
# list through AI6_GATEWAY_ALLOWED_TOOLS if a future Open WebUI release renames
# its web tools.
_ALLOWED_TOOLS_RAW = os.environ.get(
    "AI6_GATEWAY_ALLOWED_TOOLS",
    "search_web,fetch_url,web_search,fetch_webpage",
)
ALLOWED_TOOLS = {x.strip() for x in _ALLOWED_TOOLS_RAW.split(",") if x.strip()}

app = FastAPI(title="AI6 Model Gateway", version="1.4.0")

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
    return len(worker.get("gpu_devices") or [])


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


def _score_role(model_id: str, role: str) -> int:
    name = model_id.lower()
    score = 0
    if role == "coding":
        if "coder" in name:
            score += 100
        if "qwen2.5" in name or "qwen3" in name:
            score += 20
        if "14b" in name or "16b" in name:
            score += 10
    elif role == "reasoning":
        for token, points in (("32b", 35), ("30b", 33), ("27b", 30), ("22b", 24), ("14b", 12)):
            if token in name:
                score += points
        if "coder" not in name:
            score += 10
    elif role == "chat":
        if "coder" not in name:
            score += 25
        if "14b" in name or "9b" in name or "8b" in name:
            score += 15
    return score


def _best_group_for_role(
    groups: dict[str, list[dict[str, Any]]], role: str
) -> tuple[str, list[dict[str, Any]]] | None:
    choices: list[tuple[int, int, int, str, list[dict[str, Any]]]] = []
    for model_id, workers in groups.items():
        if not workers:
            continue
        role_score = _score_role(model_id, role)
        max_ctx = max((_ctx(w) for w in workers), default=0)
        choices.append((role_score, len(workers), max_ctx, model_id, workers))
    if not choices:
        return None
    choices.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3].lower()))
    score, _, _, model_id, workers = choices[0]
    if score <= 0:
        return None
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

    for role in ("coding", "reasoning", "chat"):
        match = _best_group_for_role(groups, role)
        if match:
            aliases[role] = match

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
    aliases = _virtual_aliases(groups)
    if model_id in aliases:
        _, candidates = aliases[model_id]
        return await _pick_from_candidates(model_id, candidates)

    if "@" in model_id:
        base, port_text = model_id.rsplit("@", 1)
        try:
            wanted_port = int(port_text)
        except ValueError:
            wanted_port = -1
        for w in groups.get(base, []):
            if w["port"] == wanted_port:
                return w

    return await _pick_from_candidates(model_id, groups.get(model_id) or [])


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    # Some clients may emit a flatter schema.
    if tool.get("name"):
        return str(tool["name"])
    return None


def _sanitize_openwebui_body(body: dict[str, Any]) -> dict[str, Any]:
    if PASS_TOOLS:
        return body

    tools = body.get("tools")
    if not isinstance(tools, list):
        body.pop("tools", None)
        body.pop("tool_choice", None)
        body.pop("parallel_tool_calls", None)
        return body

    allowed = []
    for tool in tools:
        name = _tool_name(tool)
        if name and name in ALLOWED_TOOLS:
            allowed.append(tool)

    if allowed:
        body["tools"] = allowed
        # If Open WebUI forced a now-filtered tool by name, fall back to auto.
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            fn = choice.get("function") or {}
            forced_name = fn.get("name") if isinstance(fn, dict) else None
            if forced_name and forced_name not in ALLOWED_TOOLS:
                body["tool_choice"] = "auto"
        elif isinstance(choice, str) and choice not in ("auto", "none", "required"):
            body["tool_choice"] = "auto"
    else:
        body.pop("tools", None)
        body.pop("tool_choice", None)
        body.pop("parallel_tool_calls", None)

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
        "allowed_tools": sorted(ALLOWED_TOOLS),
    }


@app.get("/v1/models")
async def models():
    groups = await _ready_groups()
    data = []

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
