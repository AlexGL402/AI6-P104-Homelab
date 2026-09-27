#!/usr/bin/env python3
"""
Fixed vLLM chat benchmark used for AI6/CMP/P104 comparisons.

Defaults reproduce the 2026-09-27 Qwen3-14B-AWQ test:
- 12 concurrent requests
- 512 output tokens/request
- temperature 0
- thinking disabled
- fixed coding prompt
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = (
    "Write a production-quality Python implementation of an asynchronous HTTP crawler "
    "with retries, timeout handling, URL deduplication, SHA256 hashing, logging, "
    "graceful shutdown, and type hints."
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8013/v1/chat/completions")
    p.add_argument("--model", default="/home/ai6/models/vllm/Qwen3-14B-AWQ")
    p.add_argument("-n", "--requests", type=int, default=12)
    p.add_argument("-o", "--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=600)
    return p.parse_args()

def main():
    args = parse_args()

    def one(i):
        data = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        req = urllib.request.Request(
            args.url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            x = json.loads(r.read())
        dt = time.time() - t0
        u = x.get("usage", {})
        return {
            "id": i,
            "seconds": dt,
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.requests) as ex:
        futures = [ex.submit(one, i) for i in range(args.requests)]
        results = [f.result() for f in as_completed(futures)]

    wall = time.time() - t0
    prompt = sum(x["prompt_tokens"] for x in results)
    out = sum(x["completion_tokens"] for x in results)

    print(f"requests      : {args.requests}")
    print(f"prompt tokens : {prompt}")
    print(f"output tokens : {out}")
    print(f"wall time     : {wall:.3f} s")
    print(f"aggregate     : {out / wall:.2f} tok/s")
    print(f"per req       : {out / args.requests / wall:.2f} tok/s")

if __name__ == "__main__":
    main()
