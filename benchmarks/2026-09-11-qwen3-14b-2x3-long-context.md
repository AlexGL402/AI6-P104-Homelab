# Qwen3 14B — 2+2+2 long-context benchmark

Date: 2026-09-11

## Hardware / layout

- 6 × NVIDIA P104-100 8 GB
- Profile: 3 independent workers, 2 GPUs each
- 8081 → GPU 0,1
- 8082 → GPU 2,3
- 8083 → GPU 4,5
- Model: Qwen3 14B Q4_K_M
- Context configured: 32768
- Test prompt: 10,074 prompt tokens per worker
- Output: 512 completion tokens per worker
- All three requests ran concurrently

## Results

| Worker | Prompt tokens | Completion tokens | Prompt eval | Generation |
|---|---:|---:|---:|---:|
| 8081 | 10,074 | 512 | 109.72 tok/s | 16.17 tok/s |
| 8082 | 10,074 | 512 | 109.45 tok/s | 16.07 tok/s |
| 8083 | 10,074 | 512 | 109.64 tok/s | 16.05 tok/s |

Aggregate generation throughput: **48.29 tok/s**

Total wall time for the concurrent run: **124.832 s**

## Short-context reference

Earlier concurrent 512-token test on the same 2+2+2 layout:

- 8081: 20.24 tok/s
- 8082: 20.21 tok/s
- 8083: 20.23 tok/s
- Aggregate: **60.68 tok/s**

Long ~10k context therefore reduces generation throughput to about **48.29 tok/s aggregate**.

## Comparison / conclusion

For this AI6 host, Qwen3 14B in 2+2+2 is useful when three independent workers are required. For heavier coding-agent workloads with long context, the existing 3+3 Qwen3-Coder 30B profile remains the preferred primary profile based on prior testing (~22.9 tok/s generation on a real ~10k Aider context).

During the 2+2+2 full-load test, all six GPUs were active and the host monitor showed roughly 797 W aggregate GPU power at the observed peak.
