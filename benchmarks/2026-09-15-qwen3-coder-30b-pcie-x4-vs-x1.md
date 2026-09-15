# Qwen3-Coder 30B: PCIe Gen1 x4 vs x1 on 2× P104-100

Date: 2026-09-15
Host: Huanan / Windows 10
GPU: 2× NVIDIA P104-100 8 GB
Driver: 576.28
CUDA: 12.9
Runtime: Ollama 0.33.3
Model: `qwen3-coder:30b` (`06c1097efce0`, 18 GB)
Ollama residency: 25% CPU / 75% GPU
Context: 4096

## Test prompt

```text
Write a Python function that recursively scans a directory, calculates SHA256 for every file, skips symlinks, handles permission errors, and returns a dict sorted by path. Include type hints and a short explanation.
```

## PCIe topology

### Direct motherboard slots

- GPU 0: PCIe Gen1 x4
- GPU 1: PCIe Gen1 x4

### Mining risers

- GPU 0: PCIe Gen1 x1
- GPU 1: PCIe Gen1 x1

Both P104-100 cards report a maximum link width of x4 and Gen1.

## Results

| Metric | Direct Gen1 x4 | Riser Gen1 x1 | Change |
|---|---:|---:|---:|
| Prompt tokens | 52 | 52 | same |
| Prompt speed | 14.51 tok/s | 1.03 tok/s | ~14.1× slower |
| Generated tokens | 571 | 605 | slightly different output length |
| Generation speed | 29.66 tok/s | 9.48 tok/s | ~3.13× slower |
| Total time | 22.799 s | 114.346 s | ~5.02× longer |

## Raw timing: direct Gen1 x4

```text
prompt eval time =    3582.57 ms /    52 tokens (   68.90 ms per token,    14.51 tokens per second)
eval time        =   19216.47 ms /   571 tokens (   33.71 ms per token,    29.66 tokens per second)
total time       =   22799.04 ms /   623 tokens
```

## Raw timing: riser Gen1 x1

```text
n_gen =    586, tg =   9.51 t/s, tg_3s =   8.77 t/s
prompt eval time =   50630.14 ms /    52 tokens (  973.66 ms per token,     1.03 tokens per second)
eval time        =   63716.23 ms /   605 tokens (  105.49 ms per token,     9.48 tokens per second)
total time       =  114346.37 ms /   657 tokens
```

## Conclusion

On this 2×P104-100 setup, reducing each GPU from PCIe Gen1 x4 to Gen1 x1 causes a very large slowdown for this partially CPU-offloaded 30B model:

- prompt processing drops from **14.51 tok/s** to **1.03 tok/s**;
- generation drops from **29.66 tok/s** to **9.48 tok/s**;
- end-to-end runtime increases from **22.8 s** to **114.3 s**.

For multi-GPU LLM inference on P104-100, standard x1 mining risers are therefore a major bottleneck in this workload. Providing more PCIe lanes per card is likely to be much more valuable than adding additional x1-connected GPUs for single-stream inference.
