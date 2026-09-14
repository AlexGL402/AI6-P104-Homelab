# Open WebUI helpers for AI6

## AI6 Token Stats

`functions/ai6_token_stats.py` is an Open WebUI Filter that adds a compact status line below each assistant reply, for example:

```text
↑ 120 · ↓ 399 · Σ 519 · ⏱ 16.0s · ⚡ 24.94 t/s
```

### Install

1. Pull the repository on AI6.
2. Open Open WebUI → Admin Panel / Workspace → Functions.
3. Create a new **Filter** function.
4. Paste the full contents of `openwebui/functions/ai6_token_stats.py`.
5. Save and enable it for `qwen38-01`, `qwen38-23`, and `qwen38-45`.
6. In each model's Capabilities, enable **Usage** and **Status Updates**.

The filter also requests `stream_options.include_usage=true`, but Open WebUI's Usage capability should still be enabled so provider usage metadata is preserved.

For the current 2+2+2 profile the context window is 8192. Set the filter valve `context_window=8192`; enable `show_context` if you also want context utilization displayed.
