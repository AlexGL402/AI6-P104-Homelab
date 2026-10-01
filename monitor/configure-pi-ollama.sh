#!/usr/bin/env bash
set -Eeuo pipefail
python3 - <<'PY'
import json
from pathlib import Path
p=Path.home()/".pi/agent/models.json"
data=json.loads(p.read_text()) if p.exists() else {}
providers=data.setdefault("providers", data.get("providers", {})) if "providers" in data else data
prov=providers.setdefault("ai6-ollama", {})
prov.update({"api":"openai-completions","baseUrl":"http://127.0.0.1:11434/v1","apiKey":"ollama"})
models={m.get("id"):m for m in prov.get("models",[]) if isinstance(m,dict) and m.get("id")}
models["qwen3:8b"]={"id":"qwen3:8b","name":"Qwen3 8B Ollama","contextWindow":16384,"maxTokens":4096,"reasoning":False}
models["qwen3:14b"]={"id":"qwen3:14b","name":"Qwen3 14B Ollama","contextWindow":16384,"maxTokens":4096,"reasoning":False}
prov["models"]=list(models.values())
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
print("Pi Ollama models configured:", ", ".join(models))
PY
