# Agentic coding tests

This file records real end-to-end coding-agent tests performed through:

`Qwen3-Coder 30B -> Open WebUI -> AI6 Terminal -> Linux container -> files/processes/network`

## Test 1 — create and execute Python file

**Task**

Create `hello.py` containing:

```python
print("Hello from AI6 Terminal")
```

Run it and report the actual output.

**Result:** PASS

Observed tool flow:

- `write_file`
- `run_command`
- `get_process_status`

Verified output:

```text
Hello from AI6 Terminal
```

The file was also visible in the Open Terminal file browser.

## Test 2 — create and validate FastAPI application

**Task**

Create a small FastAPI project, install/check dependencies, launch it, test it, and recover from errors automatically.

**Result:** PASS

The model:

- created `fastapi_project/`
- created `main.py`
- created `requirements.txt`
- checked/installed dependencies
- attempted to start the server
- detected that port 8000 was occupied by Open Terminal
- recovered by using another port
- tested the API with HTTP requests

Verified from a separate Windows PC on the LAN:

- `http://<AI6_LAN_IP>:8001/` returned `{"Hello":"World"}`
- `http://<AI6_LAN_IP>:8001/docs` opened FastAPI Swagger UI

The actual LAN address is intentionally not stored in this document because AI6 may receive a different address after moving to another network. This confirmed real network access from the generated application, not merely an internal tool result.

## Test 3 — modify an existing project

**Task**

Add `POST /users` with a Pydantic model to the existing FastAPI project, update tests, restart the application, and verify the endpoint.

**Result:** PASS

Observed agent behavior:

- read existing `main.py`
- edited existing source rather than recreating the project
- added a Pydantic `User` model
- added `POST /users`
- updated `test_api.py`
- attempted server restart
- encountered port 8001 still in use
- recovered by launching on port 8002
- ran API tests and verified the new endpoint
- returned LAN, Swagger, and ReDoc URLs

Human intervention during the coding task itself: **none**.

## Current weakness observed

Port/process handling is functional but somewhat inefficient. The model sometimes tries several shell commands before determining whether a port is occupied, then falls back to a different port.

A system-prompt rule was added to prefer:

- `ss -ltn`
- or a short Python socket check
- first-free-port selection from 8001-8010
- verification of the exact listening port after launch
- an actual HTTP request before claiming success
- dynamic detection of the current LAN address before returning browser URLs

## Assessment

The system has passed basic agentic-development checks:

`inspect -> edit -> execute -> observe -> recover -> test -> report`

This is materially different from simple code generation: the model can operate on an existing filesystem, run commands, diagnose a failure, modify its approach, and validate the final result.
