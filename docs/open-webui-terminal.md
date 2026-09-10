# Open WebUI + Open Terminal

This setup turns the local Qwen3-Coder workers into practical coding agents that can read/write files, run commands, start development servers, inspect failures, and verify results.

## Open WebUI

Open WebUI runs in Docker and is published on LAN port 3000.

```bash
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  -e WEBUI_SECRET_KEY="<secret>" \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

LAN URL:

```text
http://<AI6_LAN_IP>:3000
```

Get the current LAN IP from the default route:

```bash
ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}'
```

## Open Terminal

Open Terminal runs as a separate Docker container with persistent home storage.

```bash
docker run -d \
  --name open-terminal \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 8001-8010:8001-8010 \
  --memory 2g \
  --cpus 2 \
  -v open-terminal:/home/user \
  -e OPEN_TERMINAL_API_KEY="<secret>" \
  ghcr.io/open-webui/open-terminal
```

Ports:

- 8000: Open Terminal API
- 8001-8010: development applications launched by the coding agent

Do not commit the real API key to Git.

## Open WebUI terminal connection

In Open WebUI:

```text
Admin -> Settings -> Integrations -> Open Terminal
```

Portable connection configuration:

```text
URL: http://host.docker.internal:8000
Auth: Bearer
API key: <OPEN_TERMINAL_API_KEY>
```

`host.docker.internal` is mapped to the Docker host through `host-gateway`, so this configuration keeps working if the AI6 machine receives a different LAN address.

Verified from inside the Open WebUI container:

```bash
docker exec open-webui getent hosts host.docker.internal
docker exec open-webui curl -s -o /dev/null -w 'HTTP %{http_code}\n' \
  http://host.docker.internal:8000/openapi.json
```

Expected result: the hostname resolves to the Docker host gateway and the HTTP check returns `200`.

## Authentication troubleshooting

A healthy authenticated Open Terminal endpoint can be verified with:

```bash
KEY=$(docker inspect open-terminal \
  --format '{{range .Config.Env}}{{println .}}{{end}}' |
  sed -n 's/^OPEN_TERMINAL_API_KEY=//p')

curl -i \
  -H "Authorization: Bearer $KEY" \
  http://127.0.0.1:8000/system
```

Expected response: HTTP 200.

The same test from the Open WebUI container:

```bash
docker exec open-webui curl -i \
  -H "Authorization: Bearer $KEY" \
  http://host.docker.internal:8000/system
```

## Verified coding-agent workflow

The setup has successfully completed end-to-end tasks where Qwen3-Coder:

1. inspected an existing FastAPI project;
2. created and modified source files;
3. created tests;
4. launched uvicorn;
5. detected a port conflict;
6. restarted on a different port;
7. performed real HTTP checks;
8. returned working LAN and Swagger URLs.

Example development URLs:

```text
http://<AI6_LAN_IP>:8001
http://<AI6_LAN_IP>:8001/docs
```

The application must bind to `0.0.0.0` inside Open Terminal for Docker port publishing to make it reachable from the LAN. The coding-agent system prompt should detect the current LAN address dynamically instead of storing a fixed IP.
