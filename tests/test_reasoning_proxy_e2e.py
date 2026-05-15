import json
import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_proxy(port: int, proc: subprocess.Popen[str], timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            output = ""
            if proc.stdout is not None:
                output = proc.stdout.read()
            raise AssertionError(f"proxy exited before startup\n{output}")

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health/liveliness",
                timeout=1,
            ):
                return
        except Exception:
            time.sleep(0.25)

    output = ""
    if proc.stdout is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        output = proc.stdout.read()
    raise AssertionError(f"proxy did not start within {timeout} seconds\n{output}")


def _collect_stream_text(payload: str) -> str:
    visible_parts = []

    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        event = line[6:]
        if event == "[DONE]":
            continue

        chunk = json.loads(event)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                visible_parts.append(content)

    return "".join(visible_parts)


class _RecordingHandler(BaseHTTPRequestHandler):
    requests = []
    stream_events = [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "<rea"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [
                {"index": 0, "delta": {"content": "soning>private notes"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [
                {"index": 0, "delta": {"content": "</rea"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [
                {"index": 0, "delta": {"content": "soning>final"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [
                {"index": 0, "delta": {"content": " visible answer"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    ]

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw_body)
        type(self).requests.append(payload)

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in type(self).stream_events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        body = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<reasoning>private notes</reasoning>final visible answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def test_proxy_injects_reasoning_and_strips_streaming_response():
    repo_root = Path(__file__).resolve().parents[1]
    litellm_bin = repo_root / ".venv/bin/litellm"
    litellm_cmd = str(litellm_bin) if litellm_bin.exists() else shutil.which("litellm")
    if litellm_cmd is None:
        pytest.skip("litellm CLI is not available for live proxy e2e")

    upstream_port = _free_port()
    proxy_port = _free_port()

    _RecordingHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), _RecordingHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    proc = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_path = tmpdir_path / "config.yaml"
            callback_link = tmpdir_path / "custom_callbacks.py"
            callback_link.symlink_to(repo_root / "custom_callbacks.py")

            config_path.write_text(
                textwrap.dedent(
                    f"""\
                    model_list:
                      - model_name: "*"
                        litellm_params:
                          model: hosted_vllm/MODEL_PLACEHOLDER
                          api_base: http://127.0.0.1:{upstream_port}/v1

                    litellm_settings:
                      callbacks:
                        - custom_callbacks.request_modifier
                        - custom_callbacks.reasoning_stripper
                      drop_params: true
                    """
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            pythonpath = env.get("PYTHONPATH", "")
            extra_paths = [str(tmpdir_path), str(repo_root)]
            env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([pythonpath] if pythonpath else []))

            proc = subprocess.Popen(
                [litellm_cmd, "--config", str(config_path), "--port", str(proxy_port)],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            _wait_for_proxy(proxy_port, proc)

            non_stream_req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "*",
                        "messages": [{"role": "user", "content": "Return the visible answer only."}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(non_stream_req, timeout=15) as resp:
                non_stream_body = json.loads(resp.read().decode("utf-8"))

            stream_req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "*",
                        "messages": [{"role": "user", "content": "Return the visible answer only."}],
                        "stream": True,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(stream_req, timeout=15) as resp:
                body = resp.read().decode("utf-8")

            assert len(_RecordingHandler.requests) >= 2, "mock upstream did not receive both proxy requests"
            non_stream_upstream_payload = _RecordingHandler.requests[-2]
            stream_upstream_payload = _RecordingHandler.requests[-1]
            non_stream_user_content = non_stream_upstream_payload["messages"][-1]["content"]
            stream_user_content = stream_upstream_payload["messages"][-1]["content"]
            streamed_text = _collect_stream_text(body)

            assert isinstance(non_stream_user_content, str)
            assert non_stream_user_content.startswith("<system-reminder>")
            assert "You MUST begin every response" in non_stream_user_content
            assert non_stream_user_content.endswith("Return the visible answer only.")

            assert isinstance(stream_user_content, str)
            assert stream_user_content.startswith("<system-reminder>")
            assert "You MUST begin every response" in stream_user_content
            assert stream_user_content.endswith("Return the visible answer only.")

            assert non_stream_body["choices"][0]["message"]["content"] == "final visible answer"
            assert "<reasoning>" not in streamed_text
            assert streamed_text == "final visible answer"
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()
