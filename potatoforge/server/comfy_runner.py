"""Sequential ComfyUI execution for server-owned local jobs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import websockets

from .comfy_workflow import apply_bindings, load_json, validate_bindings
from .jobs import Job, LocalJobManager

logger = logging.getLogger(__name__)


def activation_workflow(workflow: dict[str, Any], *, first_action: str,
                        finalize: bool) -> dict[str, Any]:
    """Set the explicit calibration lifecycle for one submitted workflow."""
    result = copy.deepcopy(workflow)
    for node in result.values():
        if node["class_type"] == "PotatoForgeActivationCalibration":
            node["inputs"]["session_action"] = first_action
        elif node["class_type"] == "PotatoForgeFinalizeActivationCalibration":
            node["inputs"]["action"] = "finalize" if finalize else "accumulate"
    return result


def comfy_url(value: str) -> str:
    parsed = urlparse(value.strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Comfy URL must be an http(s) server origin, such as http://127.0.0.1:8188.")
    return value.strip().rstrip("/")


class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = comfy_url(base_url)
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=15)

    async def __aenter__(self) -> ComfyClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.http.aclose()

    async def get(self, path: str) -> Any:
        response = await self.http.get(path)
        response.raise_for_status()
        return response.json()

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        response = await self.http.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def submit(self, workflow: dict[str, Any], client_id: str) -> str:
        try:
            result = await self.post("/prompt", {"prompt": workflow, "client_id": client_id})
        except httpx.HTTPStatusError as error:
            raise ValueError(f"Comfy rejected prompt: {error.response.text[:2000]}") from error
        except httpx.RequestError as error:
            raise RuntimeError("Comfy prompt submission did not return a response. It may already be queued; check Comfy history before retrying.") from error
        # A timed-out POST may already have queued a prompt. Never retry it.
        prompt_id = result.get("prompt_id") if isinstance(result, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Comfy did not return a prompt_id: {result}")
        return prompt_id

    async def cancel(self, prompt_id: str) -> str:
        response = await self.http.post(f"/api/jobs/{prompt_id}/cancel")
        if response.status_code in (404, 405):
            queue = await self.get("/queue")
            pending = queue.get("queue_pending", [])
            if any(isinstance(item, list) and len(item) > 1 and item[1] == prompt_id for item in pending):
                # Legacy fallback only removes this queued prompt.
                response = await self.http.post("/queue", json={"delete": [prompt_id]})
                response.raise_for_status()
                return "removed"
            return "wait"
        response.raise_for_status()
        return "targeted"

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = await self.get(f"/history/{prompt_id}")
        return data.get(prompt_id) if isinstance(data, dict) else None

    async def job_status(self, prompt_id: str) -> dict[str, Any] | None:
        response = await self.http.get(f"/api/jobs/{prompt_id}")
        if response.status_code in (404, 405):
            return None
        response.raise_for_status()
        return response.json()

    def ws_url(self, client_id: str) -> str:
        return self.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + f"/ws?clientId={client_id}"


def history_outcome(record: dict[str, Any]) -> tuple[bool, str | None]:
    status = record.get("status", {})
    if not isinstance(status, dict):
        return False, "History has no execution status."
    if status.get("status_str") == "success" and status.get("completed") is True:
        return True, None
    messages = status.get("messages", [])
    for message in messages:
        if isinstance(message, (list, tuple)) and len(message) > 1 and message[0] in {"execution_error", "execution_interrupted"}:
            detail = message[1]
            return False, json.dumps(detail, ensure_ascii=False)[:2000]
    return False, f"History status: {status.get('status_str', 'unknown')}"


class ComfyRunner(LocalJobManager):
    def __init__(self, state_path: Path | None = None) -> None:
        super().__init__()
        self.state_path = state_path
        self._previous: dict[str, Any] | None = None
        if state_path and state_path.is_file():
            try:
                self._previous = json.loads(state_path.read_text(encoding="utf-8"))
                if self._previous.get("status") in {"starting", "running", "stopping"}:
                    self._previous["status"] = "failed"
                    self._previous["error"] = "PotatoForge restarted during this run; check Comfy history before rerunning."
                    self._save(self._previous)
            except (OSError, ValueError, AttributeError):
                logger.warning("Could not read prior Comfy Runner state from %s", state_path)

    def current(self) -> dict[str, Any] | None:
        return super().current() or self._previous

    def _save(self, snapshot: dict[str, Any]) -> None:
        self._previous = snapshot
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError:
            logger.exception("Could not persist Comfy Runner state")

    def set_running(self, job: Job) -> None:
        super().set_running(job)
        snapshot = super().current()
        if snapshot:
            self._save(snapshot)

    def record_progress(self, job: Job, progress: dict[str, Any], line: str) -> None:
        super().record_progress(job, progress, line)
        snapshot = super().current()
        if snapshot:
            self._save(snapshot)

    def finish(self, job: Job, status: str, *, result: dict[str, Any] | None = None,
               error: str | None = None) -> None:
        super().finish(job, status, result=result, error=error)
        snapshot = super().current()
        if snapshot:
            self._save(snapshot)

    def start_run(self, base_url: str, workflow_path: Path, dataset_path: Path,
                  bindings: list[dict[str, str]], final_workflow_path: Path | None = None) -> dict[str, Any]:
        base_url = comfy_url(base_url)
        workflow, workflow_raw = load_json(workflow_path)
        dataset, dataset_raw = load_json(dataset_path)
        validate_bindings(workflow, dataset, bindings)
        calibration_nodes = [node for node in workflow.values()
                             if node["class_type"] == "PotatoForgeActivationCalibration"]
        if calibration_nodes:
            if len(calibration_nodes) != 1 or sum(
                node["class_type"] == "PotatoForgeFinalizeActivationCalibration"
                for node in workflow.values()
            ) != 1:
                raise ValueError("Activation calibration workflows need one start node and one finalizer.")
            identity = calibration_nodes[0]["inputs"].get("model_identity")
            if len(dataset["runs"]) > 1 and (not isinstance(identity, str) or not identity.strip()):
                raise ValueError("Multi-run activation calibration requires model_identity on the calibration node.")
        final_workflow = None
        final_hash = None
        if final_workflow_path is not None:
            final_workflow, final_raw = load_json(final_workflow_path)
            final_hash = hashlib.sha256(final_raw).hexdigest()
            from .comfy_workflow import inspect_workflow
            inspect_workflow(final_workflow)
            final_calibration = [node for node in final_workflow.values()
                                 if node["class_type"] == "PotatoForgeActivationCalibration"]
            if final_calibration:
                if len(final_calibration) != 1 or sum(
                    node["class_type"] == "PotatoForgeFinalizeActivationCalibration"
                    for node in final_workflow.values()
                ) != 1:
                    raise ValueError("The final activation workflow needs one session node and one finalizer.")
                if not calibration_nodes or any(
                    final_calibration[0]["inputs"].get(key) != calibration_nodes[0]["inputs"].get(key)
                    for key in ("session_name", "model_identity", "collection_mode")
                ):
                    raise ValueError("The final activation workflow must name the same session, model, and mode.")
        metadata = {
            "workflow_path": str(workflow_path), "workflow_sha256": hashlib.sha256(workflow_raw).hexdigest(),
            "dataset_path": str(dataset_path), "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
            "bindings": bindings, "base_url": base_url,
            "final_workflow_path": str(final_workflow_path) if final_workflow_path else None,
            "final_workflow_sha256": final_hash,
        }
        snapshot = super().start(lambda job: self._run(job, workflow, dataset, bindings, final_workflow, metadata))
        self._save(snapshot)
        return snapshot

    async def _run(self, job: Job, workflow: dict[str, Any], dataset: dict[str, Any],
                   bindings: list[dict[str, str]], final_workflow: dict[str, Any] | None,
                   metadata: dict[str, Any]) -> None:
        calibration_node = next((node for node in workflow.values()
                                 if node["class_type"] == "PotatoForgeActivationCalibration"), None)
        deferred_activation_final = final_workflow is not None and any(
            node["class_type"] == "PotatoForgeActivationCalibration"
            for node in final_workflow.values()
        )
        rows = [{"index": index, "seed": str(row["seed"]) if "seed" in row else None, "status": "pending", "prompt_id": None,
                 "node_id": None, "error": None, "cached_nodes": [], "executed_nodes": []}
                for index, row in enumerate(dataset["runs"], 1)]
        try:
            self.set_running(job)
            async with ComfyClient(metadata["base_url"]) as client:
                system = await client.get("/system_stats")
                metadata["comfy_version"] = system.get("system", {}).get("comfyui_version") if isinstance(system, dict) else None
                client_id = str(uuid4())
                for index, row in enumerate(dataset["runs"]):
                    if self.is_cancelled(job):
                        break
                    record = rows[index]
                    record["status"] = "submitting"
                    self._progress(job, rows, index)
                    try:
                        async with websockets.connect(client.ws_url(client_id), open_timeout=10) as socket:
                            submitted = apply_bindings(workflow, row, bindings)
                            if calibration_node is not None:
                                submitted = activation_workflow(
                                    submitted,
                                    first_action=(calibration_node["inputs"].get("session_action", "start")
                                                  if index == 0 else "resume"),
                                    finalize=index == len(dataset["runs"]) - 1 and not deferred_activation_final,
                                )
                            prompt_id = await client.submit(submitted, client_id)
                            record["prompt_id"] = prompt_id
                            record["status"] = "queued"
                            self._progress(job, rows, index)
                            await self._wait(client, socket, client_id, job, record, rows, index)
                    except Exception as error:
                        record["status"] = "failed"
                        record["error"] = str(error)
                        raise
                if not self.is_cancelled(job) and final_workflow is not None:
                    record = {"index": "final", "status": "submitting", "prompt_id": None,
                              "node_id": None, "error": None, "cached_nodes": [], "executed_nodes": []}
                    rows.append(record)
                    index = len(rows) - 1
                    try:
                        async with websockets.connect(client.ws_url(client_id), open_timeout=10) as socket:
                            record["prompt_id"] = await client.submit(
                                activation_workflow(final_workflow, first_action="resume", finalize=True)
                                if any(node["class_type"] == "PotatoForgeActivationCalibration" for node in final_workflow.values())
                                else final_workflow,
                                client_id,
                            )
                            record["status"] = "queued"
                            self._progress(job, rows, index)
                            await self._wait(client, socket, client_id, job, record, rows, index)
                    except Exception as error:
                        record["status"] = "failed"
                        record["error"] = str(error)
                        raise
            if self.is_cancelled(job):
                for record in rows:
                    if record["status"] == "pending":
                        record["status"] = "skipped"
                self.finish(job, "cancelled", result={"rows": rows, "metadata": metadata})
            else:
                self.finish(job, "succeeded", result={"rows": rows, "metadata": metadata})
        except Exception as error:
            logger.exception("Comfy runner job %s failed", job.id)
            for record in rows:
                if record["status"] == "pending":
                    record["status"] = "skipped"
            self.finish(job, "failed", result={"rows": rows, "metadata": metadata}, error=str(error))

    def _progress(self, job: Job, rows: list[dict[str, Any]], index: int) -> None:
        record = rows[index]
        self.record_progress(job, {"current": index + 1, "total": len(rows), "rows": [dict(row) for row in rows]},
                             f"Run {record['index']}: {record['status']}")

    async def _wait(self, client: ComfyClient, socket: Any, client_id: str, job: Job, record: dict[str, Any],
                    rows: list[dict[str, Any]], index: int) -> None:
        prompt_id = record["prompt_id"]
        started = time.monotonic()
        cancel_sent = False
        last_check = 0.0
        replacement = None
        try:
            while time.monotonic() - started < 7200:
                if self.is_cancelled(job) and not cancel_sent:
                    cancellation = await client.cancel(prompt_id)
                    cancel_sent = True
                    if cancellation == "removed":
                        record["status"] = "cancelled"
                        return
                    if cancellation == "wait":
                        record["error"] = "This Comfy version cannot safely cancel a running prompt; waiting for it to finish."
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=3)
                    event = json.loads(raw) if isinstance(raw, str) else None
                    if isinstance(event, dict):
                        data = event.get("data") or {}
                        if isinstance(data, dict) and data.get("prompt_id") == prompt_id:
                            kind = event.get("type")
                            if kind == "execution_cached":
                                record["cached_nodes"] = data.get("nodes", [])
                            elif kind == "executing":
                                record["node_id"] = data.get("node")
                                if data.get("node") is not None:
                                    record["status"] = "running"
                                    record["executed_nodes"].append(data["node"])
                            elif kind == "progress":
                                record["progress"] = {"value": data.get("value"), "max": data.get("max")}
                            elif kind == "execution_error":
                                record["error"] = json.dumps(data, ensure_ascii=False)[:2000]
                            self._progress(job, rows, index)
                except asyncio.TimeoutError:
                    pass
                except websockets.exceptions.ConnectionClosed:
                    try:
                        replacement = await websockets.connect(client.ws_url(client_id), open_timeout=5)
                        socket = replacement
                    except (OSError, TimeoutError, websockets.exceptions.WebSocketException):
                        await asyncio.sleep(3)
                if time.monotonic() - last_check < 3:
                    continue
                last_check = time.monotonic()
                history = await client.history(prompt_id)
                if history is not None:
                    successful, error = history_outcome(history)
                    if not successful:
                        record["status"] = "cancelled" if self.is_cancelled(job) else "failed"
                        record["error"] = record["error"] or error
                        self._progress(job, rows, index)
                        if not self.is_cancelled(job):
                            raise RuntimeError(f"Run {record['index']} failed: {record['error']}")
                        return
                    record["status"] = "completed"
                    self._progress(job, rows, index)
                    return
                state = await client.job_status(prompt_id)
                if state and state.get("status") in {"error", "failed", "cancelled"}:
                    record["status"] = "cancelled" if self.is_cancelled(job) else "failed"
                    record["error"] = state.get("error_message") or state.get("status")
                    self._progress(job, rows, index)
                    if not self.is_cancelled(job):
                        raise RuntimeError(f"Run {record['index']} failed: {record['error']}")
                    return
                if state and state.get("status") == "completed":
                    # History may lag the job endpoint briefly; success needs history confirmation.
                    continue
            raise TimeoutError(f"Run {record['index']} did not finish within two hours.")
        finally:
            if replacement is not None:
                await replacement.close()


async def launch_comfy(base_url: str, command: Path, working_directory: Path,
                       timeout_seconds: int = 120) -> dict[str, Any]:
    async with ComfyClient(base_url) as client:
        try:
            await client.get("/system_stats")
            return {"ready": True, "already_running": True, "pid": None}
        except (httpx.HTTPError, ValueError):
            pass
        if command.suffix.lower() not in {".bat", ".cmd"} or not command.is_file() or not working_directory.is_dir() or any(char in str(command) for char in "&|<>^%!"):
            raise ValueError("Launch requires a safe existing .bat/.cmd file and working directory.")
        process = subprocess.Popen(["cmd.exe", "/d", "/c", str(command)], cwd=working_directory,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Comfy launcher exited with code {process.returncode}.")
            try:
                await client.get("/system_stats")
                return {"ready": True, "already_running": False, "pid": process.pid}
            except (httpx.HTTPError, ValueError):
                await asyncio.sleep(2)
        raise TimeoutError(f"Comfy did not become ready within {timeout_seconds} seconds (PID {process.pid}).")
