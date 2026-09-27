"""Small checks for the Comfy Runner's safety contracts."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from potatoforge.server.comfy_runner import ComfyRunner, activation_workflow, history_outcome
from potatoforge.server.comfy_workflow import apply_bindings, inspect_dataset, validate_bindings
from potatoforge.server.app import create_app


class ComfyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "original"}},
            "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1, "linked": ["1", 0]}},
        }
        self.dataset = {"format": "potatoforge_comfy_dataset", "version": 1, "runs": [
            {"prompt": "first", "seed": 4, "resolution": 0.7},
            {"prompt": "second", "seed": 8, "resolution": 0.7},
        ]}
        self.bindings = [
            {"dataset_field": "prompt", "node_id": "1", "class_type": "CLIPTextEncode", "input_name": "text"},
            {"dataset_field": "seed", "node_id": "2", "class_type": "RandomNoise", "input_name": "noise_seed"},
        ]

    def test_fresh_workflow_for_each_run(self):
        validate_bindings(self.workflow, self.dataset, self.bindings)
        first = apply_bindings(self.workflow, self.dataset["runs"][0], self.bindings)
        second = apply_bindings(self.workflow, self.dataset["runs"][1], self.bindings)
        self.assertEqual(first["1"]["inputs"]["text"], "first")
        self.assertEqual(second["1"]["inputs"]["text"], "second")
        self.assertEqual(self.workflow["1"]["inputs"]["text"], "original")
        self.assertEqual(self.workflow["2"]["inputs"]["noise_seed"], 1)

    def test_activation_workflow_actions_are_explicit_and_do_not_mutate_template(self):
        workflow = {
            "start": {"class_type": "PotatoForgeActivationCalibration", "inputs": {"session_action": "start"}},
            "end": {"class_type": "PotatoForgeFinalizeActivationCalibration", "inputs": {"action": "finalize"}},
        }
        first = activation_workflow(workflow, first_action="start", finalize=False)
        middle = activation_workflow(workflow, first_action="resume", finalize=False)
        last = activation_workflow(workflow, first_action="resume", finalize=True)
        self.assertEqual([item["start"]["inputs"]["session_action"] for item in (first, middle, last)],
                         ["start", "resume", "resume"])
        self.assertEqual([item["end"]["inputs"]["action"] for item in (first, middle, last)],
                         ["accumulate", "accumulate", "finalize"])
        self.assertEqual(workflow["end"]["inputs"]["action"], "finalize")

    def test_invalid_values_and_stale_binding_stop_before_submission(self):
        self.dataset["runs"][1]["resolution"] = 1.3
        with self.assertRaisesRegex(ValueError, "Run 2 resolution"):
            inspect_dataset(self.dataset)
        self.dataset["runs"][1]["resolution"] = 0.7
        self.bindings[0]["class_type"] = "changed"
        with self.assertRaisesRegex(ValueError, "Stale binding"):
            validate_bindings(self.workflow, self.dataset, self.bindings)

    def test_full_width_seed_is_preserved(self):
        seed = 18446744073709551615
        self.dataset["runs"][0]["seed"] = seed
        validate_bindings(self.workflow, self.dataset, self.bindings)
        self.assertEqual(apply_bindings(self.workflow, self.dataset["runs"][0], self.bindings)["2"]["inputs"]["noise_seed"], seed)

    def test_history_must_confirm_success(self):
        self.assertEqual(history_outcome({"status": {"status_str": "success", "completed": True}}), (True, None))
        success, error = history_outcome({"status": {"status_str": "error", "completed": True,
            "messages": [["execution_error", {"node_id": "2", "exception_message": "bad"}]]}})
        self.assertFalse(success)
        self.assertIn("node_id", error)

    def test_file_inspection_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow.json"
            dataset = root / "dataset.json"
            workflow.write_text(json.dumps(self.workflow), encoding="utf-8")
            dataset.write_text(json.dumps(self.dataset), encoding="utf-8")
            client = TestClient(create_app(root / "library.toml"))
            self.assertEqual(len(client.get("/api/comfy/json-files", params={"folder": str(root)}).json()["files"]), 2)
            self.assertEqual(client.post("/api/comfy/workflows/inspect", json={"path": str(workflow)}).json()["nodes"][0]["id"], "1")
            self.assertEqual(client.post("/api/comfy/datasets/inspect", json={"path": str(dataset)}).json()["run_count"], 2)

    def test_backend_restart_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({"id": "old", "status": "running"}), encoding="utf-8")
            restored = ComfyRunner(state).current()
            self.assertEqual(restored["status"], "failed")
            self.assertIn("restarted", restored["error"])


class SequentialRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_submits_next_row_only_after_history_success(self):
        submitted = []

        class FakeClient:
            def __init__(self, base_url):
                self.base_url = base_url

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, path):
                return {"system": {}}

            def ws_url(self, client_id):
                return "ws://fake/ws"

            async def submit(self, workflow, client_id):
                submitted.append(
                    (workflow["1"]["inputs"]["text"],
                     workflow["3"]["inputs"]["session_action"],
                     workflow["4"]["inputs"]["action"])
                    if "3" in workflow else
                    workflow["1"]["inputs"]["text"] if "1" in workflow else "final workflow"
                )
                return f"prompt-{len(submitted)}"

            async def history(self, prompt_id):
                return {"status": {"status_str": "success", "completed": True}}

        class FakeSocket:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def recv(self):
                return "{}"

        fixture = ComfyRunnerTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as directory:
            workflow_path = Path(directory) / "workflow.json"
            dataset_path = Path(directory) / "dataset.json"
            workflow_path.write_text(json.dumps(fixture.workflow), encoding="utf-8")
            dataset_path.write_text(json.dumps(fixture.dataset), encoding="utf-8")
            manager = ComfyRunner()
            with patch("potatoforge.server.comfy_runner.ComfyClient", FakeClient), \
                 patch("potatoforge.server.comfy_runner.websockets.connect", return_value=FakeSocket()):
                manager.start_run("http://127.0.0.1:8188", workflow_path, dataset_path, fixture.bindings)
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if manager.current()["status"] in {"succeeded", "failed"}:
                        break
                self.assertEqual(manager.current()["status"], "succeeded")
                self.assertEqual(submitted, ["first", "second"])
                submitted.clear()
                fixture.workflow["3"] = {"class_type": "PotatoForgeActivationCalibration",
                                         "inputs": {"session_name": "dataset", "model_identity": "model-a",
                                                    "session_action": "start"}}
                fixture.workflow["4"] = {"class_type": "PotatoForgeFinalizeActivationCalibration",
                                         "inputs": {"action": "finalize"}}
                workflow_path.write_text(json.dumps(fixture.workflow), encoding="utf-8")
                manager = ComfyRunner()
                manager.start_run("http://127.0.0.1:8188", workflow_path, dataset_path, fixture.bindings)
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if manager.current()["status"] in {"succeeded", "failed"}:
                        break
                self.assertEqual(manager.current()["status"], "succeeded")
                self.assertEqual(submitted, [("first", "start", "accumulate"),
                                             ("second", "resume", "finalize")])
                submitted.clear()
                final_path = Path(directory) / "final.json"
                final_path.write_text(json.dumps({"9": {"class_type": "OtherFinal",
                                                         "inputs": {}}}), encoding="utf-8")
                manager = ComfyRunner()
                manager.start_run("http://127.0.0.1:8188", workflow_path, dataset_path,
                                  fixture.bindings, final_workflow_path=final_path)
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if manager.current()["status"] in {"succeeded", "failed"}:
                        break
                self.assertEqual(manager.current()["status"], "succeeded")
                self.assertEqual(submitted, [("first", "start", "accumulate"),
                                             ("second", "resume", "finalize"), "final workflow"])


if __name__ == "__main__":
    unittest.main()
