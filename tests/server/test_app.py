from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from potatoforge.application.commands import inspect_model
from potatoforge.server.app import create_app
from potatoforge.server.audit_jobs import AuditJobManager
from potatoforge.server.quantize_jobs import QuantizeJobManager


class ServerAppTests(TestCase):
    def _write_source_model(self, path: Path, *dtypes: str) -> None:
        offset = 0
        tensors = {}
        for index, dtype in enumerate(dtypes or ("F16",)):
            size = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}[dtype]
            tensors[f"model.{index}.weight"] = {
                "dtype": dtype,
                "shape": [1, 1],
                "data_offsets": [offset, offset + size],
            }
            offset += size
        header = json.dumps(tensors).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * offset)

    def _audit_fixture(self, root: Path):
        models = root / "models"
        outputs = root / "outputs"
        models.mkdir()
        outputs.mkdir()
        source = models / "source.safetensors"
        self._write_source_model(source)
        client = TestClient(create_app(root / "models.toml"))
        client.post("/api/model-folders", json={"path": str(models)})
        client.post("/api/output-folders", json={"path": str(outputs)})
        return client, source, outputs

    def _quantize_fixture(self, root: Path):
        models = root / "models"
        profiles = root / "profiles"
        outputs = root / "outputs"
        models.mkdir()
        profiles.mkdir()
        outputs.mkdir()
        source = models / "source.safetensors"
        self._write_source_model(source)
        profile = profiles / "profile.json"
        profile.write_text("{}", encoding="utf-8")
        client = TestClient(create_app(root / "models.toml"))
        client.post("/api/model-folders", json={"path": str(models)})
        client.post("/api/profile-folders", json={"path": str(profiles)})
        client.post("/api/output-folders", json={"path": str(outputs)})
        return client, source, profile, outputs

    def test_health(self) -> None:
        response = TestClient(create_app()).get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_static_app_serves_assets_and_spa_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            (root / "index.html").write_text("<html>PotatoForge</html>", encoding="utf-8")
            (root / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
            client = TestClient(create_app(static_dir=root))

            self.assertEqual(client.get("/").text, "<html>PotatoForge</html>")
            self.assertEqual(client.get("/inspect").text, "<html>PotatoForge</html>")
            self.assertEqual(client.get("/assets/app.js").text, "console.log('ok')")
            self.assertEqual(client.get("/api/missing").status_code, 404)

    def test_model_folders_are_persisted_and_models_are_discovered(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            self._write_source_model(nested / "model.safetensors")
            config_path = root / "models.toml"
            client = TestClient(create_app(config_path))

            self.assertEqual(client.get("/api/model-folders").json(), {"folders": []})
            self.assertTrue(config_path.exists())
            response = client.post("/api/model-folders", json={"path": str(root)})

            self.assertEqual(response.status_code, 201)
            models = client.get("/api/models").json()["models"]
            self.assertEqual(
                models,
                [{"name": "model.safetensors", "path": str(nested / "model.safetensors"), "quantizable": True}],
            )

            response = client.request("DELETE", "/api/model-folders", json={"path": str(root)})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"folders": []})
            self.assertEqual(client.get("/api/models").json(), {"models": []})

    def test_lora_folders_are_persisted_and_loras_are_discovered(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            lora = nested / "style.safetensors"
            lora.write_bytes(b"fixture")
            client = TestClient(create_app(root / "models.toml"))

            response = client.post("/api/lora-folders", json={"path": str(root)})

            self.assertEqual(response.status_code, 201)
            self.assertEqual(
                client.get("/api/loras").json(),
                {"loras": [{"name": "style.safetensors", "path": str(lora)}]},
            )

            response = client.request("DELETE", "/api/lora-folders", json={"path": str(root)})
            self.assertEqual(response.json(), {"folders": []})

    def test_profile_folders_are_persisted_and_profiles_are_grouped(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            nested = profiles / "int8"
            nested.mkdir(parents=True)
            (profiles / "default.json").write_text("{}", encoding="utf-8")
            (nested / "fast.json").write_text("{}", encoding="utf-8")
            client = TestClient(create_app(root / "models.toml"))

            response = client.post("/api/profile-folders", json={"path": str(profiles)})

            self.assertEqual(response.status_code, 201)
            response = client.post("/api/output-folders", json={"path": str(root)})
            self.assertEqual(response.status_code, 201)
            self.assertEqual(client.get("/api/output-folders").json(), {"folders": [str(root)]})
            self.assertEqual(
                client.get("/api/profiles").json()["profiles"],
                [
                    {
                        "folder": "int8",
                        "name": "fast.json",
                        "path": str(nested / "fast.json"),
                    },
                    {
                        "folder": "profiles",
                        "name": "default.json",
                        "path": str(profiles / "default.json"),
                    },
                ],
            )

    def test_profile_details_returns_validated_rules(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            profiles.mkdir()
            profile = profiles / "mixed.json"
            profile.write_text(
                '{"format_version":1,"profile_id":"mixed","default":"keep",'
                '"rules":[{"action":"int8","prefix":"blocks.",'
                '"suffixes":[".weight"]}]}',
                encoding="utf-8",
            )
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/profile-folders", json={"path": str(profiles)})

            response = client.get("/api/profile-details", params={"path": str(profile)})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "profile_id": "mixed",
                    "default": "keep",
                    "rules": [
                        {
                            "action": "int8",
                            "prefix": "blocks.",
                            "suffixes": [".weight"],
                        }
                    ],
                },
            )

    def test_audit_reports_are_discovered_from_configured_output_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            nested = outputs / "reports" / "weights"
            nested.mkdir(parents=True)
            root_report = outputs / "root.json"
            nested_audit = nested / "a.json"
            nested_other = nested / "z.json"
            root_report.write_text("{}", encoding="utf-8")
            nested_audit.write_text("{}", encoding="utf-8")
            nested_other.write_text("{}", encoding="utf-8")
            (outputs / "ignored.txt").write_text("ignored", encoding="utf-8")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})

            response = client.get("/api/audit-reports")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "reports": [
                        {
                            "folder": "outputs",
                            "name": "root.json",
                            "path": str(root_report),
                        },
                        {
                            "folder": "reports/weights",
                            "name": "a.json",
                            "path": str(nested_audit),
                        },
                        {
                            "folder": "reports/weights",
                            "name": "z.json",
                            "path": str(nested_other),
                        },
                    ],
                },
            )

    def test_audit_methods_use_the_conversion_catalog(self) -> None:
        response = TestClient(create_app()).get("/api/audit-methods")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["methods"],
            [
                {"value": "int8", "label": "INT8"},
                {"value": "int6", "label": "INT6 Rowwise"},
                {"value": "int7", "label": "INT7 Rowwise"},
                {"value": "int8_convrot", "label": "INT8 ConvRot"},
                {"value": "int8_convrot_mse", "label": "INT8 ConvRot MSE"},
                {"value": "int5_convrot", "label": "INT5 ConvRot"},
                {"value": "int5_convrot_mse", "label": "INT5 ConvRot MSE"},
                {"value": "int6_convrot", "label": "INT6 ConvRot"},
                {"value": "int6_convrot_mse", "label": "INT6 ConvRot MSE"},
                {"value": "int7_convrot", "label": "INT7 ConvRot"},
                {"value": "int7_convrot_mse", "label": "INT7 ConvRot MSE"},
                {"value": "convrot_w4a4", "label": "W4A4 ConvRot"},
                {"value": "convrot_w4a4_mse", "label": "W4A4 ConvRot MSE"},
                {"value": "w4a8", "label": "W4A8 ConvRot"},
                {"value": "w6a8", "label": "W6A8 ConvRot"},
                {"value": "w3a8", "label": "W3A8 ConvRot"},
                {"value": "w5a8", "label": "W5A8 ConvRot"},
            ],
        )

    def test_inspect_calls_the_application_workflow(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "source.safetensors"
            report = root / "inspect"
            model.write_bytes(b"fixture")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(root)})
            client.post("/api/output-folders", json={"path": str(root)})
            with patch("potatoforge.server.app.inspect_model") as inspect_model:
                inspect_model.return_value.report = {"summary": {"tensor_count": 1}}
                response = client.post(
                    "/api/inspect",
                    json={"model_path": str(model), "output": str(report)},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["result"]["summary"]["tensor_count"], 1)
            self.assertEqual(inspect_model.call_args.args, (model,))
            self.assertEqual(inspect_model.call_args.kwargs["output"], report.with_suffix(".json"))

    def test_inspect_rejects_report_output_outside_configured_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, source, _ = self._audit_fixture(root)
            with patch("potatoforge.server.app.inspect_model") as inspect_model:
                response = client.post(
                    "/api/inspect",
                    json={
                        "model_path": str(source),
                        "output": str(root / "outside" / "inspect.json"),
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Report output path must be inside a configured output folder.",
            )
            inspect_model.assert_not_called()

    def test_audit_returns_summary_and_paths(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            output = outputs / "weight-audit"
            expected_output = output.with_suffix(".json")
            with patch("potatoforge.server.app.audit_model") as audit_model:
                audit_model.return_value = SimpleNamespace(
                    document={
                        "summary": {
                            "audited_layer_count": 264,
                            "skipped_tensor_count": 2,
                        }
                    }
                )
                response = client.post(
                    "/api/audit",
                    json={
                        "source_path": str(source),
                        "output": str(output),
                        "device": "cuda",
                        "overwrite": True,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "result": {
                        "source_path": str(source),
                        "audit_path": str(expected_output),
                        "audited_layer_count": 264,
                        "skipped_tensor_count": 2,
                        "warning": "Weight reconstruction only; not runtime speed or image quality.",
                    }
                },
            )
            self.assertEqual(audit_model.call_args.args, (source, expected_output))
            self.assertEqual(
                audit_model.call_args.kwargs,
                {
                    "device": "cuda",
                    "overwrite": True,
                    "methods": [
                        "int8_convrot",
                        "int8_convrot_mse",
                        "int5_convrot",
                        "int5_convrot_mse",
                        "int6_convrot",
                        "int6_convrot_mse",
                        "int7_convrot",
                        "int7_convrot_mse",
                        "convrot_w4a4",
                        "convrot_w4a4_mse",
                        "w4a8",
                        "w6a8",
                        "w3a8",
                        "w5a8",
                    ],
                },
            )

    def test_audit_rejects_source_outside_configured_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, _, outputs = self._audit_fixture(root)
            source = root / "outside.safetensors"
            self._write_source_model(source)
            with patch("potatoforge.server.app.audit_model") as audit_model:
                response = client.post(
                    "/api/audit",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Model path must be inside a configured model folder.",
            )
            audit_model.assert_not_called()

    def test_audit_job_starts_and_forwards_request(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            output = outputs / "audit.json"
            started = Event()
            received: dict[str, object] = {}

            def mocked_audit(source_path, output_path, **kwargs):
                received.update(
                    {
                        "source_path": source_path,
                        "output_path": output_path,
                        "device": kwargs["device"],
                        "overwrite": kwargs["overwrite"],
                        "methods": kwargs["methods"],
                    }
                )
                started.set()
                return SimpleNamespace(
                    document={
                        "summary": {
                            "audited_layer_count": 1,
                            "skipped_tensor_count": 0,
                        }
                    }
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.audit_model",
                side_effect=mocked_audit,
            ):
                response = live_client.post(
                    "/api/audit/jobs",
                    json={
                        "source_path": str(source),
                        "output": str(output),
                        "device": "cuda",
                        "overwrite": True,
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))
                events = live_client.get(f"/api/audit/jobs/{job_id}/events")
                current = live_client.get("/api/audit/jobs/current")

            self.assertEqual(
                received,
                {
                    "source_path": source,
                    "output_path": output,
                    "device": "cuda",
                    "overwrite": True,
                    "methods": [
                        "int8_convrot",
                        "int8_convrot_mse",
                        "int5_convrot",
                        "int5_convrot_mse",
                        "int6_convrot",
                        "int6_convrot_mse",
                        "int7_convrot",
                        "int7_convrot_mse",
                        "convrot_w4a4",
                        "convrot_w4a4_mse",
                        "w4a8",
                        "w6a8",
                        "w3a8",
                        "w5a8",
                    ],
                },
            )
            self.assertEqual(current.json()["job"]["id"], job_id)
            self.assertIn('"status":"succeeded"', events.text)

    def test_audit_job_rejects_paths_before_creation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, source, outputs = self._audit_fixture(root)
            outside = root / "outside.safetensors"
            outside.write_bytes(b"fixture")
            with patch("potatoforge.server.app.audit_model") as audit_model:
                response = client.post(
                    "/api/audit/jobs",
                    json={
                        "source_path": str(outside),
                        "output": str(outputs / "audit.json"),
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Model path must be inside a configured model folder.",
            )
            self.assertEqual(client.get("/api/audit/jobs/current").json(), {"job": None})
            audit_model.assert_not_called()

            output_response = client.post(
                "/api/audit/jobs",
                json={
                    "source_path": str(source),
                    "output": str(root / "outside" / "audit.json"),
                },
            )
            self.assertEqual(output_response.status_code, 422)
            self.assertEqual(
                output_response.json()["detail"],
                "Output path must be inside a configured output folder.",
            )
            self.assertEqual(client.get("/api/audit/jobs/current").json(), {"job": None})

    def test_audit_job_rejects_a_second_active_job(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            started = Event()
            release = Event()
            output = outputs / "audit.json"

            def blocked_audit(*args, **kwargs):
                started.set()
                release.wait(timeout=5)
                return SimpleNamespace(
                    document={
                        "summary": {
                            "audited_layer_count": 1,
                            "skipped_tensor_count": 0,
                        }
                    }
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.audit_model",
                side_effect=blocked_audit,
            ):
                payload = {
                    "source_path": str(source),
                    "output": str(output),
                }
                response = live_client.post("/api/audit/jobs", json=payload)
                self.assertEqual(response.status_code, 202)
                self.assertTrue(started.wait(timeout=2))
                second = live_client.post("/api/audit/jobs", json=payload)
                self.assertEqual(second.status_code, 409)

                release.set()
                job_id = response.json()["job"]["id"]
                events = live_client.get(f"/api/audit/jobs/{job_id}/events")

            self.assertIn('"status":"succeeded"', events.text)

    def test_audit_job_can_be_stopped(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            started = Event()
            release = Event()
            callback: dict[str, object] = {}

            def blocked_audit(*args, **kwargs):
                callback["value"] = kwargs["on_entry_started"]
                started.set()
                release.wait(timeout=5)
                callback["value"](1, 2, "blocks.0.weight")

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.audit_model",
                side_effect=blocked_audit,
            ):
                response = live_client.post(
                    "/api/audit/jobs",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                    },
                )
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))

                stopped = live_client.request("DELETE", f"/api/audit/jobs/{job_id}")
                self.assertEqual(stopped.status_code, 200)
                self.assertEqual(stopped.json()["job"]["status"], "stopping")

                release.set()
                events = live_client.get(f"/api/audit/jobs/{job_id}/events")

            self.assertIn('"status":"cancelled"', events.text)
            self.assertIn('"error":"Audit stopped."', events.text)

    def test_audit_job_streams_exact_progress_and_summary(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            started = Event()
            release = Event()
            callback: dict[str, object] = {}

            def mocked_audit(*args, **kwargs):
                callback["value"] = kwargs["on_entry_started"]
                started.set()
                release.wait(timeout=5)
                return SimpleNamespace(
                    document={
                        "summary": {
                            "audited_layer_count": 264,
                            "skipped_tensor_count": 2,
                        }
                    }
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.audit_model",
                side_effect=mocked_audit,
            ):
                response = live_client.post(
                    "/api/audit/jobs",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                    },
                )
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))
                callback["value"](
                    42,
                    264,
                    "model.diffusion_model.blocks.7.attn.to_q.weight",
                )
                release.set()
                events = live_client.get(f"/api/audit/jobs/{job_id}/events")

            self.assertEqual(events.status_code, 200)
            self.assertEqual(events.headers["content-type"], "text/event-stream; charset=utf-8")
            self.assertIn(
                "[42/264] model.diffusion_model.blocks.7.attn.to_q.weight",
                events.text,
            )
            self.assertIn('"current":42,"total":264', events.text)
            self.assertIn('"audited_layer_count":264', events.text)
            self.assertIn('"skipped_tensor_count":2', events.text)
            self.assertLess(events.text.index("event: state"), events.text.index("event: output"))
            self.assertLess(events.text.index("event: output"), events.text.index("event: complete"))
            self.assertIn("event: complete\n", events.text)

    def test_audit_job_replays_output_to_late_subscriber(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            finished = Event()

            def mocked_audit(*args, **kwargs):
                kwargs["on_entry_started"](
                    1,
                    2,
                    "blocks.0.attn.to_q.weight",
                )
                finished.set()
                return SimpleNamespace(
                    document={
                        "summary": {
                            "audited_layer_count": 2,
                            "skipped_tensor_count": 0,
                        }
                    }
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.audit_model",
                side_effect=mocked_audit,
            ):
                response = live_client.post(
                    "/api/audit/jobs",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                    },
                )
                self.assertTrue(finished.wait(timeout=2))
                job_id = response.json()["job"]["id"]
                events = live_client.get(f"/api/audit/jobs/{job_id}/events")

            self.assertIn("[1/2] blocks.0.attn.to_q.weight", events.text)
            self.assertIn('"sequence":1', events.text)
            self.assertIn('"status":"succeeded"', events.text)
            self.assertLess(events.text.index("event: output"), events.text.index("event: complete"))

    def test_audit_job_streams_to_a_live_subscriber_in_order(self) -> None:
        with TemporaryDirectory() as directory:
            started = Event()
            release = Event()
            callback: dict[str, object] = {}

            def mocked_audit(*args, **kwargs):
                callback["value"] = kwargs["on_entry_started"]
                started.set()
                release.wait(timeout=2)
                return SimpleNamespace(
                    document={
                        "summary": {
                            "audited_layer_count": 1,
                            "skipped_tensor_count": 0,
                        }
                    }
                )

            async def assert_stream() -> None:
                manager = AuditJobManager()
                request = SimpleNamespace(
                    source_path=Path(directory) / "source.safetensors",
                    output=Path(directory) / "audit.json",
                    device="cpu",
                    overwrite=False,
                )
                job = manager.start(request, mocked_audit)
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                stream = manager.stream(manager.subscribe(job["id"]))

                self.assertIn('"status":"running"', await anext(stream))
                callback["value"](1, 1, "blocks.0.weight")
                self.assertIn("event: output", await anext(stream))
                release.set()
                self.assertIn('"status":"succeeded"', await anext(stream))
                self.assertIn("event: complete", await anext(stream))

            asyncio.run(assert_stream())

    def test_audit_job_reports_expected_errors_and_unknown_jobs(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.audit_model",
                side_effect=ValueError("CUDA is unavailable"),
            ):
                response = live_client.post(
                    "/api/audit/jobs",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                        "device": "cuda",
                    },
                )
                job_id = response.json()["job"]["id"]
                events = live_client.get(f"/api/audit/jobs/{job_id}/events")
                unknown = live_client.get(
                    "/api/audit/jobs/00000000-0000-0000-0000-000000000000/events"
                )

            self.assertEqual(events.status_code, 200)
            self.assertIn('"status":"failed"', events.text)
            self.assertIn('"error":"CUDA is unavailable"', events.text)
            self.assertIn("event: complete\n", events.text)
            self.assertEqual(unknown.status_code, 404)

    def test_audit_rejects_output_outside_configured_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, source, _ = self._audit_fixture(root)
            with patch("potatoforge.server.app.audit_model") as audit_model:
                response = client.post(
                    "/api/audit",
                    json={
                        "source_path": str(source),
                        "output": str(root / "outside" / "audit.json"),
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Output path must be inside a configured output folder.",
            )
            audit_model.assert_not_called()

    def test_audit_translates_existing_output_error(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            with patch(
                "potatoforge.server.app.audit_model",
                side_effect=FileExistsError("Output already exists"),
            ):
                response = client.post(
                    "/api/audit",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                    },
                )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json(), {"detail": "Output already exists"})

    def test_audit_translates_workflow_value_error(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, outputs = self._audit_fixture(Path(directory))
            with patch(
                "potatoforge.server.app.audit_model",
                side_effect=ValueError("CUDA is unavailable"),
            ):
                response = client.post(
                    "/api/audit",
                    json={
                        "source_path": str(source),
                        "output": str(outputs / "audit.json"),
                        "device": "cuda",
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json(), {"detail": "CUDA is unavailable"})

    def test_optimize_returns_profile_details_and_forwards_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            profiles = root / "profiles"
            outputs.mkdir()
            profiles.mkdir()
            audit = outputs / "audit.json"
            profile = profiles / "optimized"
            expected_profile = profile.with_suffix(".json")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})
            client.post("/api/profile-folders", json={"path": str(profiles)})
            optimized = SimpleNamespace(target_bytes=9126805504, output_bytes=9100000000)
            with patch("potatoforge.server.app.optimize_profile") as optimize_mock:
                optimize_mock.return_value = SimpleNamespace(
                    audit_path=audit,
                    output_path=expected_profile,
                    profile_id="source-8.5gib",
                    optimized=optimized,
                    dry_run=False,
                )
                response = client.post(
                    "/api/optimize",
                    json={
                        "audit_path": str(audit),
                        "output_path": str(profile),
                        "profile_id": "source-8.5gib",
                        "target_size_gib": 8.5,
                        "overwrite": True,
                        "dry_run": False,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "result": {
                        "audit_path": str(audit),
                        "profile_path": str(expected_profile),
                        "profile_id": "source-8.5gib",
                        "target_bytes": 9126805504,
                        "estimated_output_bytes": 9100000000,
                        "dry_run": False,
                        "warning": "Generated profiles still require artifact and runtime validation.",
                    }
                },
            )
            self.assertEqual(optimize_mock.call_args.args, (audit, expected_profile))
            self.assertEqual(
                optimize_mock.call_args.kwargs,
                {
                    "profile_id": "source-8.5gib",
                    "target_size_gib": 8.5,
                    "overwrite": True,
                    "dry_run": False,
                },
            )

    def test_optimize_rejects_audit_outside_configured_output_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            profiles = root / "profiles"
            outputs.mkdir()
            profiles.mkdir()
            outside_audit = root / "outside.json"
            profile = profiles / "optimized.json"
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})
            client.post("/api/profile-folders", json={"path": str(profiles)})
            with patch("potatoforge.server.app.optimize_profile") as optimize_mock:
                response = client.post(
                    "/api/optimize",
                    json={
                        "audit_path": str(outside_audit),
                        "output_path": str(profile),
                        "profile_id": "source-8.5gib",
                        "target_size_gib": 8.5,
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Audit report path must be inside a configured output folder.",
            )
            optimize_mock.assert_not_called()

    def test_optimize_rejects_profile_outside_configured_profile_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            profiles = root / "profiles"
            outputs.mkdir()
            profiles.mkdir()
            audit = outputs / "audit.json"
            outside_profile = root / "outside.json"
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})
            client.post("/api/profile-folders", json={"path": str(profiles)})
            with patch("potatoforge.server.app.optimize_profile") as optimize_mock:
                response = client.post(
                    "/api/optimize",
                    json={
                        "audit_path": str(audit),
                        "output_path": str(outside_profile),
                        "profile_id": "source-8.5gib",
                        "target_size_gib": 8.5,
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Profile path must be inside a configured profile folder.",
            )
            optimize_mock.assert_not_called()

    def test_optimize_translates_existing_profile_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            profiles = root / "profiles"
            outputs.mkdir()
            profiles.mkdir()
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})
            client.post("/api/profile-folders", json={"path": str(profiles)})
            with patch(
                "potatoforge.server.app.optimize_profile",
                side_effect=FileExistsError("Profile already exists"),
            ):
                response = client.post(
                    "/api/optimize",
                    json={
                        "audit_path": str(outputs / "audit.json"),
                        "output_path": str(profiles / "profile.json"),
                        "profile_id": "source-8.5gib",
                        "target_size_gib": 8.5,
                    },
                )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json(), {"detail": "Profile already exists"})

    def test_optimize_preserves_workflow_value_error_detail(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            profiles = root / "profiles"
            outputs.mkdir()
            profiles.mkdir()
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})
            client.post("/api/profile-folders", json={"path": str(profiles)})
            with patch(
                "potatoforge.server.app.optimize_profile",
                side_effect=ValueError("Profile optimization requires a format_version 8 audit. Regenerate the weight audit to include tensor L2 energy."),
            ):
                response = client.post(
                    "/api/optimize",
                    json={
                        "audit_path": str(outputs / "stale.json"),
                        "output_path": str(profiles / "profile.json"),
                        "profile_id": "source-8.5gib",
                        "target_size_gib": 8.5,
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json(),
                {"detail": "Profile optimization requires a format_version 8 audit. Regenerate the weight audit to include tensor L2 energy."},
            )

    def test_optimize_dry_run_returns_no_profile_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            profiles = root / "profiles"
            outputs.mkdir()
            profiles.mkdir()
            audit = outputs / "audit.json"
            profile = profiles / "estimate.json"
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/output-folders", json={"path": str(outputs)})
            client.post("/api/profile-folders", json={"path": str(profiles)})
            with patch("potatoforge.server.app.optimize_profile") as optimize_mock:
                optimize_mock.return_value = SimpleNamespace(
                    audit_path=audit,
                    output_path=profile,
                    profile_id="estimate",
                    optimized=SimpleNamespace(target_bytes=100, output_bytes=90),
                    dry_run=True,
                )
                response = client.post(
                    "/api/optimize",
                    json={
                        "audit_path": str(audit),
                        "output_path": str(profile),
                        "profile_id": "estimate",
                        "target_size_gib": 1.0,
                        "dry_run": True,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["result"],
                {
                    "audit_path": str(audit),
                    "profile_path": None,
                    "profile_id": "estimate",
                    "target_bytes": 100,
                    "estimated_output_bytes": 90,
                    "dry_run": True,
                    "warning": "Generated profiles still require artifact and runtime validation.",
                },
            )
            self.assertEqual(optimize_mock.call_args.kwargs["dry_run"], True)

    def test_inspect_rejects_report_output_same_as_model(self) -> None:
        with TemporaryDirectory() as directory:
            model = Path(directory) / "model.safetensors"
            model.write_bytes(b"fixture")
            writer = Mock()

            with self.assertRaisesRegex(
                ValueError, "Report output path cannot be the source checkpoint"
            ):
                inspect_model(
                    model,
                    output=model,
                    overwrite=True,
                    read_header=lambda _: {"tensor": {"data_offsets": [0, 4]}},
                    write_document=writer,
                )

            writer.assert_not_called()
            self.assertEqual(model.read_bytes(), b"fixture")

    def test_quantize_dry_run_calls_the_application_workflow(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            source = models / "source.safetensors"
            self._write_source_model(source)
            profile = root / "profile.json"
            profile.write_text("{}", encoding="utf-8")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(root)})
            client.post("/api/output-folders", json={"path": str(root)})
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                quantize_model.return_value = SimpleNamespace(
                    estimated_bytes=123456,
                    output_path=None,
                    adapter_count=0,
                )
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "output_path": None,
                        "profile": str(profile),
                        "device": "cuda",
                        "dry_run": True,
                        "timing": True,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {"result": {"estimated_output_bytes": 123456, "dry_run": True}},
            )
            self.assertEqual(quantize_model.call_args.args, (source, None))
            self.assertEqual(
                quantize_model.call_args.kwargs,
                {
                    "profile": profile,
                    "device": "cuda",
                    "adapter_path": [],
                    "adapter_strength": [],
                    "dry_run": True,
                    "timing": True,
                },
            )

    def test_quantize_conversion_returns_output_details(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            source = models / "source.safetensors"
            self._write_source_model(source)
            output = root / "quantized.safetensors"
            output.write_bytes(b"quantized")
            profile = root / "profile.json"
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(root)})
            client.post("/api/output-folders", json={"path": str(root)})
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                quantize_model.return_value = SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=0,
                )
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "output_path": str(output),
                        "profile": str(profile),
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "result": {
                        "output_path": str(output),
                        "output_bytes": len(b"quantized"),
                        "adapter_count": 0,
                    }
                },
            )

    def test_quantize_conversion_includes_timing_report_when_requested(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, profile, outputs = self._quantize_fixture(Path(directory))
            output = outputs / "quantized.safetensors"
            output.write_bytes(b"quantized")
            timing_report = "Quantization timing\n  total: 1.25s"
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                quantize_model.return_value = SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=1,
                    timing=SimpleNamespace(render=Mock(return_value=timing_report)),
                )
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "output_path": str(output),
                        "profile": str(profile),
                        "timing": True,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["result"],
                {
                    "output_path": str(output),
                    "output_bytes": len(b"quantized"),
                    "adapter_count": 1,
                    "timing_report": timing_report,
                },
            )
            self.assertTrue(quantize_model.call_args.kwargs["timing"])

    def test_quantize_forwards_lora_files_and_strengths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            loras = root / "loras"
            models.mkdir()
            loras.mkdir()
            source = models / "source.safetensors"
            adapter = loras / "lora.safetensors"
            self._write_source_model(source)
            adapter.write_bytes(b"fixture")
            output = root / "quantized.safetensors"
            output.write_bytes(b"quantized")
            profile = root / "profile.json"
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/lora-folders", json={"path": str(loras)})
            client.post("/api/profile-folders", json={"path": str(root)})
            client.post("/api/output-folders", json={"path": str(root)})
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                quantize_model.return_value = SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=1,
                )
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "output_path": str(output),
                        "profile": str(profile),
                        "adapter_path": [str(adapter)],
                        "adapter_strength": [0.75],
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                quantize_model.call_args.kwargs,
                {
                    "profile": profile,
                    "device": "cpu",
                    "adapter_path": [adapter],
                    "adapter_strength": [0.75],
                    "dry_run": False,
                    "timing": False,
                },
            )

    def test_quantize_adds_safetensors_extension_when_output_has_none(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            source = models / "source.safetensors"
            self._write_source_model(source)
            output = root / "quantized.safetensors"
            output.write_bytes(b"quantized")
            profile = root / "profile.json"
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(root)})
            client.post("/api/output-folders", json={"path": str(root)})
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                quantize_model.return_value = SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=0,
                )
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "output_path": str(root / "quantized"),
                        "profile": str(profile),
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(quantize_model.call_args.args, (source, output))

    def test_quantize_rejects_source_outside_configured_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            outside = root / "outside.safetensors"
            outside.write_bytes(b"fixture")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(outside),
                        "profile": str(root / "profile.json"),
                        "dry_run": True,
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Model path must be inside a configured model folder.",
            )
            quantize_model.assert_not_called()

    def test_quantize_rejects_profile_outside_configured_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            profiles = root / "profiles"
            models.mkdir()
            profiles.mkdir()
            source = models / "source.safetensors"
            self._write_source_model(source)
            outside_profile = root / "profile.json"
            outside_profile.write_text("{}", encoding="utf-8")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(profiles)})

            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "profile": str(outside_profile),
                        "dry_run": True,
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"],
                "Profile path must be inside a configured profile folder.",
            )
            quantize_model.assert_not_called()

    def test_quantize_translates_existing_output_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            source = models / "source.safetensors"
            self._write_source_model(source)
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(root)})
            client.post("/api/output-folders", json={"path": str(root)})
            with patch(
                "potatoforge.server.app.quantize_model",
                side_effect=FileExistsError("Output already exists"),
            ):
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "output_path": str(root / "output.safetensors"),
                        "profile": str(root / "profile.json"),
                    },
                )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json(), {"detail": "Output already exists"})

    def test_quantize_translates_workflow_value_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            source = models / "source.safetensors"
            self._write_source_model(source)
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(root)})
            with patch(
                "potatoforge.server.app.quantize_model",
                side_effect=ValueError("CUDA is unavailable"),
            ):
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "profile": str(root / "profile.json"),
                        "dry_run": True,
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json(), {"detail": "CUDA is unavailable"})

    def test_quantize_allows_models_with_non_float_tensors(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            source = models / "quantized.safetensors"
            self._write_source_model(source, "F16", "U8")
            profile = root / "profile.json"
            profile.write_text("{}", encoding="utf-8")
            client = TestClient(create_app(root / "models.toml"))
            client.post("/api/model-folders", json={"path": str(models)})
            client.post("/api/profile-folders", json={"path": str(root)})
            with patch("potatoforge.server.app.quantize_model") as quantize_model:
                quantize_model.return_value = SimpleNamespace(
                    estimated_bytes=123456,
                    output_path=None,
                    adapter_count=0,
                )
                response = client.post(
                    "/api/quantize",
                    json={
                        "source_path": str(source),
                        "profile": str(profile),
                        "dry_run": True,
                    },
                )

            self.assertEqual(response.status_code, 200)
            quantize_model.assert_called_once()

    def test_quantize_jobs_return_current_job_and_reject_a_second_start(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, profile, outputs = self._quantize_fixture(Path(directory))
            started = Event()
            release = Event()
            output = outputs / "output.safetensors"
            command_kwargs: dict[str, object] = {}

            def blocked_quantize(*args, **kwargs):
                command_kwargs.update(kwargs)
                started.set()
                release.wait(timeout=5)
                output.write_bytes(b"quantized")
                return SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=0,
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.quantize_model",
                side_effect=blocked_quantize,
            ):
                payload = {
                    "source_path": str(source),
                    "output_path": str(output),
                    "profile": str(profile),
                }
                response = live_client.post("/api/quantize/jobs", json=payload)
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))

                current = live_client.get("/api/quantize/jobs/current")
                self.assertIn(current.json()["job"]["status"], {"starting", "running"})
                second = live_client.post("/api/quantize/jobs", json=payload)
                self.assertEqual(second.status_code, 409)

                release.set()
                events = live_client.get(f"/api/quantize/jobs/{job_id}/events")
                current = live_client.get("/api/quantize/jobs/current").json()["job"]
                self.assertEqual(events.status_code, 200)
                self.assertIn("event: complete\n", events.text)
                self.assertNotIn("timing_report", current["result"])
                self.assertFalse(command_kwargs["timing"])

    def test_quantize_job_can_be_stopped(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, profile, outputs = self._quantize_fixture(Path(directory))
            started = Event()
            release = Event()
            callback: dict[str, object] = {}
            output = outputs / "output.safetensors"

            def blocked_quantize(*args, **kwargs):
                callback["value"] = kwargs["on_entry_started"]
                started.set()
                release.wait(timeout=5)
                callback["value"](
                    1,
                    2,
                    {"action": "int8", "tensor_name": "blocks.0.weight"},
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.quantize_model",
                side_effect=blocked_quantize,
            ):
                response = live_client.post(
                    "/api/quantize/jobs",
                    json={
                        "source_path": str(source),
                        "output_path": str(output),
                        "profile": str(profile),
                        "timing": True,
                    },
                )
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))

                stopped = live_client.request("DELETE", f"/api/quantize/jobs/{job_id}")
                self.assertEqual(stopped.status_code, 200)
                self.assertEqual(stopped.json()["job"]["status"], "stopping")

                release.set()
                events = live_client.get(f"/api/quantize/jobs/{job_id}/events")

            self.assertIn('"status":"cancelled"', events.text)
            self.assertIn('"error":"Quantization stopped."', events.text)
            self.assertNotIn("timing_report", events.text)

    def test_quantize_job_events_replay_progress_and_result(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, profile, outputs = self._quantize_fixture(Path(directory))
            started = Event()
            release = Event()
            callback: dict[str, object] = {}
            command_kwargs: dict[str, object] = {}
            output = outputs / "output.safetensors"
            timing_report = "Quantization timing\n  total: 2.50s"

            def mocked_quantize(*args, **kwargs):
                callback["value"] = kwargs["on_entry_started"]
                command_kwargs.update(kwargs)
                started.set()
                release.wait(timeout=5)
                output.write_bytes(b"quantized")
                return SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=2,
                    timing=SimpleNamespace(render=Mock(return_value=timing_report)),
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.quantize_model",
                side_effect=mocked_quantize,
            ):
                response = live_client.post(
                    "/api/quantize/jobs",
                    json={
                        "source_path": str(source),
                        "output_path": str(output),
                        "profile": str(profile),
                        "timing": True,
                    },
                )
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))
                callback["value"](
                    1,
                    2,
                    {"action": "int8_convrot", "tensor_name": "blocks.7.attn.to_q.weight"},
                )
                release.set()

                events = live_client.get(f"/api/quantize/jobs/{job_id}/events")
                current = live_client.get("/api/quantize/jobs/current").json()["job"]

            self.assertEqual(events.headers["content-type"], "text/event-stream; charset=utf-8")
            self.assertIn(
                "[1/2] int8_convrot: blocks.7.attn.to_q.weight",
                events.text,
            )
            self.assertIn('"current":1,"total":2', events.text)
            self.assertIn('"output_path":"', events.text)
            self.assertIn('"adapter_count":2', events.text)
            self.assertIn(json.dumps(timing_report), events.text)
            self.assertEqual(current["result"]["timing_report"], timing_report)
            self.assertTrue(command_kwargs["timing"])
            self.assertLess(events.text.index("event: state"), events.text.index("event: output"))
            self.assertLess(events.text.index("event: output"), events.text.index("event: complete"))
            self.assertIn('"status":"succeeded"', events.text)

    def test_quantize_job_streams_to_a_live_subscriber_in_order(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output.safetensors"
            started = Event()
            release = Event()
            callback: dict[str, object] = {}

            def mocked_quantize(*args, **kwargs):
                callback["value"] = kwargs["on_entry_started"]
                started.set()
                release.wait(timeout=2)
                output.write_bytes(b"quantized")
                return SimpleNamespace(
                    estimated_bytes=None,
                    output_path=output,
                    adapter_count=0,
                )

            async def assert_stream() -> None:
                manager = QuantizeJobManager()
                request = SimpleNamespace(
                    source_path=Path(directory) / "source.safetensors",
                    output_path=output,
                    profile=Path(directory) / "profile.json",
                    device="cpu",
                    adapter_path=[],
                    adapter_strength=[],
                    dry_run=False,
                    timing=False,
                )
                job = manager.start(request, mocked_quantize)
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                stream = manager.stream(manager.subscribe(job["id"]))

                self.assertIn('"status":"running"', await anext(stream))
                callback["value"](
                    1,
                    2,
                    {"action": "int8", "tensor_name": "blocks.0.weight"},
                )
                self.assertIn("event: output", await anext(stream))
                release.set()
                self.assertIn('"status":"succeeded"', await anext(stream))
                self.assertIn("event: complete", await anext(stream))

            asyncio.run(assert_stream())

    def test_quantize_job_estimate_has_no_output_events(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, profile, _ = self._quantize_fixture(Path(directory))
            started = Event()
            release = Event()
            callback: dict[str, object] = {}

            def mocked_estimate(*args, **kwargs):
                callback["timing"] = kwargs["timing"]
                started.set()
                release.wait(timeout=5)
                return SimpleNamespace(
                    estimated_bytes=123456,
                    output_path=None,
                    adapter_count=0,
                )

            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.quantize_model",
                side_effect=mocked_estimate,
            ):
                response = live_client.post(
                    "/api/quantize/jobs",
                    json={
                        "source_path": str(source),
                        "output_path": None,
                        "profile": str(profile),
                        "dry_run": True,
                        "timing": True,
                    },
                )
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))
                release.set()
                events = live_client.get(f"/api/quantize/jobs/{job_id}/events")
                current = live_client.get("/api/quantize/jobs/current").json()["job"]

            self.assertNotIn("event: output\n", events.text)
            self.assertIn('"estimated_output_bytes":123456', events.text)
            self.assertIn('"dry_run":true', events.text)
            self.assertNotIn("timing_report", events.text)
            self.assertEqual(
                current["result"],
                {"estimated_output_bytes": 123456, "dry_run": True},
            )
            self.assertTrue(callback["timing"])

    def test_quantize_job_validation_happens_before_creation(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, _, outputs = self._quantize_fixture(Path(directory))
            outside_profile = Path(directory) / "outside.json"
            existing_output = outputs / "existing.safetensors"
            existing_output.write_bytes(b"existing")

            invalid = client.post(
                "/api/quantize/jobs",
                json={
                    "source_path": str(source),
                    "output_path": None,
                    "profile": str(outside_profile),
                    "dry_run": True,
                },
            )
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(client.get("/api/quantize/jobs/current").json(), {"job": None})

            existing = client.post(
                "/api/quantize/jobs",
                json={
                    "source_path": str(source),
                    "output_path": str(existing_output),
                    "profile": str(Path(directory) / "profiles" / "profile.json"),
                },
            )
            self.assertEqual(existing.status_code, 409)

            configured_models = client.post(
                "/api/output-folders",
                json={"path": str(source.parent)},
            )
            self.assertEqual(configured_models.status_code, 201)
            same_path = client.post(
                "/api/quantize/jobs",
                json={
                    "source_path": str(source),
                    "output_path": str(source),
                    "profile": str(Path(directory) / "profiles" / "profile.json"),
                },
            )
            self.assertEqual(same_path.status_code, 422)
            self.assertEqual(
                same_path.json()["detail"],
                "Source and Output path cannot be the same",
            )
            self.assertEqual(client.get("/api/quantize/jobs/current").json(), {"job": None})

            stale_output = outputs / "stale.safetensors"
            stale_output.with_name(f"{stale_output.name}.partial").write_bytes(b"partial")
            stale_partial = client.post(
                "/api/quantize/jobs",
                json={
                    "source_path": str(source),
                    "output_path": str(stale_output),
                    "profile": str(Path(directory) / "profiles" / "profile.json"),
                },
            )
            self.assertEqual(stale_partial.status_code, 409)
            self.assertIn("Partial output already exists", stale_partial.json()["detail"])
            self.assertEqual(client.get("/api/quantize/jobs/current").json(), {"job": None})

    def test_quantize_job_value_error_and_unknown_job(self) -> None:
        with TemporaryDirectory() as directory:
            client, source, profile, outputs = self._quantize_fixture(Path(directory))
            output = outputs / "output.safetensors"
            with TestClient(client.app) as live_client, patch(
                "potatoforge.server.app.quantize_model",
                side_effect=ValueError("CUDA is unavailable"),
            ):
                response = live_client.post(
                    "/api/quantize/jobs",
                    json={
                        "source_path": str(source),
                        "output_path": str(output),
                        "profile": str(profile),
                        "timing": True,
                    },
                )
                job_id = response.json()["job"]["id"]
                events = live_client.get(f"/api/quantize/jobs/{job_id}/events")
                unknown = live_client.get(
                    "/api/quantize/jobs/00000000-0000-0000-0000-000000000000/events"
                )

            self.assertIn('"status":"failed"', events.text)
            self.assertIn("CUDA is unavailable", events.text)
            self.assertNotIn("timing_report", events.text)
            self.assertEqual(unknown.status_code, 404)
