from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from safetensors import safe_open
from safetensors.torch import save_file

from potatoforge.server.app import create_app


class ExtractLoraServerTests(TestCase):
    def _fixture(self, root: Path) -> tuple[TestClient, Path, Path, Path, Path]:
        models = root / "models"
        outputs = root / "outputs"
        loras = root / "loras"
        for folder in (models, outputs, loras):
            folder.mkdir()
        torch.manual_seed(0)
        base = {"blk.proj.weight": torch.randn(16, 16), "blk.norm.weight": torch.randn(16)}
        tuned = {
            "blk.proj.weight": base["blk.proj.weight"] + torch.randn(16, 16),
            "blk.norm.weight": base["blk.norm.weight"],
        }
        save_file(base, str(models / "base.safetensors"))
        save_file(tuned, str(models / "tuned.safetensors"))
        client = TestClient(create_app(root / "models.toml"))
        for route, folder in (
            ("model-folders", models),
            ("output-folders", outputs),
            ("lora-folders", loras),
        ):
            self.assertEqual(
                client.post(f"/api/{route}", json={"path": str(folder)}).status_code,
                201,
            )
        return client, models / "base.safetensors", models / "tuned.safetensors", outputs, loras

    def _payload(self, base: Path, tuned: Path, output: Path, **overrides) -> dict:
        payload = {
            "base_path": str(base),
            "tuned_path": str(tuned),
            "output_path": str(output),
            "ranks": [4],
            "dtype": "fp32",
        }
        payload.update(overrides)
        return payload

    def _wait_for_job(self, client: TestClient, job_id: str) -> dict:
        events = client.get(f"/api/lora/extract/jobs/{job_id}/events")
        self.assertEqual(events.status_code, 200)
        self.assertIn("event: complete\n", events.text)
        return client.get("/api/lora/extract/jobs/current").json()["job"]

    def test_job_extracts_one_file_per_rank_and_adds_the_rank_placeholder(self) -> None:
        with TemporaryDirectory() as directory:
            client, base, tuned, outputs, _ = self._fixture(Path(directory))
            with TestClient(client.app) as live:
                response = live.post(
                    "/api/lora/extract/jobs",
                    json=self._payload(base, tuned, outputs / "delta", ranks=[2, 8]),
                )
                self.assertEqual(response.status_code, 202)
                job = self._wait_for_job(live, response.json()["job"]["id"])

            self.assertEqual(job["status"], "succeeded")
            result = job["result"]
            self.assertEqual([o["label"] for o in result["outputs"]], ["2", "8"])
            self.assertTrue((outputs / "delta_r2.safetensors").is_file())
            self.assertTrue((outputs / "delta_r8.safetensors").is_file())
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["skipped_reasons"], {"unchanged": 1})
            by_label = {o["label"]: o for o in result["outputs"]}
            self.assertGreater(
                by_label["8"]["energy_captured"], by_label["2"]["energy_captured"]
            )
            with safe_open(str(outputs / "delta_r2.safetensors"), "pt") as handle:
                self.assertIn("diffusion_model.blk.proj.lora_up.weight", handle.keys())

    def test_output_may_be_written_into_a_configured_lora_folder(self) -> None:
        with TemporaryDirectory() as directory:
            client, base, tuned, _, loras = self._fixture(Path(directory))
            with TestClient(client.app) as live:
                response = live.post(
                    "/api/lora/extract/jobs",
                    json=self._payload(base, tuned, loras / "delta.safetensors"),
                )
                self.assertEqual(response.status_code, 202)
                job = self._wait_for_job(live, response.json()["job"]["id"])

            self.assertEqual(job["status"], "succeeded")
            self.assertTrue((loras / "delta.safetensors").is_file())

    def test_split_fused_and_adaptive_energy_are_forwarded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, base, tuned, outputs, _ = self._fixture(root)
            save_file({"l.qkv.weight": torch.randn(12, 8)}, str(base.parent / "b2.safetensors"))
            save_file({"l.qkv.weight": torch.randn(12, 8)}, str(base.parent / "t2.safetensors"))
            with TestClient(client.app) as live:
                response = live.post(
                    "/api/lora/extract/jobs",
                    json=self._payload(
                        base.parent / "b2.safetensors",
                        base.parent / "t2.safetensors",
                        outputs / "split.safetensors",
                        ranks=[8],
                        energy=0.9,
                        split_fused=["qkv:q,k,v"],
                    ),
                )
                job = self._wait_for_job(live, response.json()["job"]["id"])

            self.assertEqual(job["status"], "succeeded")
            with safe_open(str(outputs / "split.safetensors"), "pt") as handle:
                keys = set(handle.keys())
            self.assertIn("diffusion_model.l.q.lora_up.weight", keys)
            self.assertIn("diffusion_model.l.v.alpha", keys)

    def test_rejects_paths_outside_configured_folders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, base, tuned, outputs, _ = self._fixture(root)
            stray = root / "stray.safetensors"
            save_file({"a.weight": torch.randn(2, 2)}, str(stray))
            outside = root / "elsewhere"
            outside.mkdir()

            cases = {
                "base outside": self._payload(stray, tuned, outputs / "x"),
                "tuned outside": self._payload(base, stray, outputs / "x"),
                "output outside": self._payload(base, tuned, outside / "x"),
                "same checkpoint": self._payload(base, base, outputs / "x"),
            }
            for name, payload in cases.items():
                with self.subTest(name):
                    response = client.post("/api/lora/extract/jobs", json=payload)
                    self.assertEqual(response.status_code, 422, response.text)

    def test_rejects_invalid_options(self) -> None:
        with TemporaryDirectory() as directory:
            client, base, tuned, outputs, _ = self._fixture(Path(directory))
            cases = {
                "energy with two ranks": {"energy": 0.9, "ranks": [2, 4]},
                "duplicate ranks": {"ranks": [4, 4]},
                "bad regex": {"include": "("},
                "bad split rule": {"split_fused": ["qkv"]},
                "zero rank": {"ranks": [0]},
            }
            for name, overrides in cases.items():
                with self.subTest(name):
                    response = client.post(
                        "/api/lora/extract/jobs",
                        json=self._payload(base, tuned, outputs / "x", **overrides),
                    )
                    self.assertEqual(response.status_code, 422, response.text)

    def test_refuses_to_overwrite_existing_output_or_partial(self) -> None:
        with TemporaryDirectory() as directory:
            client, base, tuned, outputs, _ = self._fixture(Path(directory))
            (outputs / "taken.safetensors").write_bytes(b"x")
            (outputs / "half.safetensors.partial").write_bytes(b"x")
            for name in ("taken.safetensors", "half.safetensors"):
                with self.subTest(name):
                    response = client.post(
                        "/api/lora/extract/jobs",
                        json=self._payload(base, tuned, outputs / name),
                    )
                    self.assertEqual(response.status_code, 409, response.text)

    def test_second_job_is_rejected_and_a_job_can_be_stopped(self) -> None:
        with TemporaryDirectory() as directory:
            client, base, tuned, outputs, _ = self._fixture(Path(directory))
            started = Event()
            release = Event()

            def blocked_extract(*args, **kwargs):
                started.set()
                release.wait(timeout=5)
                kwargs["on_tensor_started"](1, 2, "blk.proj.weight")

            with TestClient(client.app) as live, patch(
                "potatoforge.server.app.extract_lora",
                side_effect=blocked_extract,
            ):
                payload = self._payload(base, tuned, outputs / "slow.safetensors")
                response = live.post("/api/lora/extract/jobs", json=payload)
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job"]["id"]
                self.assertTrue(started.wait(timeout=2))

                second = live.post("/api/lora/extract/jobs", json=payload)
                self.assertEqual(second.status_code, 409)

                stopped = live.request("DELETE", f"/api/lora/extract/jobs/{job_id}")
                self.assertEqual(stopped.json()["job"]["status"], "stopping")

                release.set()
                events = live.get(f"/api/lora/extract/jobs/{job_id}/events")

            self.assertIn('"status":"cancelled"', events.text)
            self.assertIn("LoRA extraction stopped.", events.text)

    def test_unknown_job_ids_return_404(self) -> None:
        with TemporaryDirectory() as directory:
            client, *_ = self._fixture(Path(directory))
            self.assertEqual(client.get("/api/lora/extract/jobs/current").json(), {"job": None})
            self.assertEqual(client.get("/api/lora/extract/jobs/nope/events").status_code, 404)
            self.assertEqual(client.request("DELETE", "/api/lora/extract/jobs/nope").status_code, 404)
