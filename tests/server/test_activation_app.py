from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from potatoforge.server.app import create_app
from potatoforge.server.model_library import (
    discover_activation_audit_caches,
    discover_activation_calibrations,
    load_activation_folders,
)


class ActivationServerTests(TestCase):
    def _fixture(self, root: Path) -> tuple[TestClient, Path, Path, Path, Path]:
        models = root / 'models'
        activation = root / 'activation'
        profiles = root / 'profiles'
        models.mkdir()
        activation.mkdir()
        profiles.mkdir()
        source = models / 'source.safetensors'
        source.write_bytes(b'fixture')
        calibration = activation / 'calibration.json'
        calibration.write_text(
            json.dumps(
                {
                    'format': 'potatoforge_activation_calibration',
                    'version': 1,
                }
            ),
            encoding='utf-8',
        )
        calibration.with_suffix('.safetensors').write_bytes(b'fixture')
        client = TestClient(create_app(root / 'models.toml'))
        self.assertEqual(
            client.post('/api/model-folders', json={'path': str(models)}).status_code,
            201,
        )
        self.assertEqual(
            client.post('/api/activation-folders', json={'path': str(activation)}).status_code,
            201,
        )
        self.assertEqual(
            client.post('/api/profile-folders', json={'path': str(profiles)}).status_code,
            201,
        )
        return client, source, activation, calibration, profiles

    def test_activation_config_and_discovery_are_pair_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, _, activation, calibration, _ = self._fixture(root)
            nested = activation / 'nested'
            nested.mkdir()
            valid_cache = nested / 'cache.activation-audit.json'
            valid_cache.write_text(
                json.dumps(
                    {
                        'format': 'potatoforge_activation_quant_audit',
                        'format_version': 1,
                    }
                ),
                encoding='utf-8',
            )
            valid_cache.with_suffix('.safetensors').write_bytes(b'fixture')
            (activation / 'generic.json').write_text('{}', encoding='utf-8')
            (activation / 'orphan.json').write_text(
                json.dumps(
                    {
                        'format': 'potatoforge_activation_calibration',
                        'version': 1,
                    }
                ),
                encoding='utf-8',
            )

            self.assertEqual(load_activation_folders(root / 'models.toml'), [activation])
            self.assertEqual(
                discover_activation_calibrations(root / 'models.toml'),
                [
                    {
                        'folder': 'activation',
                        'name': calibration.name,
                        'path': str(calibration),
                        'version': 1,
                        'collection_mode': None,
                    }
                ],
            )
            self.assertEqual(
                discover_activation_audit_caches(root / 'models.toml'),
                [
                    {
                        'folder': 'nested',
                        'name': valid_cache.name,
                        'path': str(valid_cache),
                    }
                ],
            )
            self.assertEqual(
                client.get('/api/activation-calibrations').json()['calibrations'],
                discover_activation_calibrations(root / 'models.toml'),
            )

    def test_activation_discovery_includes_v2_mode_and_rejects_invalid_modes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, _, activation, _, _ = self._fixture(root)
            expected = []
            for name, mode in (('cvar', 'cvar'), ('full', 'full')):
                calibration = activation / f'{name}.json'
                calibration.write_text(
                    json.dumps(
                        {
                            'format': 'potatoforge_activation_calibration',
                            'version': 2,
                            'collection_mode': mode,
                        }
                    ),
                    encoding='utf-8',
                )
                calibration.with_suffix('.safetensors').write_bytes(b'fixture')
                expected.append(
                    {
                        'folder': 'activation',
                        'name': calibration.name,
                        'path': str(calibration),
                        'version': 2,
                        'collection_mode': mode,
                    }
                )

            for name, mode in (('missing', None), ('unknown', 'compact'), ('typed', 2)):
                invalid = activation / f'{name}.json'
                document = {
                    'format': 'potatoforge_activation_calibration',
                    'version': 2,
                }
                if mode is not None:
                    document['collection_mode'] = mode
                invalid.write_text(json.dumps(document), encoding='utf-8')
                invalid.with_suffix('.safetensors').write_bytes(b'fixture')

            expected.append(
                {
                    'folder': 'activation',
                    'name': 'calibration.json',
                    'path': str(activation / 'calibration.json'),
                    'version': 1,
                    'collection_mode': None,
                }
            )
            expected.sort(key=lambda item: item['name'])
            self.assertEqual(
                discover_activation_calibrations(root / 'models.toml'),
                expected,
            )
            self.assertEqual(
                client.get('/api/activation-calibrations').json()['calibrations'],
                expected,
            )

    def test_catalogs_and_activation_folder_preserve_other_lists(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, _, activation, _, _ = self._fixture(root)
            self.assertEqual(client.get('/api/activation-folders').json(), {'folders': [str(activation)]})
            methods = client.get('/api/activation-methods').json()
            self.assertEqual(
                methods['default_methods'],
                ['convrot_w4a4', 'convrot_w4a4_mse', 'int5_convrot', 'int5_convrot_mse', 'int6_convrot', 'int6_convrot_mse', 'int8_convrot', 'int7_convrot', 'int7_convrot_mse'],
            )
            self.assertEqual(
                client.get('/api/activation-metrics').json()['default_metric'],
                'aggregate_observed_relative_sse',
            )
            self.assertEqual(
                client.request(
                    'DELETE',
                    '/api/activation-folders',
                    json={'path': str(activation)},
                ).json(),
                {'folders': []},
            )
            self.assertEqual(client.get('/api/model-folders').status_code, 200)

    def test_activation_audit_job_forwards_request_and_replays_progress(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, source, activation, calibration, _ = self._fixture(root)
            received: dict[str, object] = {}

            def mocked_audit(source_path, calibration_path, output_path, **kwargs):
                received.update(
                    {
                        'source_path': source_path,
                        'calibration_path': calibration_path,
                        'output_path': output_path,
                        'device': kwargs['device'],
                        'methods': kwargs['methods'],
                    }
                )
                kwargs['on_tensor_started'](1, 2, 'blocks.0.weight')
                return SimpleNamespace(
                    source_path=source_path,
                    activation_calibration=calibration_path,
                    cache_metadata_path=activation / 'cache.json',
                    cache_tensors_path=activation / 'cache.safetensors',
                    method_count=2,
                    layer_count=2,
                )

            with TestClient(client.app) as live_client, patch(
                'potatoforge.server.app.activation_audit_model',
                side_effect=mocked_audit,
            ):
                response = live_client.post(
                    '/api/activation/audit/jobs',
                    json={
                        'source_path': str(source),
                        'activation_calibration': str(calibration),
                        'output': str(activation / 'cache'),
                        'device': 'cpu',
                        'methods': ['int8_convrot', 'convrot_w4a4'],
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()['job']['id']
                events = live_client.get(
                    '/api/activation/audit/jobs/' + job_id + '/events'
                )

            self.assertEqual(
                received,
                {
                    'source_path': source,
                    'calibration_path': calibration,
                    'output_path': activation / 'cache',
                    'device': 'cpu',
                    'methods': ['int8_convrot', 'convrot_w4a4'],
                },
            )
            self.assertIn('[1/2] blocks.0.weight', events.text)
            self.assertIn('"cache_metadata_path"', events.text)
            self.assertIn('"status":"succeeded"', events.text)

    def test_activation_audit_stop_cancels_before_first_tensor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, source, activation, calibration, _ = self._fixture(root)
            started = Event()
            release = Event()

            def blocked_audit(*args, **kwargs):
                started.set()
                release.wait(timeout=5)
                kwargs['check_cancelled']()
                raise AssertionError('cancellation check should stop the audit')

            with TestClient(client.app) as live_client, patch(
                'potatoforge.server.app.activation_audit_model',
                side_effect=blocked_audit,
            ):
                response = live_client.post(
                    '/api/activation/audit/jobs',
                    json={
                        'source_path': str(source),
                        'activation_calibration': str(calibration),
                        'output': str(activation / 'cache'),
                        'device': 'cpu',
                        'methods': ['int8_convrot'],
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()['job']['id']
                self.assertTrue(started.wait(timeout=2))

                stopped = live_client.request(
                    'DELETE',
                    f'/api/activation/audit/jobs/{job_id}',
                )
                self.assertEqual(stopped.status_code, 200)
                self.assertEqual(stopped.json()['job']['status'], 'stopping')
                release.set()
                events = live_client.get(
                    '/api/activation/audit/jobs/' + job_id + '/events'
                )

            self.assertIn('"status":"cancelled"', events.text)
            self.assertNotIn('"status":"failed"', events.text)

    def test_activation_audit_rejects_outside_paths_and_bf16(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, source, activation, calibration, _ = self._fixture(root)
            outside = root / 'outside.json'
            with patch('potatoforge.server.app.activation_audit_model') as workflow:
                outside_response = client.post(
                    '/api/activation/audit/jobs',
                    json={
                        'source_path': str(source),
                        'activation_calibration': str(outside),
                        'output': str(activation / 'cache'),
                    },
                )
                bf16_response = client.post(
                    '/api/activation/audit/jobs',
                    json={
                        'source_path': str(source),
                        'activation_calibration': str(calibration),
                        'output': str(activation / 'cache'),
                        'methods': ['bf16'],
                    },
                )

            self.assertEqual(outside_response.status_code, 422)
            self.assertEqual(bf16_response.status_code, 422)
            workflow.assert_not_called()

    def test_activation_review_and_optimize_validate_before_workflow(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, _, activation, calibration, profiles = self._fixture(root)
            outside = root / 'outside.json'
            with patch(
                'potatoforge.server.app.score_activation_audit'
            ) as score, patch(
                'potatoforge.server.app.optimize_activation_profile'
            ) as optimize:
                score_response = client.post(
                    '/api/activation/score',
                    json={
                        'audit_cache': str(outside),
                        'activation_calibration': str(calibration),
                    },
                )
                optimize_response = client.post(
                    '/api/activation/optimize',
                    json={
                        'audit_cache': str(activation / 'cache.json'),
                        'activation_calibration': str(calibration),
                        'output_path': str(profiles / 'profile.json'),
                        'profile_id': 'profile',
                        'target_size_gib': 1,
                        'promotion_budget_mib': 0,
                    },
                )

            self.assertEqual(score_response.status_code, 422)
            self.assertEqual(optimize_response.status_code, 422)
            score.assert_not_called()
            optimize.assert_not_called()

    def test_activation_review_and_optimize_forward_valid_requests(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, _, activation, calibration, profiles = self._fixture(root)
            optimize_result = SimpleNamespace(
                audit_cache=activation / 'cache.json',
                profile_path=profiles / 'profile.json',
                summary_path=profiles / 'profile-summary.json',
                profile_id='profile',
                metric='eval_cvar20_observed_relative_sse',
                target_bytes=1024,
                promotion_budget_bytes=None,
                estimated_output_bytes=900,
                dry_run=False,
            )
            with patch(
                'potatoforge.server.app.score_activation_audit',
                return_value={'summary': {'layer_count': 1}},
            ) as score, patch(
                'potatoforge.server.app.inspect_activation_audit',
                return_value={'summary': {'layer_count': 1}},
            ) as inspect, patch(
                'potatoforge.server.app.optimize_activation_profile',
                return_value=optimize_result,
            ) as optimize:
                score_response = client.post(
                    '/api/activation/score',
                    json={
                        'audit_cache': str(activation / 'cache.json'),
                        'activation_calibration': str(calibration),
                        'metric': 'aggregate_observed_relative_sse',
                    },
                )
                inspect_response = client.post(
                    '/api/activation/inspect',
                    json={
                        'audit_cache': str(activation / 'cache.json'),
                        'activation_calibration': str(calibration),
                        'top_n': 3,
                    },
                )
                optimize_response = client.post(
                    '/api/activation/optimize',
                    json={
                        'audit_cache': str(activation / 'cache.json'),
                        'activation_calibration': str(calibration),
                        'output_path': str(profiles / 'profile'),
                        'profile_id': ' profile ',
                        'target_size_gib': 1,
                        'metric': 'eval_cvar20_observed_relative_sse',
                    },
                )

            self.assertEqual(score_response.status_code, 200)
            self.assertEqual(inspect_response.status_code, 200)
            self.assertEqual(optimize_response.status_code, 200)
            self.assertEqual(
                optimize_response.json()['result']['metric'],
                'eval_cvar20_observed_relative_sse',
            )
            score.assert_called_once()
            inspect.assert_called_once()
            optimize.assert_called_once()
            self.assertEqual(
                optimize.call_args.kwargs,
                {
                    'summary_output': profiles / 'profile-summary.json',
                    'source_path': None,
                    'target_size_gib': 1.0,
                    'promotion_budget_mib': None,
                    'metric': 'eval_cvar20_observed_relative_sse',
                    'methods': None,
                    'baseline_method': 'bf16',
                    'exclude_prefixes': [],
                    'exclude_suffixes': [],
                    'profile_id': 'profile',
                    'overwrite': False,
                    'dry_run': False,
                },
            )


if __name__ == '__main__':
    import unittest

    unittest.main()
