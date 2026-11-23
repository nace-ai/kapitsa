import contextlib
import copy
import io
import json
import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_ab as r


class RunnerTests(unittest.TestCase):
    def response(self):
        return {'done': True, 'done_reason': 'length', 'prompt_eval_count': 4096,
                'eval_count': 256, 'eval_duration': 2550000000,
                'message': {'content': 'answer', 'thinking': 'reason'}}

    def test_decode_excludes_first_prefill_token(self):
        result = r.parse_result(self.response(), 4096)
        self.assertEqual(result['decode_tps'], 100)
        self.assertEqual(result['cached_tokens'], 0)

    def test_thinking_changes_canonical_hash(self):
        a = self.response()
        b = copy.deepcopy(a)
        b['message']['thinking'] = 'different'
        self.assertNotEqual(r.parse_result(a, 4096)['generated_text_sha256'],
                            r.parse_result(b, 4096)['generated_text_sha256'])

    def test_fail_closed_response_fields(self):
        changes = [{'prompt_eval_count': 4095}, {'eval_count': 255}, {'eval_count': 257},
                   {'done_reason': 'stop'}, {'done': False}, {'eval_duration': 0},
                   {'eval_duration': float('nan')}, {'prompt_eval_cached_count': 1},
                   {'message': {'content': '', 'thinking': ''}}, {'error': 'context rejected'}]
        for change in changes:
            with self.subTest(change=change):
                body = self.response()
                body.update(change)
                with self.assertRaises(ValueError):
                    r.parse_result(body, 4096)

    def test_schedule_is_abab_per_actual_length(self):
        self.assertEqual([(a, rep) for n, a, rep in r.SCHEDULE if n == 32768],
                         [('A', 1), ('B', 1), ('A', 2), ('B', 2)])

    def test_exact_prompt_fit_and_explicit_thinking(self):
        class Stub:
            def count(self, body):
                self.last = body
                return 20 + body['messages'][0]['content'].count(' x')
        stub = Stub()
        body = json.loads(r.fit_prompt(stub, 4096))
        self.assertEqual(stub.count(body), 4096)
        self.assertTrue(body['think'])
        self.assertEqual(r.count_body(body)['thinking'], {'type': 'enabled'})

    def test_gate_requires_all_arms_hashes_and_two_percent(self):
        records = [{'prompt_tokens': n, 'arm': arm, 'repeat': rep, 'status': 'ok',
                    'measurement': {'decode_tps': 100 if arm == 'A' else 98,
                                    'generated_text_sha256': str(n)}}
                   for n, arm, rep in r.SCHEDULE]
        self.assertTrue(r.summarize(records)['pass'])
        records[-1]['measurement']['decode_tps'] = 97.9
        self.assertFalse(r.summarize(records)['pass'])
        records[-1]['status'] = 'error'
        self.assertFalse(r.summarize(records)['pass'])

    def test_snapshot_uses_percentage_not_strict_free_ram(self):
        def capture(argv):
            outputs = {'power': "Now drawing from 'AC Power'", 'mode': 'AC Power:\n powermode 2',
                       'percent': 'System-wide memory free percentage: 8%', 'therm': 'No thermal warning level has been recorded'}
            text = {'pmset -g batt': outputs['power'], 'pmset -g custom': outputs['mode'],
                    'pmset -g therm': outputs['therm'], 'memory_pressure -Q': outputs['percent'],
                    'sysctl -n kern.memorystatus_vm_pressure_level': '1',
                    'sysctl -n vm.page_free_count': '100', 'sysctl -n hw.pagesize': '16384',
                    'sysctl vm.swapusage': 'total = 1024.00M used = 512.00M free = 512.00M'}
            return {'returncode': 0, 'stdout': text.get(' '.join(argv), 'diagnostic')}
        with patch.object(r, 'capture', side_effect=capture):
            snap = r.resource_snapshot()
        self.assertEqual(snap['timing_invalid_reasons'], [])
        self.assertLess(snap['free_gib'], 3)

    def test_manifest_never_starts_server(self):
        with patch.object(r.sys, 'argv', ['run_ab.py', '--manifest-only']), \
                patch.object(r.subprocess, 'Popen', side_effect=AssertionError('server forbidden')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(r.main(), 0)
        self.assertEqual(json.loads(output.getvalue())['options']['num_predict'], 256)

    def test_guard_terminates_owned_server_and_aborts(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            server = r.Server(Path('/unused'), Path(folder), 1, 1)
            server.process = Mock()
            server.process.poll.return_value = None
            server.process.returncode = -15
            server.process.pid = 12345
            server.stop_guard.wait = Mock(return_value=False)
            snap = {'timing_invalid_reasons': ['memory pressure level at least 4'], 'swap_used_gib': 0}
            with patch.object(r, 'resource_snapshot', return_value=snap):
                server.monitor()
            server.process.terminate.assert_called()
            self.assertIsNotNone(server.guard_error)
            with self.assertRaises(r.ResourceAbort):
                server.__exit__(None, None, None)

    def test_swap_growth_aborts(self):
        server = r.Server(Path('/unused'), Path('/unused'), 1, 1)
        server.initial_swap = 2
        with self.assertRaises(r.ResourceAbort):
            server.check_snapshot({'timing_invalid_reasons': [], 'swap_used_gib': 3.01})

    def test_port_probe_allows_time_wait(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.create_connection(('127.0.0.1', port)) as client:
                peer, _ = listener.accept()
                with peer:
                    peer.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b'')
        r.check_port_available(port)

    def test_port_probe_rejects_active_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            with self.assertRaises(OSError):
                r.check_port_available(listener.getsockname()[1])

    def test_save_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'record.json'
            r.save(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                r.save(path, {'a': 2})


if __name__ == '__main__':
    unittest.main()
