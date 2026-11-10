"""CPU-only Python contracts against the existing native ABI double; no GPU/model."""

import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from kapitsa import Engine, KapitsaError, Status


ROOT = Path(__file__).resolve().parents[2]
REQUEST = {"messages": [{"role": "user", "content": "hello"}]}


class PythonSDK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="kapitsa-python-test-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.library = str(Path(cls.tmp.name) / "fixture.dylib")
        subprocess.run([
            "c++", "-std=c++20", "-O2", "-shared", "-fPIC",
            "-I" + str(ROOT / "sdk/c"), "-I" + str(ROOT / "third_party"),
            str(ROOT / "sdk/node/test/fixture.cpp"), "-o", cls.library,
        ], check=True)
        cls.native = ctypes.CDLL(cls.library)
        cls.native.fixture_live.restype = ctypes.c_int

    def tearDown(self):
        self.assertEqual(self.native.fixture_live(), 0, "native handles or strings leaked")

    def load(self):
        return Engine.load({"model_dir": "fixture"}, library_path=self.library)

    def test_unicode_stream_and_result(self):
        deltas = []
        with self.load() as engine:
            result = engine.generate(REQUEST, on_delta=deltas.append)
        self.assertEqual(len(deltas), 6)
        self.assertEqual(result["content"], "猫🐈" * 6)
        self.assertEqual("".join(d["content"] for d in deltas), result["content"])
        self.assertEqual(deltas[0]["reasoning"], "想")
        self.assertEqual(result["tool_calls"][0]["arguments"], '{"city":"北京"}')
        self.assertEqual(result["finish_reason"], "stop")

    def test_no_callback_and_reuse(self):
        with self.load() as engine:
            for _ in range(3):
                self.assertEqual(engine.generate(REQUEST)["content"], "猫🐈" * 6)

    def test_error_strings_and_reload(self):
        for _ in range(2):
            with self.assertRaises(KapitsaError) as caught:
                Engine.load({"model_dir": "bad"}, library_path=self.library)
            self.assertEqual(caught.exception.status, Status.ERROR)
            self.assertIn("fixture load error", str(caught.exception))
        with self.load() as engine:
            with self.assertRaises(KapitsaError) as caught:
                engine.request({})
            self.assertEqual(caught.exception.status, Status.INVALID_ARGUMENT)
            with self.assertRaises(KapitsaError) as caught:
                self.load()
            self.assertEqual(caught.exception.status, Status.BUSY)

    def test_precancel_and_one_shot(self):
        with self.load() as engine, engine.request(REQUEST) as request:
            request.cancel()
            with self.assertRaises(KapitsaError) as caught:
                request.run()
            self.assertEqual(caught.exception.status, Status.CANCELLED)
            self.assertIsNone(caught.exception.result)
            with self.assertRaisesRegex(RuntimeError, "already started"):
                request.run()

    def test_callback_cancellation_has_partial_result(self):
        with self.load() as engine:
            with self.assertRaises(KapitsaError) as caught:
                engine.generate(REQUEST, on_delta=lambda _: False)
            self.assertEqual(caught.exception.status, Status.CANCELLED)
            self.assertEqual(caught.exception.result["content"], "猫🐈")

    def test_callback_exception_does_not_cross_c(self):
        marker = KeyboardInterrupt("callback failure")

        def fail(_):
            raise marker

        with self.load() as engine:
            with self.assertRaises(KeyboardInterrupt) as caught:
                engine.generate(REQUEST, on_delta=fail)
            self.assertIs(caught.exception, marker)
            self.assertEqual(engine.generate(REQUEST)["finish_reason"], "stop")

    def test_callback_reentry_and_close_rejected(self):
        with self.load() as engine:
            for operation in ("run", "close"):
                with engine.request(REQUEST) as request:
                    with self.assertRaisesRegex(RuntimeError, "delta callback"):
                        request.run(on_delta=lambda _, op=operation: getattr(request, op)())

    def test_engine_generate_inside_callback_does_not_leak(self):
        with self.load() as engine:
            with self.assertRaisesRegex(RuntimeError, "delta callback"):
                engine.generate(REQUEST, on_delta=lambda _: engine.generate(REQUEST))
            self.assertEqual(engine.generate(REQUEST)["finish_reason"], "stop")

    def test_cancel_inside_callback(self):
        with self.load() as engine, engine.request(REQUEST) as request:
            with self.assertRaises(KapitsaError) as caught:
                request.run(on_delta=lambda _: request.cancel())
            self.assertEqual(caught.exception.status, Status.CANCELLED)

    def test_thread_cancel_and_close_waits_for_callback(self):
        entered, release, closed, close_started = (threading.Event() for _ in range(4))
        outcomes = []
        with self.load() as engine:
            request = engine.request(REQUEST)

            def delta(_):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release callback")

            def generate():
                try:
                    outcomes.append(request.run(on_delta=delta))
                except BaseException as exc:
                    outcomes.append(exc)

            def close():
                close_started.set()
                request.close()
                closed.set()

            runner = threading.Thread(target=generate)
            closer = threading.Thread(target=close)
            runner.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, "already started"):
                    request.run()
                request.cancel()  # Called while C is waiting in a Python callback.
                closer.start()
                self.assertTrue(close_started.wait(5))
                self.assertFalse(closed.wait(0.05))
            finally:
                release.set()
                runner.join(5)
                if closer.ident is not None:
                    closer.join(5)
                request.close()
            self.assertFalse(runner.is_alive())
            self.assertTrue(closed.is_set())
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], KapitsaError)
            self.assertEqual(outcomes[0].status, Status.CANCELLED)
            request.cancel()  # Closed operations are safe and idempotent.
            request.close()
            with self.assertRaises(RuntimeError):
                request.run()

    def test_request_retains_closed_engine(self):
        engine = self.load()
        with engine.request(REQUEST) as request:
            engine.close()
            engine.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                engine.request(REQUEST)
            self.assertEqual(request.run()["finish_reason"], "stop")
        with self.load():
            pass

    def test_deferred_error_preserves_result_and_frees_both_strings(self):
        with self.load() as engine:
            with self.assertRaises(KapitsaError) as caught:
                engine.generate({"messages": [{"role": "user", "content": "deferred"}]})
            self.assertEqual(caught.exception.status, Status.ERROR)
            self.assertEqual(caught.exception.result["content"], "猫🐈")
            self.assertIn("deferred pipeline error", str(caught.exception))
            with self.assertRaisesRegex(KapitsaError, "engine unusable"):
                engine.generate(REQUEST)
        with self.load():
            pass

    def test_python_validation_and_async_callback(self):
        async def delta(_):
            return True

        with self.load() as engine, engine.request(REQUEST) as request:
            for bad in (None, [], "{}"):
                with self.assertRaises(TypeError):
                    engine.request(bad)
            with self.assertRaises(ValueError):
                engine.request({"temperature": float("nan")})
            with self.assertRaises(TypeError):
                request.run(on_delta=delta)
            with self.assertRaisesRegex(TypeError, "awaitable"):
                request.run(on_delta=lambda d: delta(d))

    def test_environment_library_path(self):
        with patch.dict("os.environ", {"KAPITSA_LIBRARY": self.library}):
            with Engine.load({"model_dir": "fixture"}):
                pass
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "library_path"):
                Engine.load({})


if __name__ == "__main__":
    unittest.main(argv=sys.argv)
