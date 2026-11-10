"""Real-library CPU contracts, plus an explicitly requested model smoke test."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from kapitsa import Engine, KapitsaError, Status


def expect_error(status, call):
    try:
        call()
    except KapitsaError as error:
        assert error.status == status, (error.status, status)
        return error
    raise AssertionError(f"Expected {status.name}")


def cpu(library):
    # Invalid options and unsupported metadata are rejected before GPU initialization.
    for options in ({}, {"model_dir": 42}, {"typo": True}):
        expect_error(Status.INVALID_ARGUMENT, lambda: Engine.load(options, library_path=library))
    with tempfile.TemporaryDirectory(prefix="kapitsa-python-native-") as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        lock = Path(tmp) / "gpu.lock"
        with patch.dict(os.environ, {"BENCH_GPU_LOCK": str(lock), "NI_LOCK_HELD": "0",
                                     "NACE_GPU_LOCK_HELD": "0"}):
            with lock.open("w") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                expect_error(Status.BUSY, lambda: Engine.load({"model_dir": str(model)}, library_path=library))
            expect_error(Status.ERROR, lambda: Engine.load({"model_dir": str(model)}, library_path=library))
            with lock.open("w") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("PASS real-library CPU options, contention and failed-load cleanup")


def model_smoke(library, model):
    # Engine.load takes the real shared GPU lock, held through the last retained request.
    options = {"model_dir": model, "context": 2048, "slots": 1, "spec_depth": 0,
               "dflash": "none", "disk_cache": "0"}
    request = {"messages": [{"role": "user", "content": "Reply with the word hello."}],
               "max_tokens": 32, "temperature": 0, "seed": 7,
               "template_kwargs": {"enable_thinking": False}}
    with Engine.load(options, library_path=library) as engine:
        deltas = []
        result = engine.generate(request, on_delta=deltas.append)
        assert deltas and (result["content"] or result["reasoning"] or result["tool_calls"])
        assert result["completion_tokens"] > 0
        assert result["finish_reason"] in ("stop", "length", "tool_calls")
        expect_error(Status.INVALID_ARGUMENT, lambda: engine.request({"messages": []}))
        with engine.request(request) as pending:
            pending.cancel()
            assert expect_error(Status.CANCELLED, pending.run).result is None
        cancelled = expect_error(Status.CANCELLED, lambda: engine.generate(request, on_delta=lambda _: False))
        assert cancelled.result is not None
        with engine.request(request) as retained:
            engine.close()
            assert retained.run()["completion_tokens"] > 0
    # All retained requests have closed, so the native singleton and GPU lock can be reacquired.
    with Engine.load(options, library_path=library) as reloaded:
        assert reloaded.generate(request)["completion_tokens"] > 0
    print(json.dumps({"status": "PASS", "test": "real-model Python smoke", "deltas": len(deltas),
                      "completion_tokens": result["completion_tokens"], "finish_reason": result["finish_reason"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", help="Explicitly allow loading this existing model and using the GPU")
    args = parser.parse_args()
    cpu(args.library)
    if args.model:
        model_smoke(args.library, args.model)
