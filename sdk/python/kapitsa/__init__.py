"""Synchronous Python bindings for Kapitsa's in-process C ABI."""

import ctypes as C
from enum import IntEnum
import inspect
import json
import os
import threading


class Status(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    BUSY = 2
    CANCELLED = 3
    ERROR = 4


class KapitsaError(RuntimeError):
    def __init__(self, status, message, result=None):
        super().__init__(message)
        self.status = status
        self.result = result


_DELTA = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_size_t, C.c_void_p)
_libraries = {}
_library_lock = threading.Lock()
_callbacks = threading.local()
_key = object()


def _json(value):
    if not isinstance(value, dict):
        raise TypeError("options and requests must be dictionaries")
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


class _API:
    def __init__(self, path):
        self.library = C.CDLL(path)  # CDLL releases the GIL during native calls.
        output = C.POINTER(C.c_void_p)
        signatures = {
            "abi_version": (C.c_uint32, []),
            "engine_create": (C.c_int, [C.c_char_p, output, output]),
            "engine_destroy": (None, [C.c_void_p]),
            "request_create": (C.c_int, [C.c_void_p, C.c_char_p, output, output]),
            "generate": (C.c_int, [C.c_void_p, _DELTA, C.c_void_p, output, output]),
            "request_cancel": (None, [C.c_void_p]),
            "request_destroy": (None, [C.c_void_p]),
            "string_free": (None, [C.c_void_p]),
        }
        for name, (result, args) in signatures.items():
            fn = getattr(self.library, "kapitsa_" + name)
            fn.restype, fn.argtypes = result, args
            setattr(self, name, fn)
        if self.abi_version() != 1:
            raise RuntimeError("Unsupported Kapitsa ABI version (expected 1)")

    def result(self, status, error, result=None):
        try:
            message = C.string_at(error).decode("utf-8") if error else None
            response = json.loads(C.string_at(result)) if result else None
            if status != Status.OK:
                raise KapitsaError(status, message or f"Kapitsa status {status}", response)
            return response
        finally:
            self.string_free(error)
            self.string_free(result)


def _api(path):
    path = path or os.environ.get("KAPITSA_LIBRARY")
    if not path:
        raise ValueError("library_path or KAPITSA_LIBRARY must name libkapitsa_sdk.dylib")
    path = os.path.realpath(path)
    with _library_lock:
        if path not in _libraries:
            _libraries[path] = _API(path)
        # Keep loaded libraries alive for handles and native callbacks.
        return _libraries[path]


class Engine:
    """Use load() and close(), preferably with a context manager."""

    def __init__(self, key, api, handle):
        if key is not _key:
            raise TypeError("Use Engine.load()")
        self._api, self._handle = api, handle
        self._lock = threading.Lock()

    @classmethod
    def load(cls, options, *, library_path=None):
        api = _api(library_path)
        handle, error = C.c_void_p(), C.c_void_p()
        transferred = False
        try:
            status = api.engine_create(_json(options), C.byref(handle), C.byref(error))
            error_ptr = error
            error = C.c_void_p()
            api.result(status, error_ptr)
            engine = cls(_key, api, handle)
            transferred = True
            return engine
        finally:
            api.string_free(error)
            if not transferred and handle:
                api.engine_destroy(handle)

    def request(self, options):
        data = _json(options)
        with self._lock:
            if not self._handle:
                raise RuntimeError("Engine is closed")
            handle, error = C.c_void_p(), C.c_void_p()
            transferred = False
            try:
                status = self._api.request_create(self._handle, data, C.byref(handle), C.byref(error))
                error_ptr = error
                error = C.c_void_p()
                self._api.result(status, error_ptr)
                request = Request(_key, self._api, handle)
                transferred = True
                return request
            finally:
                self._api.string_free(error)
                if not transferred and handle:
                    self._api.request_destroy(handle)

    def generate(self, options, *, on_delta=None):
        if getattr(_callbacks, "active", False):
            raise RuntimeError("Do not generate from a delta callback")
        with self.request(options) as request:
            return request.run(on_delta=on_delta)

    def close(self):
        """Release this engine handle; existing requests retain their native engine."""
        with self._lock:
            if self._handle:
                self._api.engine_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        with self._lock:
            if not self._handle:
                raise RuntimeError("Engine is closed")
        return self

    def __exit__(self, *_):
        self.close()


class Request:
    """One-shot request; cancel() may run concurrently with run()."""

    def __init__(self, key, api, handle):
        if key is not _key:
            raise TypeError("Use Engine.request()")
        self._api, self._handle = api, handle
        self._condition = threading.Condition()
        self._started = self._running = self._closing = False

    def run(self, *, on_delta=None):
        if getattr(_callbacks, "active", False):
            raise RuntimeError("Do not run requests from a delta callback")
        if on_delta is not None and (not callable(on_delta) or inspect.iscoroutinefunction(on_delta)):
            raise TypeError("on_delta must be a synchronous callable")
        with self._condition:
            if self._closing or self._started:
                raise RuntimeError("Request is closed or already started")
            self._started = self._running = True
        result, error = C.c_void_p(), C.c_void_p()
        callback_error = []

        def delta(ptr, length, _):
            _callbacks.active = True
            try:
                value = on_delta(json.loads(C.string_at(ptr, length)))
                if inspect.isawaitable(value):
                    if inspect.iscoroutine(value):
                        value.close()
                    raise TypeError("on_delta must not return an awaitable")
                return 0 if value is False else 1
            except BaseException as exc:
                callback_error.append(exc)
                return 0
            finally:
                _callbacks.active = False

        try:
            callback = _DELTA(delta) if on_delta is not None else _DELTA()
            status = self._api.generate(self._handle, callback, None, C.byref(result), C.byref(error))
            if callback_error:
                raise callback_error[0]
            # Transfer both allocated strings to the decoder, even for errors.
            result_ptr, error_ptr = result, error
            result, error = C.c_void_p(), C.c_void_p()
            return self._api.result(status, error_ptr, result_ptr)
        finally:
            self._api.string_free(result)
            self._api.string_free(error)
            with self._condition:
                self._running = False
                self._condition.notify_all()

    def cancel(self):
        with self._condition:
            if self._handle:
                self._api.request_cancel(self._handle)

    def close(self):
        """Cancel, wait for run() to return from C, then release the request."""
        if getattr(_callbacks, "active", False):
            raise RuntimeError("Do not close requests from a delta callback; use cancel()")
        with self._condition:
            self._closing = True
            if self._running:
                self._api.request_cancel(self._handle)
                self._condition.wait_for(lambda: not self._running)
            if self._handle:
                self._api.request_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        with self._condition:
            if self._closing:
                raise RuntimeError("Request is closed")
        return self

    def __exit__(self, *_):
        self.close()
