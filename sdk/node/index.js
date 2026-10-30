import koffi from 'koffi';
import { realpathSync } from 'node:fs';

export const Status = Object.freeze({ OK: 0, INVALID_ARGUMENT: 1, BUSY: 2, CANCELLED: 3, ERROR: 4 });

export class KapitsaError extends Error {
  constructor(status, message, result = null) {
    super(message);
    this.name = 'KapitsaError';
    this.status = status;
    this.result = result;
  }
}

const libraries = new Map();
const callbackType = koffi.pointer(koffi.proto('int', ['void *', 'size_t', 'void *']));
const output = koffi.out(koffi.pointer('void *'));
const token = Symbol('Engine');
const asyncCall = (fn, ...args) => new Promise((resolve, reject) => {
  fn.async(...args, (error, value) => error ? reject(error) : resolve(value));
});

function bindings(path) {
  if (!path) throw new TypeError('libraryPath or KAPITSA_LIBRARY must name libkapitsa_sdk.dylib');
  path = realpathSync(path);
  if (libraries.has(path)) return libraries.get(path);
  const lib = koffi.load(path);
  if (lib.func('uint32_t kapitsa_abi_version()')() !== 1)
    throw new Error('Unsupported Kapitsa ABI version (expected 1)');
  const api = {
    create: lib.func('kapitsa_engine_create', 'int', ['str', output, output]),
    destroy: lib.func('void kapitsa_engine_destroy(void *engine)'),
    request: lib.func('kapitsa_request_create', 'int', ['void *', 'str', output, output]),
    generate: lib.func('kapitsa_generate', 'int', ['void *', callbackType, 'void *', output, output]),
    cancel: lib.func('void kapitsa_request_cancel(void *request)'),
    release: lib.func('void kapitsa_request_destroy(void *request)'),
    free: lib.func('void kapitsa_string_free(void *string)'),
  };
  // Keep the library loaded: unloading it while native handles/workers exist is unsafe.
  libraries.set(path, api);
  return api;
}

function json(value) {
  const text = JSON.stringify(value, (_, item) => {
    if (typeof item === 'number' && !Number.isFinite(item))
      throw new TypeError('JSON numbers must be finite');
    return item;
  });
  if (text === undefined) throw new TypeError('Expected a JSON value');
  return text;
}

function check(status, error, result = null) {
  if (status !== Status.OK) {
    const message = error ? koffi.decode(error, 'char', -1) :
      status === Status.CANCELLED ? 'Generation cancelled' : `Kapitsa status ${status}`;
    throw new KapitsaError(status, message, result);
  }
}

export class Engine {
  #api;
  #handle;
  #active = new Set();
  #closing;

  constructor(key, api, handle) {
    if (key !== token) throw new TypeError('Use Engine.load()');
    this.#api = api;
    this.#handle = handle;
  }

  static async load(options, { libraryPath = process.env.KAPITSA_LIBRARY } = {}) {
    const api = bindings(libraryPath);
    const handle = [null], error = [null];
    try {
      const status = await asyncCall(api.create, json(options), handle, error);
      check(status, error[0]);
      return new Engine(token, api, handle[0]);
    } finally {
      api.free(error[0]);
    }
  }

  async generate(request, { onDelta, signal } = {}) {
    if (this.#closing) throw new Error('Engine is closed');
    if (onDelta !== undefined && typeof onDelta !== 'function')
      throw new TypeError('onDelta must be a synchronous function');
    if (signal !== undefined && !(signal instanceof AbortSignal))
      throw new TypeError('signal must be an AbortSignal');
    if (signal?.aborted) throw new KapitsaError(Status.CANCELLED, 'Generation cancelled');
    const api = this.#api, handle = [null], error = [null], result = [null];
    // Creation only parses the request. Keep it on this thread so close cannot race it.
    try { check(api.request(this.#handle, json(request), handle, error), error[0]); }
    finally { api.free(error[0]); error[0] = null; }
    let callback = null, failed = false, callbackError, finish;
    const job = { done: new Promise(resolve => { finish = resolve; }), cancel: () => api.cancel(handle[0]) };
    this.#active.add(job);
    try {
      if (onDelta) callback = koffi.register((ptr, length) => {
        try {
          const value = onDelta(JSON.parse(koffi.decode(ptr, 'char', length)));
          if (value && typeof value.then === 'function') {
            Promise.resolve(value).catch(() => {});
            throw new TypeError('onDelta must be synchronous; do not return a Promise');
          }
          return value === false ? 0 : 1;
        } catch (error) {
          failed = true;
          callbackError = error;
          return 0;
        }
      }, callbackType);
      signal?.addEventListener('abort', job.cancel, { once: true });
      const status = await asyncCall(api.generate, handle[0], callback, null, result, error);
      if (failed) throw callbackError;
      const response = result[0] ? JSON.parse(koffi.decode(result[0], 'char', -1)) : null;
      check(status, error[0], response);
      return response;
    } finally {
      signal?.removeEventListener('abort', job.cancel);
      if (callback !== null) koffi.unregister(callback);
      api.free(result[0]);
      api.free(error[0]);
      // The asynchronous native call has returned: cancel and destroy can no longer race it.
      api.release(handle[0]);
      this.#active.delete(job);
      finish();
    }
  }

  close() {
    if (!this.#closing) {
      // Install the closing flag before cancelling or yielding to user callbacks.
      this.#closing = Promise.resolve().then(async () => {
        const jobs = [...this.#active];
        for (const job of jobs) job.cancel();
        await Promise.all(jobs.map(job => job.done));
        await asyncCall(this.#api.destroy, this.#handle);
        this.#handle = null;
      });
    }
    return this.#closing;
  }
}
