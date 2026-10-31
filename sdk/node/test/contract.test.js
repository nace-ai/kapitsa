import assert from 'node:assert/strict';
import { after, test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import koffi from 'koffi';
import { Engine, KapitsaError, Status } from '../index.js';

const root = fileURLToPath(new URL('../../..', import.meta.url));
const temporary = mkdtempSync(join(tmpdir(), 'kapitsa-node-'));
const libraryPath = join(temporary, 'fixture.dylib');
execFileSync('c++', ['-std=c++20', '-dynamiclib', '-pthread', '-I', join(root, 'sdk/c'),
  '-I', join(root, 'third_party'), fileURLToPath(new URL('./fixture.cpp', import.meta.url)), '-o', libraryPath]);
const live = koffi.load(libraryPath).func('int fixture_live()');
after(() => { assert.equal(live(), 0); rmSync(temporary, { recursive: true, force: true }); });
const load = () => Engine.load({ model_dir: 'fixture', slots: 2 }, { libraryPath });
const request = (content = 'normal') => ({ messages: [{ role: 'user', content }] });
const status = value => error => error instanceof KapitsaError && error.status === value;

test('an incompatible ABI is refused before creating an engine', async () => {
  const source = join(temporary, 'bad-abi.cpp'), badLibrary = join(temporary, 'bad-abi.dylib');
  writeFileSync(source, '#include "kapitsa.h"\nextern "C" uint32_t kapitsa_abi_version() { return 2; }\n');
  execFileSync('c++', ['-dynamiclib', '-I', join(root, 'sdk/c'), source, '-o', badLibrary]);
  await assert.rejects(Engine.load({ model_dir: 'fixture' }, { libraryPath: badLibrary }), /ABI version/);
  assert.equal(live(), 0);
});

test('async loading, native failure diagnostics, retry and singleton', async () => {
  await assert.rejects(Engine.load({ model_dir: 'bad' }, { libraryPath }), error => {
    assert.equal(error.message, 'fixture load error'); return status(Status.ERROR)(error);
  });
  let tick = false;
  const timer = setTimeout(() => { tick = true; }, 0);
  const engine = await load();
  clearTimeout(timer);
  assert.equal(tick, true);
  await assert.rejects(load(), status(Status.BUSY));
  await engine.close();
  assert.equal(live(), 0);
});

test('reuse, UTF-8 deltas, tool arguments, metrics and parallel requests', async () => {
  const engine = await load();
  try {
    let text = '', tick = false;
    const timer = setTimeout(() => { tick = true; }, 0);
    const a = engine.generate(request(), { onDelta(delta) {
      assert.equal(delta.reasoning, '想');
      assert.equal(delta.tool_calls[0].arguments, '{"city":"北京"}');
      text += delta.content;
    } });
    const [first, second] = await Promise.all([a, engine.generate(request())]);
    clearTimeout(timer);
    assert.equal(tick, true);
    assert.equal(text, first.content);
    assert.equal(second.content, first.content);
    assert.equal(first.prefill_ms, 1.5);
    assert.equal(first.finish_reason, 'stop');
    assert.equal((await engine.generate(request())).content, text);
  } finally { await engine.close(); }
});

test('callback false, pre-abort and abort while no callback is installed', async () => {
  const engine = await load();
  try {
    let deltas = 0;
    await assert.rejects(engine.generate(request(), { onDelta() { ++deltas; return false; } }), error => {
      assert.equal(error.result.finish_reason, 'cancelled');
      assert.equal(error.result.content, '猫🐈'); return status(Status.CANCELLED)(error);
    });
    assert.equal(deltas, 1);
    await assert.rejects(engine.generate(request(), { signal: AbortSignal.abort() }), status(Status.CANCELLED));
    const controller = new AbortController();
    const generation = engine.generate(request(), { signal: controller.signal });
    const timer = setTimeout(() => controller.abort(), 15);
    await assert.rejects(generation, status(Status.CANCELLED));
    clearTimeout(timer);
    await engine.generate(request());
    controller.abort(); // A completed request must no longer have a native cancel listener.
  } finally { await engine.close(); }
});

test('callback exceptions and invalid requests release native resources', async () => {
  const engine = await load();
  try {
    for (const thrown of [new Error('client callback'), undefined]) {
      await engine.generate(request(), { onDelta() { throw thrown; } }).then(
        () => assert.fail('callback exception lost'), error => assert.equal(error, thrown));
    }
    await assert.rejects(engine.generate(request(), { onDelta: async () => {} }), /synchronous/);
    await assert.rejects(engine.generate({ messages: [] }), status(Status.INVALID_ARGUMENT));
    await assert.rejects(engine.generate({ ...request(), temperature: NaN }), /finite/);
    await engine.generate(request());
  } finally { await engine.close(); }
});

test('close during generation drains requests and rejects new work; close is idempotent', async () => {
  const engine = await load();
  const controller = new AbortController();
  const a = engine.generate(request(), { signal: controller.signal });
  const b = engine.generate(request());
  const settled = Promise.allSettled([a, b]);
  const close = engine.close();
  assert.equal(engine.close(), close);
  await assert.rejects(engine.generate(request()), /closed/);
  await close;
  for (const result of await settled) assert.ok(status(Status.CANCELLED)(result.reason));
  controller.abort();
  assert.equal(live(), 0);
  await (await load()).close();
});

test('close from a delta callback does not destroy an in-flight request', async () => {
  const engine = await load();
  let closing;
  await assert.rejects(engine.generate(request(), { onDelta() { closing = engine.close(); } }),
    status(Status.CANCELLED));
  await closing;
  assert.equal(live(), 0);
});

test('deferred error preserves partial output; failed engine is closed before reloading', async () => {
  const engine = await load();
  try {
    await assert.rejects(engine.generate(request('deferred')), error => {
      assert.equal(error.message, 'deferred pipeline error');
      assert.equal(error.result.content, '猫🐈'); return status(Status.ERROR)(error);
    });
    await assert.rejects(engine.generate(request()), /engine unusable/);
  } finally { await engine.close(); }
  const replacement = await load();
  try { await replacement.generate(request()); } finally { await replacement.close(); }
  assert.equal(live(), 0);
});

test('real C ABI rejects malformed options without loading a model', { skip: !process.env.KAPITSA_LIBRARY }, async () => {
  for (const options of [{}, { model_dir: temporary, invalid: true }])
    await assert.rejects(Engine.load(options), status(Status.INVALID_ARGUMENT));
});
