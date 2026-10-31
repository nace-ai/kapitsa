// Opt-in: run under the shared GPU lock; the SDK itself exercises a temporary lock.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Engine, Status } from '../index.js';

test('real model: streaming, reuse, concurrency, cancellation, close and reload', {
  skip: !process.env.KAPITSA_MODEL, timeout: 600_000,
}, async () => {
  assert.equal(process.env.NI_LOCK_HELD, '1', 'Run under the shared GPU lock with NI_LOCK_HELD=1');
  const temporary = mkdtempSync(join(tmpdir(), 'kapitsa-node-model-'));
  process.env.BENCH_GPU_LOCK = join(temporary, 'gpu.lock');
  delete process.env.NI_LOCK_HELD;
  delete process.env.NACE_GPU_LOCK_HELD;
  const options = { model_dir: process.env.KAPITSA_MODEL, context: 2048, slots: 2,
    spec_depth: 0, dflash: 'none', disk_cache: '0' };
  const request = { messages: [{ role: 'user', content: 'Count from one to ten.' }],
    max_tokens: 32, temperature: 0, template_kwargs: { enable_thinking: false } };
  let engine;
  try {
    engine = await Engine.load(options);
    let deltas = 0, tick = false;
    const timer = setTimeout(() => { tick = true; }, 0);
    const result = await engine.generate(request, { onDelta: () => { ++deltas; } });
    clearTimeout(timer);
    assert.ok(tick && deltas > 0 && result.completion_tokens > 0);
    assert.equal((await engine.generate(request)).content, result.content);
    for (const response of await Promise.all([engine.generate(request), engine.generate(request)]))
      assert.ok(response.completion_tokens > 0);
    const controller = new AbortController();
    await assert.rejects(engine.generate(request, { signal: controller.signal,
      onDelta: () => { controller.abort(); },
    }), error => error.status === Status.CANCELLED);
    const thrown = new Error('consumer error');
    await assert.rejects(engine.generate(request, { onDelta: () => { throw thrown; } }), error => error === thrown);
    const pending = engine.generate(request);
    const rejection = assert.rejects(pending, error => error.status === Status.CANCELLED);
    await engine.close();
    await rejection;
    engine = await Engine.load(options);
    assert.ok((await engine.generate(request)).completion_tokens > 0);
  } finally {
    await engine?.close();
    rmSync(temporary, { recursive: true, force: true });
  }
});
