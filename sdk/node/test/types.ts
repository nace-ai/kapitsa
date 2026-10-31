import { Engine, KapitsaError, Status, type GenerateRequest, type GenerateResult } from '../index.js';

const request: GenerateRequest = {
  messages: [{ role: 'user', content: 'Hello' }],
  max_tokens: 32, temperature: 0, template_kwargs: { enable_thinking: false },
};
async function usage() {
  const engine = await Engine.load({ model_dir: '/models/4b', slots: 2 }, { libraryPath: '/lib/kapitsa.dylib' });
  try {
    const result: GenerateResult = await engine.generate(request, {
      signal: new AbortController().signal,
      onDelta: delta => { delta.tool_calls.forEach(call => JSON.parse(call.arguments)); },
    });
    result.content.toUpperCase();
  } catch (error) {
    if (error instanceof KapitsaError && error.status === Status.CANCELLED) error.result?.content;
  } finally { await engine.close(); }
  // @ts-expect-error Engine instances must be loaded through the C ABI.
  new Engine();
  // @ts-expect-error Native ABI v1 has no model field on generation requests.
  engine.generate({ ...request, model: '4b' });
  // @ts-expect-error Streaming callbacks must not return promises.
  engine.generate(request, { onDelta: async () => true });
  // @ts-expect-error Kapitsa supports one to four slots.
  Engine.load({ model_dir: '/models/4b', slots: 8 });
}
void usage;
