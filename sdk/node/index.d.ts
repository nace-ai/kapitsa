export declare const Status: Readonly<{
  OK: 0; INVALID_ARGUMENT: 1; BUSY: 2; CANCELLED: 3; ERROR: 4;
}>;
export type StatusCode = typeof Status[keyof typeof Status];
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export interface EngineOptions {
  model_dir: string;
  context?: number;
  slots?: 1 | 2 | 3 | 4;
  prefill_chunk?: number;
  spec_depth?: number;
  mem_budget_gb?: number;
  dflash?: string;
  disk_cache?: string;
  gpu_lock_timeout_ms?: number;
}
export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | Json[] | null;
  [key: string]: Json | undefined;
}
export interface GenerateRequest {
  messages: Message[];
  tools?: Json[];
  template_kwargs?: { [key: string]: Json };
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
  stop?: string[];
}
export interface ToolCall { id: string; name: string; arguments: string }
export interface Delta { content: string; reasoning: string; tool_calls: ToolCall[] }
export interface GenerateResult extends Delta {
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'cancelled' | 'error';
  error: string;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  prefill_ms: number;
  decode_ms: number;
}
export declare class KapitsaError extends Error {
  readonly status: StatusCode;
  /** Partial output, including on cancellation or a native generation error. */
  readonly result: GenerateResult | null;
  constructor(status: StatusCode, message: string, result?: GenerateResult | null);
}
export interface GenerateOptions {
  /** Runs on the JS thread. Return false to cancel. Exceptions cancel and are rethrown. */
  onDelta?: (delta: Delta) => boolean | void;
  signal?: AbortSignal;
}
export declare class Engine {
  private constructor();
  static load(options: EngineOptions, binding?: { libraryPath?: string }): Promise<Engine>;
  generate(request: GenerateRequest, options?: GenerateOptions): Promise<GenerateResult>;
  /** Cancel active requests, wait for their native calls, then release the engine. Idempotent. */
  close(): Promise<void>;
}
