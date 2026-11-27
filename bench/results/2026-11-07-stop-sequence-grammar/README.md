# Vocabulary grammar checks

Source cf2f25c7f9f38f9babfc51a0def9d55697544b73 on the M3 Ultra described in the preceding evidence folder. The completed invocation `build/grammar_test /Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit` passed308checks, including the vocabulary masks skipped by the earlier tokenizer-free run. The original233-check log is unchanged.

This CPU test overlapped the first two native4Bbenchmark arms. Those timings are retained but do not establish a clean performance gate. A separate full4B A/B/A/B is queued after the current model checks and the completed grammar test. No additional native timing is claimed here.
