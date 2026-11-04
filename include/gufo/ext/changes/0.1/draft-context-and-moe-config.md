# Draft context windows, per-step draft caps and MoE configuration

Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
- `TextRunnerDescriptor::prefill_step_tokens` (default 0, unknown) reports the
  largest prompt step a lone request runs; intermediate prompt checkpoints sit
  on its multiples.
- `TextModelRunner::SetDraftContextHorizon(state, position)` (default no-op)
  names the next position the state is decoded from, snapshotted or shared, so
  a runner whose draft reads a bounded window may omit earlier draft context
  during the next `Prefill`.
- `TextSpeculativeConfig::mtp_prime_chunk` (default 1024) sets the rows per
  batched MTP prompt-prime slice; 0 selects the serial prime.

Speculative decoding (`include/gufo/ext/draft_backend.hpp`,
`include/gufo/ext/speculative_verifier.hpp`):
- `IDraftBackend::TargetContextWindow()` (default 0, every position) and
  `IDraftBackend::SkipTargetContext(position, count, prime)` (default false)
  let a draft advance over target positions without their features.
- `SpeculativeVerifier::Prime` and `ExtendPrompt` take a trailing
  `keep_context_from` (default 0, keep every position).
- `SpeculativeVerifier::StepRequest::max_draft_tokens` and the trailing
  `VerifyStep` argument (default unbounded) cap one step's draft length.
- `SpeculativeOptions::mtp_prime_chunk` (default 1024) mirrors the text option.

Model configuration (`include/gufo/ext/model_config.hpp`):
`ModelConfig::expert_count`, `expert_used_count`, `expert_ff_length`,
`expert_shared_ff_length` and `IsMoE()` describe routed-expert models; zero
`expert_count` is a dense model.

Experimental HIP (`include/gufo/ext/experimental/managed_alloc.hpp`):
`DeviceMemoryRequired()` reports `GUFO_QWEN_WEIGHT_MEMORY=device`, under which
`AllocateDevice` never falls back to managed memory.

Source compatible: every addition has a default. Extensions that construct
these structs with designated initializers, or override the changed virtual
functions, rebuild.
