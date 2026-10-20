# Spec: ArgGYM GRPO training on verl, a second pipeline to compare against TRL

## Why
ArgGYM's RL pipeline lives in `/home/baris/repos/ArgGYM2` (branch
`rl-training`, package `rl/`), built on TRL 1.15's synchronous GRPOTrainer
with vLLM in server mode. A second, independent pipeline on verl lets us
compare the two on the same data, reward and model: if both move a metric the
same way, we trust it; if they disagree, one has a bug. verl also gives async
RL and token-in/token-out rollouts out of the box, which TRL's main trainer
does not.

## Outcome
`/home/baris/repos/arggym-verl` is a git repo built on upstream verl (a local
clone of the latest release tag, with our code added in its own package, or
verl as a pinned dependency with our package beside it; pick the one verl's
docs recommend for custom tasks and say why). It trains with verl's own
features, used as verl intends, not re-implemented:
- GRPO advantages;
- async RL (rollout and training overlapped; verl's fully-async / one-step
  off-policy mode, whichever is its supported path), with off-policy
  correction as verl provides it;
- token-in/token-out rollouts (verl's agent loop / token-level interface), so
  the trainer scores exactly the tokens vLLM sampled; no re-tokenization of
  decoded text.

Two tasks, the same as the TRL pipeline:
1. **GSM8K hard control** (first): Qwen/Qwen3-0.6B (rev
   c1899de289a04d12100db370d81485cdf75e47ca), thinking on; training and dev
   are the screened questions (base model solves 2-6 of 8). Reuse the screen
   output at
   `/home/baris/repos/ArgGYM2/.claude/worktrees/agent-a671d3be36862eff3/outputs/rl/gsm8k-screen/qwen3-0.6b-c1899de/`
   and the selection logic of `rl/gsm8k.py` (`screened`, `load_splits`, dev
   seed 0, dev size 200) so both pipelines see identical rows. Reward: last
   `\boxed{}` vs gold, numeric; truncated completion pays 0 and stays in the
   loss. TRL reference result: dev success 0.579 at step 0 -> 0.61 at step 30
   -> 0.69 at step 90 (`rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml`).
2. **ArgGYM** (second): Qwen3.5-4B, thinking on, the ArgGYM reward and data
   as in `rl/reward.py` / `rl/pool.py` (being updated on branch `rl-003` so a
   truncated completion pays 0; follow that branch once it lands).

Match the TRL configs' decisions wherever verl allows, so the comparison
means something: model revision, LoRA r=32 alpha=32 all-linear (or full FT
only if verl's LoRA path cannot do async/TITO; say so), LR 2e-5 constant
with 5 warmup steps, 16 prompts x 8 generations per step, DAPO-style token-level
loss, clip 0.2/0.28, no KL, no reward std scaling, T=1.0 top_p=1 for training
rollouts, 8192 completion cap (GSM8K), dev eval every 10 steps with the eval
model card's sampling (T 0.6, top_p 0.95, top_k 20, 4 samples, 32,768 cap),
eval at step 0, best checkpoint by dev success. List every decision verl
forces to differ, and why.

Reward and data code: import from ArgGYM2 rather than copy, if that works
cleanly (path dependency); otherwise copy with the source commit noted. The
scorer must be the same code the eval uses.

Logging: trackio if verl supports it, else whatever verl supports locally
(no wandb uploads); log dev success, length, truncation, groups with no
signal, entropy, IS/off-policy metrics.

## Best practices
- Read verl's docs and examples for GRPO, async, agent loop, LoRA before
  writing code; cite the doc/file for each choice in a short
  `workspace/verl-port-2026-10-20/design.md`.
- Its own venv (`uv`), pinned versions; do not touch ArgGYM2's `.venv`.
- Tests on CPU for: data selection identical to ArgGYM2's (same ids), reward
  identical on fixed completions, config arithmetic. A README with the
  launch commands.
- Simplest working setup; no abstraction beyond the two tasks.

## Hardware and constraints
- 3 x A100 80GB. A TRL training run is using all 3 GPUs now and others will
  follow: do **not** use GPUs, start vLLM/Ray on GPUs, kill processes, or
  touch tmux until told GPUs are free. Do everything up to the GPU smoke
  run on CPU, then stop and report what the smoke run needs.
- Local git only: do not create a GitHub fork, push, or open PRs.
- No references to any personal notes vault in committed files.
