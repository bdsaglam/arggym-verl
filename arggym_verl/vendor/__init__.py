"""ArgGYM's training glue, vendored so this repository runs without an ArgGYM checkout.

Source: https://github.com/ieddeveci/ArgGYM at commit
a5b05fbe314e5b1eda9f756c8e3841a6ebc8598c (MIT License, `LICENSE` in this directory).
ArgGYM's own library, `arggym` (the task generators and the grader), is a
dependency instead (scripts/arggym/requirements.txt); its code is identical at
that commit and at the pinned b04e12f.

| Module | Source | Kept |
|---|---|---|
| `rl_config` | `rl/config.py` | the YAML loader and `key=value` overrides |
| `rl_gsm8k` | `rl/gsm8k.py` | GSM8K rows, screen filter, prompt, `\\boxed{}` check |
| `rl_pool` | `rl/pool.py` | building and loading the ArgGYM pool, `restrict`, prompts |
| `rl_reward` | `rl/reward.py` | `score` and `reward_of` |
| `rl_curriculum` | `rl/curriculum.py` | the pass-rate curriculum |
| `evals_prompt` | `evals/prompt.py` | templates, elicitation, `compose`, `region` |
| `evals_score` | `evals/score.py` | `answer_of`, `score_one` |
| `evals_values` | `evals/values.py` | `decode` |
| `evals_taskset` | `evals/taskset.py` | `load`, `check` |
| `conf/elicitation-cot.yaml` | `evals/conf/elicitation/cot.yaml` | as is |
| `conf/taskset.yaml`, `conf/taskset-lite.yaml` | `data/taskset.yaml`, `data/taskset-lite.yaml` | as is: the eval specs' seed windows |
| `conf/eval_keys.json.gz` | built from `data/taskset.jsonl`, `data/taskset-lite.jsonl` | the question and theory keys (`rl.pool.keys`) of every eval row |

Each module's header lists what changed besides deletions. Functions kept are
unchanged unless the header says otherwise; the parity tests
(`arggym_verl/tests/test_vendor.py`) compare their outputs with fixtures written
by ArgGYM's own code at the source commit.
"""

SOURCE_REPO = "https://github.com/ieddeveci/ArgGYM"
SOURCE_COMMIT = "a5b05fbe314e5b1eda9f756c8e3841a6ebc8598c"
