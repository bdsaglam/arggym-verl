"""Rewrite arggym_verl's ArgGYM-derived files from an ArgGYM checkout.

    PYTHONPATH=<ArgGYM checkout> .venv/bin/python scripts/arggym/write_fixtures.py

Run it when re-vendoring arggym_verl/vendor from a newer ArgGYM commit. It needs
ArgGYM's rl/ and evals/ on the path and 003's pool in data/arggym-pool (built by
scripts/arggym/prepare_data.sh). It writes, with ArgGYM's own code:

- arggym_verl/vendor/conf/eval_keys.json.gz: the eval tasksets' question and theory keys;
- data/arggym-prior/hf-qwen3.5-4b__xml_tags__cot.json: the curriculum's prior;
- arggym_verl/tests/fixtures/*.json: the expected outputs the parity tests compare with.
"""
import gzip
import hashlib
import json
import random
from pathlib import Path

from rl import config as rl_config
from rl import gsm8k, pool
from rl import reward as rl_reward
from rl.curriculum import group_outcomes
from rl.train import build_curriculum

REPO = Path(__file__).resolve().parents[2]
POOL_ROOT = str(REPO / "data/arggym-pool")
OUT = REPO / "arggym_verl/tests/fixtures"
OUT.mkdir(parents=True, exist_ok=True)

q, t = pool.keys(pool.eval_rows())
payload = {"source": "rl.pool.keys(rl.pool.eval_rows()) at ArgGYM a5b05fb: data/taskset.jsonl and data/taskset-lite.jsonl",
           "question_keys": sorted(q), "theory_keys": sorted(t)}
with gzip.GzipFile(REPO / "arggym_verl/vendor/conf/eval_keys.json.gz", "wb", mtime=0) as f:
    f.write(json.dumps(payload, indent=0).encode())
print(len(q), len(t))
m = json.loads((pool.ROOT / "results/hf-qwen3.5-4b__xml_tags__cot/metrics.json").read_text())
prior = {"source": "results/hf-qwen3.5-4b__xml_tags__cot/metrics.json at ArgGYM a5b05fb (by_task_level, the fields rl.curriculum.base_pass_counts reads): Qwen3.5-4B base eval on taskset-lite, xml_tags template, cot elicitation",
         "by_task_level": {k: {"n_scored": v["n_scored"], "success_rate": v["success_rate"]} for k, v in m["by_task_level"].items()}}
p = REPO / "data/arggym-prior/hf-qwen3.5-4b__xml_tags__cot.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(prior, indent=1) + "\n")
print(len(prior["by_task_level"]))

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")

cfg = rl_config.load(str(pool.ROOT / "rl/configs/qwen3.5-4b-003.yaml"), [f"output_root={POOL_ROOT}"])
elicit = pool.elicitation(cfg["prompt"]["elicitation"])
template = cfg["prompt"]["template"]
train = pool.restrict(pool.load_split(cfg, "train"), cfg["train_cells"])
dev = pool.load_split(cfg, "dev")
trl = {"train": pool.trl_rows(train, template, elicit), "dev": pool.trl_rows(dev, template, elicit)}

# 1. Prompts: every row's messages hashed per split, and a few rows in full.
dump("arggym_prompts.json", {
    "source": "rl.pool.trl_rows on experiment 003's pool (rl/configs/qwen3.5-4b-003.yaml), ArgGYM a5b05fb",
    "sha256": {s: sha([[r["id"], r["prompt"]] for r in rows]) for s, rows in trl.items()},
    "row_json_sha256": {s: sha([r["row_json"] for r in rows]) for s, rows in trl.items()},
    "examples": {s: [{"id": r["id"], "prompt": r["prompt"]} for r in rows[:2]] for s, rows in trl.items()},
})

# 2. ArgGYM rewards on fixed completions, one dev row per task.
by_task = {}
for r in dev:
    by_task.setdefault(r["task"], r)
STOP = [151645, 151643]
EOS, OTHER = 151645, 42
cases = []
for task, row in sorted(by_task.items()):
    ref = row["reference_answer"]
    texts = {
        "reference": (f"<think>\nworking\n</think>\n\n<answer>\n{ref}\n</answer>", [OTHER, EOS]),
        "reference_truncated": (f"<think>\nworking\n</think>\n\n<answer>\n{ref}\n</answer>", [OTHER, OTHER]),
        "reference_in_reasoning_only": (f"<think>\n<answer>\n{ref}\n</answer>\n</think>\n\nI am done.", [OTHER, EOS]),
        "reference_without_tags": (f"<think>\nok\n</think>\n\n{ref}", [OTHER, EOS]),
        "wrong": ("<think>\nhmm\n</think>\n\n<answer>no idea</answer>", [OTHER, EOS]),
        "unfinished_thought": ("<think>\nloop loop", [OTHER, OTHER]),
        "empty": ("", []),
    }
    fn = rl_reward.make_reward(STOP, template, gather=lambda x: x)
    for name, (text, ids) in texts.items():
        truncated = bool(ids) and ids[-1] not in STOP
        rec = rl_reward.score(text, row, truncated, template)
        paid = fn([text], [ids], [json.dumps(row)])[0]
        assert paid == rl_reward.reward_of(rec)
        cases.append({"row_id": row["id"], "task": task, "case": name, "text": text, "ids": ids,
                      "reward": paid,
                      "record": {k: rec[k] for k in ("success", "score", "reason", "truncated",
                                                     "no_answer_region", "answer_in_cot",
                                                     "scorer_refused", "zero_with_region")}})
dump("arggym_rewards.json", {
    "source": "rl.reward.make_reward and rl.reward.score (evals.score.score_one), ArgGYM a5b05fb; "
              "rows are the first dev row of each task in experiment 003's pool",
    "template": template, "stop_ids": STOP, "cases": cases})

# 3. GSM8K rewards on fixed completions.
PAD = 151643
G = [("<think>\n12 + 6 = 18\n</think>\n\nThe answer is \\boxed{18}.", 18.0, [OTHER, EOS]),
     ("so \\boxed{17}", 18.0, [OTHER, EOS]),
     ("first \\boxed{3}, then \\boxed{18}", 18.0, [OTHER, EOS]),
     ("\\boxed{\\$1,234}", 1234.0, [OTHER, EOS]),
     ("\\boxed{\\frac{1}{2}}", 0.5, [OTHER, EOS]),
     ("no box at all: 18", 18.0, [OTHER, EOS]),
     ("cut off but \\boxed{18}", 18.0, [OTHER, OTHER]),
     ("ends on padding \\boxed{-3.50}", -3.5, [OTHER, PAD]),
     ("", 1.0, [])]
fn = gsm8k.make_reward(STOP, gather=lambda x: x)
texts, golds, ids = zip(*G)
rewards = fn(list(texts), list(ids), list(golds), id=[f"q{i}" for i in range(len(G))])
dump("gsm8k_rewards.json", {"source": "rl.gsm8k.make_reward, ArgGYM a5b05fb", "stop_ids": STOP,
     "cases": [{"text": t, "gold": g, "ids": i, "reward": r, "correct": gsm8k.correct(t, g),
                "boxed": gsm8k.last_boxed(t)} for (t, g, i), r in zip(G, rewards)]})

# 4. Curriculum: TRL's build_curriculum on 003's config, one block of 16 a step, prompts once,
#    fed 40 steps of synthetic outcomes in TRL's order.
extra_info = [{"index": i, "id": r["id"], "task": r["task"], "level": r["level"]}
              for i, r in enumerate(trl["train"])]
ccfg = {**cfg, "prompts_per_step": 16,
        "grpo": {**cfg["grpo"], "num_generations": 1, "gradient_accumulation_steps": 1}}
cur = build_curriculum(ccfg, trl["train"])
initial = {"cells": {f"{t}|L{lv}": rows for (t, lv), rows in cur.cells.items()},
           "counts": {f"{t}|L{lv}": v for (t, lv), v in cur.counts.items()}}

def completions(step, rows, n=6):
    rng = random.Random(step)
    out = []
    for r in rows:
        info = extra_info[r]
        p = (r % 7) / 6
        for _ in range(n):
            refused = rng.random() < 0.03
            success = None if refused else rng.random() < p
            truncated = rng.random() < 0.1
            out.append({"id": info["id"], "task": info["task"], "level": info["level"],
                        "success": success, "truncated": truncated})
    return out

it = iter(cur)
steps = []
for step in range(40):
    rows = [next(it) for _ in range(16)]
    cur.update(group_outcomes(completions(step, rows)))
    steps.append({"rows": rows, "cells": cur.draws[step]["cells"],
                  "outcomes": cur.draws[step]["outcomes"],
                  "counts": {f"{t}|L{lv}": v for (t, lv), v in cur.counts.items()}})
state = cur.state_dict()
state["rng"] = json.loads(json.dumps(state["rng"]))
(OUT / "curriculum_trace.json").write_text(json.dumps({
    "source": "rl.train.build_curriculum (rl.curriculum.Curriculum) on rl/configs/qwen3.5-4b-003.yaml "
              "and its 5,700 training rows, 16 prompts a step yielded once, ArgGYM a5b05fb; "
              "outcomes from the synthetic generator in arggym_verl/tests/test_curriculum.py",
    "initial": initial, "steps": steps, "final_state": state},
    sort_keys=True, separators=(",", ":")) + "\n")

# 5. Experiment 003's settings the verl config takes.
em = rl_config.eval_model(cfg)
g = cfg["grpo"]
dump("exp003.json", {
    "source": "rl/configs/qwen3.5-4b-003.yaml and evals/conf/model/hf-qwen3.5-4b.yaml through "
              "rl.config.eval_model / dev_sampling, ArgGYM a5b05fb",
    "seed": cfg["seed"], "model_revision": cfg["model"]["revision"],
    "chat_template_kwargs": em["chat_template_kwargs"], "dev_sampling": rl_config.dev_sampling(cfg),
    "grpo": {k: g[k] for k in ("max_completion_length", "generation_kwargs", "mask_truncated_completions",
                               "temperature", "top_p", "top_k", "num_generations_eval", "max_steps")},
    "vllm_max_model_len": cfg["vllm"]["max_model_len"], "curriculum": cfg["curriculum"]})

cfg = rl_config.load(str(pool.ROOT / "rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml"),
                     
                     [f"data.screen.path={REPO}/data/gsm8k-screen/qwen3-0.6b-c1899de"])
train, dev = gsm8k.load_splits(cfg["data"])
rows = {"train": gsm8k.trl_rows(train), "dev": gsm8k.trl_rows(dev)}
(REPO / "arggym_verl/tests/fixtures/gsm8k_prompts.json").write_text(json.dumps({
    "source": "rl.gsm8k.load_splits and trl_rows on rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml "
              "with the committed screen, ArgGYM a5b05fb",
    "sha256": {s: sha([[r["id"], r["prompt"], r["gold"]] for r in v]) for s, v in rows.items()},
    "examples": {s: v[:2] for s, v in rows.items()}}, indent=1, sort_keys=True) + "\n")
print({s: len(v) for s, v in rows.items()})
