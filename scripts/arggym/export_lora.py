"""Export a verl FSDP2 LoRA-only checkpoint as a PEFT adapter and a merged bf16 model.

The verl checkpoint (save_lora_only) holds only LoRA tensors, as fp32 DTensors sharded
on dim 0 across ranks. This script

  1. gathers the shards into full tensors (fp32, unchanged),
  2. writes adapter/ : PEFT adapter_config.json + adapter_model.safetensors + tokenizer files,
  3. writes merged/  : the base checkpoint's own safetensors layout (same shard files, same
     index, vision tower and MTP weights byte-identical) with W <- bf16(W + alpha/r * B @ A)
     for every LoRA-targeted linear, plus the base's config/processor/tokenizer files.

Then `--verify` runs CPU checks: shard coverage, adapter load, adapter-vs-merged logits,
log-likelihood of step rollouts under base vs merged, and a greedy generate from merged/.

Usage (CPU only; the GPUs belong to training):
  python scripts/arggym/export_lora.py --run outputs/<run> --step 60
  python scripts/arggym/export_lora.py --run outputs/<run> --step 60 --verify-only
Output goes to outputs/export/step<N>/ unless --out is given.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

RUN: Path  # set from --run in main()
BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
BASE_DIR = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / BASE_REV
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]
MERGED_EXTRA_FILES = ["config.json", "preprocessor_config.json", "video_preprocessor_config.json",
                      "model.safetensors.index.json"]
PEFT_PREFIX = "base_model.model."  # PEFT key = PEFT_PREFIX + base module path + ".lora_X.weight"


# ---------------------------------------------------------------- gather shards
def load_shards(actor: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Gather the rank shards into full tensors. Returns (state_dict with verl keys, stats)."""
    meta = json.loads((actor / "fsdp_config.json").read_text())
    world = meta["world_size"]
    ranks = [torch.load(actor / f"model_world_size_{world}_rank_{r}.pt", map_location="cpu", weights_only=False)
             for r in range(world)]
    keys = list(ranks[0])
    assert all(list(sd) == keys for sd in ranks), "ranks disagree on key set"
    out, sq_norm, kinds = {}, 0.0, set()
    for k in keys:
        parts = [sd[k] for sd in ranks]
        t0 = parts[0]
        if hasattr(t0, "placements"):  # DTensor
            pl = tuple(t0.placements)
            kinds.add(f"DTensor{pl}")
            locs = [p._local_tensor for p in parts]
            sq_norm += sum(float(l.double().pow(2).sum()) for l in locs if pl[0].is_shard())
            if pl[0].is_shard():
                full = torch.cat(locs, dim=pl[0].dim)
            else:  # Replicate
                assert all(torch.equal(l, locs[0]) for l in locs[1:]), k
                full = locs[0]
                sq_norm += float(full.double().pow(2).sum())
            assert tuple(full.shape) == tuple(t0.shape), (k, full.shape, t0.shape)
        else:
            kinds.add("plain")
            assert all(torch.equal(p, t0) for p in parts[1:]), k
            full = t0
            sq_norm += float(full.double().pow(2).sum())
        out[k] = full.contiguous()
    return out, {"world_size": world, "n": len(keys), "norm": sq_norm ** 0.5, "kinds": sorted(kinds),
                 "dtypes": sorted({str(v.dtype) for v in out.values()})}


def to_peft_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        assert "lora_" in k and k.endswith(".default.weight") and k.startswith(PEFT_PREFIX), k
        out[k.replace(".default.weight", ".weight")] = v
    return out


def lora_pairs(peft_sd: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map base weight name (as in the base safetensors) -> (A, B)."""
    pairs = {}
    for k in peft_sd:
        if ".lora_A." in k:
            mod = k[len(PEFT_PREFIX):].replace(".lora_A.weight", "")
            pairs[mod + ".weight"] = (peft_sd[k], peft_sd[k.replace(".lora_A.", ".lora_B.")])
    return pairs


# ---------------------------------------------------------------- export
def export(actor: Path, out: Path) -> None:
    from peft import LoraConfig

    sd, stats = load_shards(actor)
    print(f"shards: {stats}")
    meta = json.loads((actor / "lora_train_meta.json").read_text())
    r, alpha = meta["r"], meta["lora_alpha"]
    peft_sd = to_peft_keys(sd)
    targets = sorted({k.split(".lora_")[0].rsplit(".", 1)[1] for k in peft_sd})
    for k, v in peft_sd.items():
        if ".lora_A." in k:
            assert v.shape[0] == r, (k, v.shape)

    # adapter/
    ad = out / "adapter"
    ad.mkdir(parents=True, exist_ok=True)
    save_file(peft_sd, ad / "adapter_model.safetensors", metadata={"format": "pt"})
    cfg = LoraConfig(task_type=meta.get("task_type", "CAUSAL_LM"), r=r, lora_alpha=alpha, lora_dropout=0.0,
                     target_modules=targets, exclude_modules=".*visual.*", bias="none",
                     base_model_name_or_path=BASE_ID, revision=BASE_REV)
    cfg.save_pretrained(ad)
    for f in TOKENIZER_FILES:
        shutil.copy2(BASE_DIR / f, ad / f)
    print(f"adapter: {len(peft_sd)} tensors, targets {targets}, r {r}, alpha {alpha}")

    # merged/ : rewrite the base shards, changing only LoRA-targeted weights
    md = out / "merged"
    md.mkdir(parents=True, exist_ok=True)
    pairs = lora_pairs(peft_sd)
    scale = alpha / r
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    missing = set(pairs) - set(index["weight_map"])
    assert not missing, f"LoRA targets absent from base: {sorted(missing)[:5]}"
    done = 0
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(BASE_DIR / shard, "pt") as f:
            st_meta = f.metadata()
        tensors = load_file(BASE_DIR / shard)
        for name in tensors:
            if name in pairs:
                a, b = pairs[name]
                w = tensors[name]
                assert w.shape == (b.shape[0], a.shape[1]), (name, w.shape)
                tensors[name] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype).contiguous()
                done += 1
        save_file(tensors, md / shard, metadata=st_meta)
        del tensors
    assert done == len(pairs), (done, len(pairs))
    for f in MERGED_EXTRA_FILES + TOKENIZER_FILES:
        shutil.copy2(BASE_DIR / f, md / f)
    # The base repo publishes no generation_config.json. Write token ids only (no sampling
    # values): stop on <|im_end|> or <|endoftext|>, pad with <|endoftext|>, as Qwen3 releases do.
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    im_end, eot = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    GenerationConfig(eos_token_id=[im_end, eot], pad_token_id=eot).save_pretrained(md)
    for p in list(md.iterdir()) + list(ad.iterdir()):
        p.chmod(0o644)
    (out / "SOURCE.json").write_text(json.dumps(
        {"checkpoint": str(actor), "base": BASE_ID, "base_revision": BASE_REV, "r": r, "lora_alpha": alpha,
         "merged_weights": done, "shard_stats": {k: v for k, v in stats.items()}}, indent=2))
    print(f"merged: {done} weights updated (scale {scale})")


# ---------------------------------------------------------------- verify
def load_vl(path, **kw):
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, **kw).eval()


def rollout_ids(tok, row, max_out):
    """Re-render the prompt with the chat template; check it decodes to the logged (special-stripped) input."""
    text = row["input"]
    assert text.startswith("system\n") and text.endswith("\nassistant\n<think>\n"), text[-60:]
    system, user = text[len("system\n"):-len("\nassistant\n<think>\n")].split("\nuser\n", 1)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    if isinstance(p_ids, dict) or hasattr(p_ids, "input_ids"):
        p_ids = p_ids["input_ids"]
    assert tok.decode(p_ids, skip_special_tokens=True) == text, "prompt re-render mismatch"
    o_ids = tok(row["output"], add_special_tokens=False)["input_ids"][:max_out]
    return p_ids, o_ids


@torch.no_grad()
def mean_ll(model, p_ids, o_ids):
    ids = torch.tensor([p_ids + o_ids])
    logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(o_ids)[:, None])
    return float(lp.mean())


@torch.no_grad()
def verify(actor: Path, out: Path, step: int, n_rollouts: int, max_out: int) -> None:
    import logging

    from peft import PeftModel
    from transformers import AutoTokenizer

    ad, md = out / "adapter", out / "merged"
    # 1. coverage
    sd, stats = load_shards(actor)
    peft_file = load_file(ad / "adapter_model.safetensors")
    ref = to_peft_keys(sd)
    assert set(peft_file) == set(ref)
    assert all(torch.equal(peft_file[k], ref[k]) for k in ref)
    fnorm = sum(float(v.double().pow(2).sum()) for v in peft_file.values()) ** 0.5
    b_norms = [float(v.norm()) for k, v in peft_file.items() if ".lora_B." in k]
    print(f"[coverage] shards n={stats['n']} norm={stats['norm']:.6f} | adapter n={len(peft_file)} "
          f"norm={fnorm:.6f} | lora_B zero: {sum(x == 0 for x in b_norms)}/{len(b_norms)}, "
          f"min/median/max |B| {min(b_norms):.4f}/{sorted(b_norms)[len(b_norms)//2]:.4f}/{max(b_norms):.4f}")

    # 2. adapter load, with warnings captured
    tok = AutoTokenizer.from_pretrained(ad)
    base = load_vl(BASE_DIR)
    caught_logs = []
    h = logging.Handler(); h.emit = lambda rec: caught_logs.append(rec.getMessage())
    logging.getLogger().addHandler(h)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, ad)
    logging.getLogger().removeHandler(h)
    msgs = [str(x.message) for x in w] + caught_logs
    bad = [m for m in msgs if "missing" in m.lower() or "unexpected" in m.lower() or "lora" in m.lower()]
    loaded = {k.replace(".default.weight", ".weight"): v for k, v in model.state_dict().items() if "lora_" in k}
    same = set(loaded) == set(peft_file) and all(torch.equal(loaded[k].float(), peft_file[k].float()) for k in loaded)
    print(f"[adapter load] lora modules in model={len(loaded)} file={len(peft_file)} identical={same} "
          f"load warnings mentioning missing/unexpected/lora: {bad or 'none'}")
    assert same

    # 3. adapter-on-base vs merged logits
    merged = load_vl(md)
    prompts = ["The capital of France is", "def fibonacci(n):",
               "In defeasible logic, a strict rule differs from a defeasible rule because"]
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        la = model(input_ids=ids).logits.float()
        lm = merged(input_ids=ids).logits.float()
        with model.disable_adapter():
            lb = model(input_ids=ids).logits.float()
        print(f"[logits] {p[:30]!r:34} max|adapter-merged|={float((la-lm).abs().max()):.4f} "
              f"max|adapter-base|={float((la-lb).abs().max()):.4f} logit scale max|x|={float(la.abs().max()):.1f} "
              f"argmax agree adapter/merged={float((la.argmax(-1)==lm.argmax(-1)).float().mean()):.2f}")

    # 4. rollout log-likelihood, base vs merged
    rows = [json.loads(l) for l in open(RUN / "rollouts" / f"{step + 1}.jsonl")]
    picked, seen = [], set()
    for r in rows:  # one per distinct task, to spread coverage
        if r["gts"] not in seen:
            seen.add(r["gts"]); picked.append(r)
        if len(picked) == n_rollouts:
            break
    for r in picked:
        p_ids, o_ids = rollout_ids(tok, r, max_out)
        with model.disable_adapter():
            llb = mean_ll(model, p_ids, o_ids)
        llm = mean_ll(merged, p_ids, o_ids)
        lla = mean_ll(model, p_ids, o_ids)
        print(f"[rollout ll] {r['gts']:45} score={r['score']} prompt={len(p_ids)} out={len(o_ids)} "
              f"base={llb:.4f} merged={llm:.4f} adapter={lla:.4f} delta={llm-llb:+.4f}")
    del model, base, merged

    # 5. reload merged/ cold and generate
    from transformers import AutoConfig
    arch = AutoConfig.from_pretrained(md).architectures
    tok2 = AutoTokenizer.from_pretrained(md)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m2, info = __import__("transformers").AutoModelForImageTextToText.from_pretrained(
            md, dtype=torch.bfloat16, output_loading_info=True)
    print(f"[reload] architectures={arch} class={type(m2).__name__} missing={info['missing_keys']} "
          f"unexpected(n)={len(info['unexpected_keys'])} sample={info['unexpected_keys'][:2]}")
    msgs = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
    ids = tok2.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                   enable_thinking=False)
    g = m2.generate(**ids, max_new_tokens=16, do_sample=False)
    print(f"[generate] {tok2.decode(g[0, ids['input_ids'].shape[1]:])!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run directory, e.g. outputs/<run>")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--max-out-tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    global RUN
    RUN = a.run.resolve()
    torch.set_num_threads(a.threads)
    actor = RUN / "checkpoints" / f"global_step_{a.step}" / "actor"
    out = a.out or a.run.resolve().parent / "export" / f"step{a.step}"
    if not a.verify_only:
        export(actor, out)
    if not a.no_verify:
        verify(actor, out, a.step, a.rollouts, a.max_out_tokens)


if __name__ == "__main__":
    main()
