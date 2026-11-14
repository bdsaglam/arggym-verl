"""Heartbeat for a verl run: every 15 min, check health and early dead-end signals.

    python scripts/arggym/watch_verl.py outputs/<run> [interval_s]

Appends one status line per check to <run>/watch.log. Exits (so the waiting
agent is notified) on the first alarm, with the reason on stdout:
  done      the run wrote EXIT=
  crash     traceback, OOM, or the served-policy guard stopped it
  gpu       a GPU idle or empty for two checks in a row
  stall     no new step or dev eval for 3x the median step time (at least 5 h)
  precision clip fraction, IS ESS, clamped share or served/track out of range
  dead-end  no learning signal by the rules below
"""
import json, re, statistics as st, subprocess, sys, time
from pathlib import Path

run = Path(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 900
log, metrics, out = run / "driver.log", run / "metrics.jsonl", run / "watch.log"


def gpus():
    """Mean utilization over 6 samples 5 s apart, and memory, per GPU."""
    util, mem = None, None
    for _ in range(6):
        rows = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        vals = [tuple(map(float, r.split(","))) for r in rows.strip().splitlines()]
        util = [u for u, _ in vals] if util is None else [a + u for a, (u, _) in zip(util, vals)]
        mem = [m for _, m in vals]
        time.sleep(5)
    return [u / 6 for u in util], mem


def series(rows, key):
    return [(r["step"], r["data"][key]) for r in rows if key in r["data"]]


def slope(pts):
    """OLS slope per step and its standard error."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - my - b * (x - mx) for x, y in zip(xs, ys)]
    se = (sum(r * r for r in res) / (len(xs) - 2) / sxx) ** 0.5
    return b, se


def check(state):
    text = log.read_text(errors="replace") if log.exists() else ""
    new, state["offset"] = text[state["offset"]:], len(text)
    if "EXIT=" in new:
        return "done", re.search(r"EXIT=\d+", new).group()
    m = re.search(r"AUTO-STOP:|Traceback|OutOfMemoryError|CUDA out of memory|served.{0,40}(stop|track)[^\n]*below", new)
    if m:
        return "crash", new[max(0, m.start() - 200): m.start() + 400].replace("\n", " | ")

    util, mem = gpus()
    idle = [i for i, (u, m) in enumerate(zip(util, mem)) if u < 10 or m < 1000]
    state["idle"] = state["idle"] + 1 if idle else 0
    if state["idle"] >= 2:
        return "gpu", f"GPUs {idle} idle/empty for 2 checks: util {util} mem {mem}"

    rows = [json.loads(l) for l in metrics.read_text().splitlines()] if metrics.exists() else []
    rows = [r for r in rows if "step" in r and "data" in r]
    train = [r for r in rows if "critic/score/mean" in r["data"]]
    step = train[-1]["step"] if train else 0
    now = time.time()
    if (step, len(rows)) != state["step"]:  # a dev eval counts as progress too
        state["step"], state["since"] = (step, len(rows)), now
    times = [v for _, v in series(train, "timing_s/step")]
    # A step with a dev eval of 768 completions takes ~4.5 h; the GPU check catches a dead run sooner.
    limit = max(5 * 3600, 3 * st.median(times)) if times else 5 * 3600
    if now - state["since"] > limit:
        return "stall", f"no new step for {(now - state['since']) / 60:.0f} min at step {step}"

    dev = series(rows, next((k for r in rows for k in r["data"]
                             if k.startswith("val-core/") and "/acc/mean@" in k), "none"))
    status = (f"step {step} util {[round(u) for u in util]} mem {[round(m / 1024) for m in mem]}G "
              f"dev {[(s, round(v, 3)) for s, v in dev]}")
    if not train:
        return None, status

    last = train[-1]["data"]
    status += (f" score {last['critic/score/mean']:.3f} len {last.get('response_length/mean', 0):.0f}"
               f" trunc {last.get('response_length/clip_ratio', 0):.3f}"
               f" clip {last.get('actor/pg_clipfrac', 0):.4f}"
               f" ess {last.get('rollout_corr/rollout_is_eff_sample_size', 1):.3f}"
               f" track {last.get('served/track', float('nan')):.2f}"
               f" kl {last.get('actor/kl_loss', 0):.4f}")

    # Precision: what the GSM8K runs showed healthy, with margin.
    if last.get("actor/pg_clipfrac", 0) > 0.02:
        return "precision", "pg_clipfrac > 0.02 (phantom clipping?) | " + status
    if last.get("rollout_corr/rollout_is_eff_sample_size", 1) < 0.9:
        return "precision", "IS ESS < 0.9 | " + status
    if last.get("rollout_corr/rollout_is_ratio_fraction_high", 0) > 0.05:
        return "precision", "over 5% of tokens clamped by TIS | " + status
    if step >= 10 and last.get("served/armed") and last.get("served/track", 1) < 0.5:
        return "precision", "served/track < 0.5 after step 10 | " + status

    # Dead-end signals.
    if step >= 10 and max(v for _, v in series(train, "actor/kl_loss")) < 1e-4:
        return "dead-end", "KL to base still < 1e-4 at step 10: the adapter is not moving | " + status
    if step >= 15:
        tr = series(train, "response_length/clip_ratio")
        first, recent = st.mean(v for _, v in tr[:5]), st.mean(v for _, v in tr[-5:])
        if recent > first + 0.10:
            return "dead-end", f"truncation drift {first:.2f} -> {recent:.2f} | " + status
        ent = series(train, "actor/entropy")
        e0, e1 = st.mean(v for _, v in ent[:5]), st.mean(v for _, v in ent[-5:])
        if not 0.5 * e0 < e1 < 1.5 * e0:
            return "dead-end", f"entropy {e0:.3f} -> {e1:.3f} | " + status
    if step >= 30 and len(dev) >= 4:
        b, se = slope(series(train, "critic/score/mean")[-30:])
        gain = max(v for s, v in dev if s > 0) - dev[0][1]
        status += f" train-slope/10 {10 * b:+.3f}±{10 * se:.3f} dev-gain {gain:+.3f}"
        # GSM8K on verl gained +0.035 by step 20 and +0.06 by step 40, with the
        # training score's 30-step slope clearly positive by then.
        if gain < 0.03 and b < 2 * se:
            return "dead-end", "no dev gain over noise and no rising training score | " + status
    return None, status


state = {"offset": 0, "idle": 0, "step": None, "since": time.time()}
while True:
    kind, msg = check(state)
    line = f"{time.strftime('%m-%d %H:%M')} {kind or 'ok'} {msg}"
    with out.open("a") as f:
        f.write(line + "\n")
    if kind:
        print(line)
        sys.exit(0)
    time.sleep(every)
