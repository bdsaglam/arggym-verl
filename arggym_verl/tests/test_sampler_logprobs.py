"""The trainer's log-probs under the rollout's sampling (verl/utils/sampler_logprobs.py) equal
vLLM's processed log-probs on the same logits, and a dense reference's values and gradients.

vLLM's own penalty and top-k/top-p functions run here on CPU (the PyTorch top-k/top-p path;
its Triton kernel needs a GPU). The dense reference spells out the rule the module states:
penalty on the response's tokens before each position, top-k with ties to the lower id,
top-p as the shortest prefix reaching p, the sampled token added when outside.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from verl.utils import sampler_logprobs
from verl.utils.sampler_logprobs import CHUNK_ROWS, TIE_CANDIDATES, SamplerParams, sampler_log_probs

CARD = SamplerParams(presence_penalty=1.5, top_k=20, top_p=0.95)


def pack(prompts, responses):
    """Packed input ids, cu_seqlens and prompt lengths, as verl's remove-padding path has them."""
    seqs = [torch.cat([p, r]) for p, r in zip(prompts, responses)]
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + len(s))
    return torch.cat(seqs), cu, [len(p) for p in prompts]


def reference(logits, input_ids, cu, prompt_lens, params, temperature=1.0, with_entropy=False):
    """Row by row, the module's rule from the raw logits in vLLM's order: penalty, temperature,
    top-k (ties to the lower id), top-p prefix, the label added if outside, log-softmax; with
    `with_entropy`, also the entropy of that distribution."""
    labels = torch.roll(input_ids, -1)
    out = torch.empty(logits.shape[0], dtype=torch.float64)
    entropy = torch.empty_like(out)
    for s, p in enumerate(prompt_lens):
        for i in range(cu[s], cu[s + 1]):
            r = i - cu[s] + 1 - p  # response position of the token row i predicts
            seen = set(input_ids[cu[s] + p : cu[s] + p + max(r, 0)].tolist())
            z = logits[i].double().clone()
            for v in seen:
                z[v] -= params.presence_penalty
            z /= temperature
            y = int(labels[i])
            if params.top_k > 0:
                order = sorted(range(len(z)), key=lambda v: (-z[v].item(), v))[: params.top_k]
                probs = torch.softmax(z[order], 0)
                keep, mass = [], 0.0
                for v, q in zip(order, probs.tolist()):
                    if mass < params.top_p:
                        keep.append(v)
                    mass += q
                support = set(keep) | {y}
            else:
                support = set(range(len(z)))
            idx = torch.tensor(sorted(support))
            out[i] = z[y] - torch.logsumexp(z[idx], 0)
            entropy[i] = torch.special.entr(torch.softmax(z[idx], 0)).sum()
    return (out, entropy) if with_entropy else out


def run(logits, input_ids, cu, prompt_lens, params, temperature=1.0, inplace=False, with_entropy=False):
    rows = logits.shape[0]
    t = torch.full((rows,), float(temperature))
    return sampler_log_probs(logits / temperature, torch.roll(input_ids, -1), params, cu, prompt_lens,
                             input_ids, temperature=t, inplace_backward=inplace, with_entropy=with_entropy)


def random_batch(seed, vocab=60, lens=((5, 40), (3, 25), (7, 1))):
    g = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(vocab, (p,), generator=g) for p, _ in lens]
    # a small vocabulary, so responses repeat tokens often
    responses = [torch.randint(vocab, (r,), generator=g) for _, r in lens]
    input_ids, cu, prompt_lens = pack(prompts, responses)
    logits = torch.randn(len(input_ids), vocab, generator=g) * 2
    return logits, input_ids, cu, prompt_lens


@pytest.mark.parametrize("params", [CARD, SamplerParams(presence_penalty=1.5),
                                    SamplerParams(top_k=20, top_p=0.95), SamplerParams(top_k=7),
                                    SamplerParams(presence_penalty=-0.5, top_k=5, top_p=0.5)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_matches_the_dense_reference(params, temperature):
    logits, input_ids, cu, prompt_lens = random_batch(0)
    got = run(logits, input_ids, cu, prompt_lens, params, temperature)
    want = reference(logits, input_ids, cu, prompt_lens, params, temperature)
    torch.testing.assert_close(got.double(), want, atol=1e-5, rtol=0)


def test_off_is_the_plain_log_softmax():
    logits, input_ids, cu, prompt_lens = random_batch(1)
    got = run(logits, input_ids, cu, prompt_lens, SamplerParams(presence_penalty=0.0, top_k=-1))
    want = torch.log_softmax(logits, 1).gather(1, torch.roll(input_ids, -1).unsqueeze(1)).squeeze(1)
    assert not SamplerParams().active and not SamplerParams(top_k=-1, top_p=1.0).active
    torch.testing.assert_close(got, want)


def test_equals_vllms_processed_logprobs_on_a_generated_response():
    """Generate as vLLM's V1 sampler does, with its own functions: fp32 logits, apply_penalties
    with the prompt and the output so far, the temperature, apply_top_k_top_p, log_softmax; sample;
    keep the sampled token's log-prob. On the same logits the trainer's log-probs are vLLM's."""
    from vllm.model_executor.layers.utils import apply_penalties
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    vocab, temperature = 50, 0.8
    g = torch.Generator().manual_seed(2)
    prompts, responses, rows, vllm_lp = [], [], [], []
    for p_len, r_len in ((6, 120), (4, 80)):
        prompt = torch.randint(vocab, (p_len,), generator=g)
        prompt_logits = torch.randn(p_len - 1, vocab, generator=g)
        out, out_logits = [], []
        for _ in range(r_len):
            raw = torch.randn(1, vocab, generator=g) * 1.5
            z = apply_penalties(raw.clone(), prompt.unsqueeze(0), torch.tensor([out + [vocab]]),
                                torch.tensor([CARD.presence_penalty]), torch.zeros(1), torch.ones(1))
            z = apply_top_k_top_p_pytorch(z / temperature, torch.tensor([CARD.top_k]), torch.tensor([CARD.top_p]))
            lp = torch.log_softmax(z, -1)[0]
            tok = int(torch.multinomial(lp.exp(), 1, generator=g))
            out.append(tok)
            out_logits.append(raw[0])
            vllm_lp.append(lp[tok].item())
        prompts.append(prompt)
        responses.append(torch.tensor(out))
        # row j predicts token j + 1: the prompt's rows, then one row per response token; the last
        # row (which predicts the next sequence's first token) is any.
        rows += [prompt_logits, torch.stack(out_logits), torch.randn(1, vocab, generator=g)]
    input_ids, cu, prompt_lens = pack(prompts, responses)
    logits = torch.cat(rows)
    assert len({t for r in responses for t in r.tolist()}) < sum(len(r) for r in responses)  # repeats
    got = run(logits, input_ids, cu, prompt_lens, CARD, temperature)
    trainer_lp = torch.cat([got[cu[s] + p - 1 : cu[s + 1] - 1] for s, p in enumerate(prompt_lens)])
    torch.testing.assert_close(trainer_lp, torch.tensor(vllm_lp), atol=1e-5, rtol=0)


def test_prompt_tokens_are_not_penalised():
    """vLLM's presence penalty counts output tokens only: a token in the prompt, not yet in the
    response, keeps its logit; the same token once in the response loses 1.5."""
    vocab = 30
    prompt, response = torch.tensor([3, 4, 3]), torch.tensor([5, 3, 3])
    input_ids, cu, prompt_lens = pack([prompt], [response])
    logits = torch.zeros(len(input_ids), vocab)
    lp = run(logits, input_ids, cu, prompt_lens, SamplerParams(presence_penalty=1.5))
    # rows 2, 3, 4 predict response tokens 0, 1, 2 (5, 3, 3)
    plain = -math.log(vocab)
    assert lp[2].item() == pytest.approx(plain)  # 5: nothing seen yet
    assert lp[3].item() == pytest.approx(-math.log(vocab - 1 + math.exp(-1.5)))  # 3: in the prompt only
    assert lp[4].item() == pytest.approx(-1.5 - math.log(vocab - 2 + 2 * math.exp(-1.5)))  # 3, after 5 and 3


def test_top_k_ties_go_to_the_lower_id_beyond_the_candidates():
    """Logits on a coarse grid tie at the k-th value; one row has more ties than the extra
    candidates topk reads, which the exact fallback places."""
    vocab, k = 200, 20
    g = torch.Generator().manual_seed(3)
    logits = (torch.randn(6, vocab, generator=g) * 4).mul(2).round().div(2)  # a 0.5 grid
    logits[0] = -5.0
    logits[0, 150:155] = 3.0  # 5 above, then 3 * TIE_CANDIDATES ties at 0 from id 40 up
    logits[0, 40 : 40 + 3 * TIE_CANDIDATES] = 0.0
    prompt, response = torch.tensor([7]), torch.tensor([150, 41, 2, 9, 11])
    input_ids, cu, prompt_lens = pack([prompt], [response])
    params = SamplerParams(top_k=k)
    got = run(logits, input_ids, cu, prompt_lens, params)
    want = reference(logits, input_ids, cu, prompt_lens, params)
    torch.testing.assert_close(got.double(), want, atol=1e-5, rtol=0)
    # row 0: support 5 + 15 lowest-id ties, 20 tokens at logits 3 and 0
    assert got[0].item() == pytest.approx(3.0 - math.log(5 * math.exp(3.0) + 15))
    # a tied token past the 15 is outside: added to the support alone
    input_ids2, cu2, pl2 = pack([prompt], [torch.tensor([40 + 20, 41, 2, 9, 11])])
    lp = run(logits, input_ids2, cu2, pl2, params)
    assert lp[0].item() == pytest.approx(0.0 - math.log(5 * math.exp(3.0) + 16))


@pytest.mark.parametrize("p, kept", [(0.79, 2), (0.81, 3), (0.96, 4)])
def test_top_p_keeps_the_shortest_prefix_reaching_p(p, kept):
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = torch.full((3, 10), -30.0)
    logits[0, [4, 1, 7, 2]] = probs.log()
    prompt, response = torch.tensor([0]), torch.tensor([4, 0])
    input_ids, cu, prompt_lens = pack([prompt], [response])
    lp = run(logits, input_ids, cu, prompt_lens, SamplerParams(top_k=4, top_p=p))
    assert lp[0].item() == pytest.approx(math.log(0.5 / probs[:kept].sum().item()), abs=1e-6)


def test_a_sampled_token_outside_the_support_stays_finite():
    logits = torch.zeros(3, 40)
    logits[0, :20] = 5.0
    prompt, response = torch.tensor([1]), torch.tensor([30, 2])  # 30 is outside the top 20
    input_ids, cu, prompt_lens = pack([prompt], [response])
    lp = run(logits, input_ids, cu, prompt_lens, SamplerParams(top_k=20))
    assert lp[0].item() == pytest.approx(0.0 - math.log(20 * math.exp(5.0) + 1))


@pytest.mark.parametrize("params", [CARD, SamplerParams(presence_penalty=1.5)])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("chunk_rows, dtype", [(CHUNK_ROWS, torch.float32), (7, torch.float32), (7, torch.bfloat16)])
def test_gradient_matches_autograd_through_the_dense_reference(params, inplace, chunk_rows, dtype, monkeypatch):
    """Also with chunks of 7 rows, so the backward's chunk loop crosses sequences' rows, and on
    BF16 logits, as the trainer's are: a coarse grid with ties, the gradient in BF16."""
    monkeypatch.setattr(sampler_logprobs, "CHUNK_ROWS", chunk_rows)
    logits, input_ids, cu, prompt_lens = random_batch(4)
    logits = logits.to(dtype)
    temperature = 1.0 if dtype == torch.bfloat16 else 0.7  # verl divides BF16 logits in BF16
    labels = torch.roll(input_ids, -1)
    weights = torch.randn(len(input_ids), generator=torch.Generator().manual_seed(5))

    x = logits.clone().requires_grad_()
    lp, _ = run(x, input_ids, cu, prompt_lens, params, temperature, inplace, with_entropy=True)
    (lp * weights).sum().backward()
    assert x.grad.dtype == dtype

    # Dense: the same transforms as constants (seen mask, support), autograd through the rest.
    y = logits.clone().double().requires_grad_()
    total = 0.0
    for s, p in enumerate(prompt_lens):
        for i in range(cu[s], cu[s + 1]):
            r = i - cu[s] + 1 - p
            seen = torch.zeros(logits.shape[1], dtype=torch.bool)
            seen[input_ids[cu[s] + p : cu[s] + p + max(r, 0)]] = True
            z = (y[i] - params.presence_penalty * seen) / temperature
            support = torch.ones_like(seen)
            if params.masks:
                zd = z.detach()
                order = sorted(range(len(zd)), key=lambda v: (-zd[v].item(), v))[: params.top_k]
                pr = torch.softmax(zd[order], 0).tolist()
                keep = [v for j, v in enumerate(order) if sum(pr[:j]) < params.top_p]
                support = torch.zeros_like(seen)
                support[keep + [int(labels[i])]] = True
            total = total + weights[i] * (z[labels[i]] - torch.logsumexp(z.masked_fill(~support, -math.inf), 0))
    total.backward()
    tol = dict(atol=1e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x.grad.double(), y.grad, **tol)


@pytest.mark.parametrize("params", [CARD, SamplerParams(presence_penalty=1.5), SamplerParams(top_k=7)])
@pytest.mark.parametrize("chunk_rows", [CHUNK_ROWS, 7])
def test_entropy_is_the_sampled_distributions(params, chunk_rows, monkeypatch):
    """The entropy returned beside the log-probs is the reference distribution's: over the
    support with the sampled token, after the penalty; the log-probs are unchanged by asking."""
    monkeypatch.setattr(sampler_logprobs, "CHUNK_ROWS", chunk_rows)
    logits, input_ids, cu, prompt_lens = random_batch(8)
    lp, entropy = run(logits, input_ids, cu, prompt_lens, params, 0.7, with_entropy=True)
    want_lp, want_entropy = reference(logits, input_ids, cu, prompt_lens, params, 0.7, with_entropy=True)
    torch.testing.assert_close(lp.double(), want_lp, atol=1e-5, rtol=0)
    torch.testing.assert_close(entropy.double(), want_entropy, atol=1e-5, rtol=0)
    torch.testing.assert_close(run(logits, input_ids, cu, prompt_lens, params, 0.7), lp)
    if params.masks:  # the support is at most k + 1 tokens
        assert entropy.max().item() <= math.log(params.top_k + 1) + 1e-6


def test_long_sequences_cross_chunks():
    """Rows of one sequence over several chunks carry the penalty across chunk boundaries."""
    vocab = 40
    g = torch.Generator().manual_seed(6)
    prompts = [torch.randint(vocab, (3,), generator=g), torch.randint(vocab, (2,), generator=g)]
    responses = [torch.randint(vocab, (CHUNK_ROWS + 50,), generator=g), torch.randint(vocab, (30,), generator=g)]
    input_ids, cu, prompt_lens = pack(prompts, responses)
    logits = torch.randn(len(input_ids) + 5, vocab, generator=g)  # 5 padding rows at the end
    labels = torch.cat([torch.roll(input_ids, -1), torch.zeros(5, dtype=torch.long)])
    got = sampler_log_probs(logits, labels, CARD, cu, prompt_lens, input_ids)
    want = reference(logits[: len(input_ids)], input_ids, cu, prompt_lens, CARD)
    torch.testing.assert_close(got[: len(input_ids)].double(), want, atol=1e-5, rtol=0)
    assert torch.isfinite(got).all()


def test_top_p_without_top_k_is_refused():
    with pytest.raises(NotImplementedError):
        SamplerParams(top_p=0.95)


# ---------------------------------------------------------------- verl's FSDP log-prob path


def engine_log_probs(logits, prompts, responses, extra, calculate_entropy=False):
    """FSDPEngineWithLMHead's prepare_model_inputs/prepare_model_outputs on a packed micro-batch
    built as the transfer queue hands it over (nested input ids, prompts and responses)."""
    from tensordict import TensorDict

    import verl.utils.torch_functional as verl_F
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.utils.padding import response_from_nested

    seqs = [torch.cat([p, r]) for p, r in zip(prompts, responses)]
    nested = lambda ts: torch.nested.as_nested_tensor(ts, layout=torch.jagged)  # noqa: E731
    mb = TensorDict({"input_ids": nested(seqs), "position_ids": nested([torch.arange(len(s)) for s in seqs]),
                     "prompts": nested(prompts), "responses": nested(responses),
                     "response_mask": nested([torch.ones_like(r) for r in responses])},
                    batch_size=len(seqs))
    tu.assign_non_tensor(mb, temperature=1.0, use_remove_padding=True, calculate_entropy=calculate_entropy,
                         **extra)
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length, engine.use_ulysses_sp = False, False
    engine.engine_config = SimpleNamespace(entropy_from_logits_with_chunking=True, entropy_checkpointing=False,
                                           entropy_from_logits_chunk_size=2048)
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits_with_chunking
    _, output_args = engine.prepare_model_inputs(mb)
    out = engine.prepare_model_outputs(SimpleNamespace(logits=logits.unsqueeze(0)), output_args, mb, None)
    return response_from_nested(out["log_probs"], mb["response_mask"]), out


def test_verls_log_prob_path_applies_the_batch_keys(monkeypatch):
    """With the batch keys MatchSampler sets, the response log-probs verl hands back are the
    reference's at each response position; without them, the plain softmax's."""
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    logits, input_ids, cu, prompt_lens = random_batch(7)
    prompts = [input_ids[cu[s] : cu[s] + p] for s, p in enumerate(prompt_lens)]
    responses = [input_ids[cu[s] + p : cu[s + 1]] for s, p in enumerate(prompt_lens)]
    keys = {"sampler_presence_penalty": 1.5, "sampler_top_k": 20, "sampler_top_p": 0.95}
    got, out = engine_log_probs(logits, prompts, responses, keys, calculate_entropy=True)
    want = reference(logits, input_ids, cu, prompt_lens, CARD)
    for s, p in enumerate(prompt_lens):
        torch.testing.assert_close(got[s].double(), want[cu[s] + p - 1 : cu[s + 1] - 1], atol=1e-5, rtol=0)
    # entropy stays the full distribution's; sampler_entropy is the sampled one's
    torch.testing.assert_close(out["entropy"].values(), torch.special.entr(logits.softmax(1)).sum(1))
    _, want_entropy = reference(logits, input_ids, cu, prompt_lens, CARD, with_entropy=True)
    torch.testing.assert_close(out["sampler_entropy"].values().double(), want_entropy, atol=1e-5, rtol=0)
    assert torch.equal(out["sampler_entropy"].offsets(), out["log_probs"].offsets())
    _, out = engine_log_probs(logits, prompts, responses, keys)  # the reference and update passes
    assert "sampler_entropy" not in out
    plain, out = engine_log_probs(logits, prompts, responses, {}, calculate_entropy=True)
    assert "sampler_entropy" not in out
    lsm = torch.log_softmax(logits, 1).gather(1, torch.roll(input_ids, -1).unsqueeze(1)).squeeze(1)
    for s, p in enumerate(prompt_lens):
        torch.testing.assert_close(plain[s], lsm[cu[s] + p - 1 : cu[s + 1] - 1])


def test_verls_log_prob_path_refuses_what_it_cannot_match(monkeypatch):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    mb = TensorDict({"input_ids": torch.nested.as_nested_tensor([torch.arange(4)], layout=torch.jagged)},
                    batch_size=1)
    tu.assign_non_tensor(mb, use_remove_padding=True, use_fused_kernels=True, sampler_top_k=20)
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    with pytest.raises(NotImplementedError, match="match_sampler"):
        engine.prepare_model_outputs(None, {}, mb, None)
