# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer-side log-probs of the distribution vLLM sampled from.

vLLM (0.24, the V1 sampler, vllm/v1/sample/sampler.py) turns a row of logits into the
distribution it draws from in this order, in float32:

1. presence penalty: ``presence_penalty`` is subtracted from every token that appears
   among the request's output tokens so far (``apply_penalties`` in
   vllm/model_executor/layers/utils.py: ``logits -= presence_penalties * output_mask``;
   the prompt only enters the repetition penalty);
2. temperature: the logits are divided by it;
3. top-k, then top-p over the probabilities renormalised on the top-k
   (vllm/v1/sample/ops/topk_topp_sampler.py, ``apply_top_k_top_p``); everything else
   is set to -inf;
4. a softmax over what is left. With ``logprobs_mode: processed_logprobs`` the
   log-probs vLLM returns are this distribution's.

The trainer's log-prob is the plain softmax over the vocabulary. ``sampler_log_probs``
gives it the sampler's: per response position t, the penalty on the tokens of the
response before t, then top-k and top-p, then the log-softmax over the support.

Top-k keeps exactly k tokens, ties at the k-th value going to the lower token id; top-p
keeps the shortest prefix of the top-k, by probability, whose mass reaches p, ties again
to the lower id. That is vLLM's Triton kernel (``apply_top_k_top_p_triton``), the path
for batches of 8 and more; its PyTorch path, below 8, keeps every token tied with the
k-th. The sampled token is always inside vLLM's support but can fall just outside the
trainer's, whose logits differ by the engines' kernel gap; it is then added to the
support, so its log-prob stays finite.

The logits are read in chunks of rows within one sequence, each chunk turned into
float32 once; nothing of size tokens x vocabulary is kept beyond the logits and their
gradient. The penalty is scattered over the tokens the chunk's last row has seen
(``first_seen``: each token's first position in the response), so no seen-mask of size
tokens x vocabulary is built. With top-k the backward is sparse: k + 1 entries a row.

On request the same loop also gives each row's entropy under the sampled distribution
(``with_entropy``): over the support with top-k, where it is k + 1 entries a row, over
the penalised vocabulary without. The full-vocabulary entropy verl logs moves with
tokens the sampler never draws; this one does not.
"""

from dataclasses import dataclass
from typing import Optional

import torch

# Rows of logits turned into float32 at once: 1 GB at a 248k vocabulary.
CHUNK_ROWS = 1024
# Candidates read past the k-th, to place ties at the k-th value by token id.
TIE_CANDIDATES = 8
# first_seen of a token the response does not contain.
NEVER = torch.iinfo(torch.int32).max


@dataclass(frozen=True)
class SamplerParams:
    """The sampling transforms to reproduce; the defaults leave the softmax as it is."""

    presence_penalty: float = 0.0
    top_k: int = -1
    top_p: float = 1.0

    def __post_init__(self):
        if self.top_p < 1.0 and self.top_k <= 0:
            raise NotImplementedError("top-p without top-k: the support would need a full sort of the vocabulary")

    @property
    def masks(self) -> bool:
        return self.top_k > 0

    @property
    def active(self) -> bool:
        return self.presence_penalty != 0.0 or self.masks


def first_seen(input_ids: torch.Tensor, cu_seqlens: list[int], prompt_lens: list[int], vocab_size: int):
    """(sequences, vocab) int32: the response position where each token first appears, NEVER if it does not."""
    num_seqs = len(prompt_lens)
    index, position = [], []
    for s in range(num_seqs):
        response = input_ids[cu_seqlens[s] + prompt_lens[s] : cu_seqlens[s + 1]]
        index.append(response.long() + s * vocab_size)
        position.append(torch.arange(response.numel(), dtype=torch.int32, device=input_ids.device))
    first = torch.full((num_seqs * vocab_size,), NEVER, dtype=torch.int32, device=input_ids.device)
    first.scatter_reduce_(0, torch.cat(index), torch.cat(position), reduce="amin")
    return first.view(num_seqs, vocab_size)


def response_positions(cu_seqlens: list[int], prompt_lens: list[int], num_rows: int, device) -> torch.Tensor:
    """(rows,) long: for the row that predicts a sequence's token j, the number of response tokens before j.

    Row i of the packed logits predicts token i + 1. Rows that predict a prompt token, and
    padding rows after the last sequence, get 0: no token counts as seen.
    """
    pos = torch.zeros(num_rows, dtype=torch.long, device=device)
    for s, p in enumerate(prompt_lens):
        start, end = cu_seqlens[s], cu_seqlens[s + 1]
        # row start + j predicts token j + 1, response position j + 1 - p
        pos[start:end] = (torch.arange(end - start, device=device) + 1 - p).clamp_(min=0)
    return pos


def _chunks(cu_seqlens: list[int], num_rows: int):
    """(sequence, start, end) row ranges of at most CHUNK_ROWS within one sequence; sequence -1 for padding."""
    bounds = [(s, cu_seqlens[s], cu_seqlens[s + 1]) for s in range(len(cu_seqlens) - 1)]
    if cu_seqlens[-1] < num_rows:
        bounds.append((-1, cu_seqlens[-1], num_rows))
    for s, start, end in bounds:
        for a in range(start, end, CHUNK_ROWS):
            yield s, a, min(a + CHUNK_ROWS, end)


def _penalised(logits, a, b, s, pos, first, penalty):
    """Rows a:b of the logits in float32 with the presence penalty applied (a copy)."""
    z = logits[a:b].to(torch.float32, copy=True)
    if first is None or s < 0:
        return z
    last = int(pos[a:b].max())
    if last > 0:
        seen = (first[s] < last).nonzero().squeeze(1)  # tokens the chunk's last row has seen
        mask = first[s, seen].unsqueeze(0) < pos[a:b].unsqueeze(1)  # (rows, seen)
        z[:, seen] -= penalty[a:b].unsqueeze(1) * mask
    return z


def _top_k_by_id(z: torch.Tensor, k: int):
    """The k largest values of each row, descending, ties ordered by lower token id first."""
    rows, vocab = z.shape
    m = min(k + TIE_CANDIDATES, vocab)
    vals, idx = z.topk(m, dim=1)
    idx, order = idx.sort(dim=1)
    vals, order2 = vals.gather(1, order).sort(dim=1, descending=True, stable=True)
    idx = idx.gather(1, order2)
    if m < vocab:
        # A row whose k-th value is tied with its last candidate may have lower-id ties outside them.
        for r in (vals[:, k - 1] == vals[:, m - 1]).nonzero().squeeze(1).tolist():
            kth = vals[r, k - 1]
            above = (z[r] > kth).nonzero().squeeze(1)  # ascending ids
            above = above[z[r, above].sort(descending=True, stable=True).indices]
            tied = (z[r] == kth).nonzero().squeeze(1)[: k - above.numel()]
            keep = torch.cat([above, tied])
            idx[r, :k], vals[r, :k] = keep, z[r, keep]
    return vals[:, :k], idx[:, :k]


def _support(z: torch.Tensor, params: SamplerParams):
    """Top-k then top-p: (values, token ids), each (rows, k); values -inf where top-p cuts."""
    vals, idx = _top_k_by_id(z, min(params.top_k, z.shape[1]))
    if params.top_p < 1.0:
        probs = vals.softmax(dim=1)
        before = torch.nn.functional.pad(probs.cumsum(dim=1)[:, :-1], (1, 0))
        vals = vals.masked_fill(before >= params.top_p, float("-inf"))
    return vals, idx


class SamplerLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, penalty, pos, first, cu_seqlens, params, inplace_backward, with_entropy):
        num_rows = logits.shape[0]
        lse = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
        label_logit = torch.empty_like(lse)
        entropy = torch.empty(num_rows if with_entropy else 0, dtype=torch.float32, device=logits.device)
        if params.masks:
            k = min(params.top_k, logits.shape[1])
            support_vals = torch.empty(num_rows, k, dtype=torch.float32, device=logits.device)
            support_idx = torch.empty(num_rows, k, dtype=torch.long, device=logits.device)
        for s, a, b in _chunks(cu_seqlens, num_rows):
            z = _penalised(logits, a, b, s, pos, first, penalty)
            y = z.gather(1, labels[a:b].unsqueeze(1)).squeeze(1)
            if params.masks:
                vals, idx = _support(z, params)
                inside = ((idx == labels[a:b].unsqueeze(1)) & (vals > float("-inf"))).any(dim=1)
                added = torch.where(inside, float("-inf"), y).unsqueeze(1)
                support = torch.cat([vals, added], dim=1)
                lse[a:b] = support.logsumexp(dim=1)
                if with_entropy:
                    entropy[a:b] = torch.special.entr((support - lse[a:b].unsqueeze(1)).exp()).sum(dim=1)
                support_vals[a:b], support_idx[a:b] = vals, idx
            else:
                lse[a:b] = z.logsumexp(dim=1)
                if with_entropy:
                    entropy[a:b] = torch.special.entr(z.sub_(lse[a:b].unsqueeze(1)).exp_()).sum(dim=1)
            label_logit[a:b] = y
            del z
        out = label_logit - lse
        ctx.mark_non_differentiable(entropy)
        ctx.params, ctx.cu_seqlens, ctx.inplace_backward = params, cu_seqlens, inplace_backward
        saved = [logits, labels, lse, label_logit]
        saved += [support_vals, support_idx] if params.masks else [penalty, pos, first]
        ctx.save_for_backward(*saved)
        return out, entropy

    @staticmethod
    def backward(ctx, grad_out, grad_entropy):
        params = ctx.params
        logits, labels, lse, label_logit, *rest = ctx.saved_tensors
        g = grad_out.float()
        # d log p(y) / d z_v = [v = y] - q_v, q the softmax over the support (penalty and mask are constants)
        label_grad = (g * (1.0 - (label_logit - lse).exp())).unsqueeze(1)
        grad = logits if ctx.inplace_backward else torch.empty_like(logits)
        if params.masks:
            support_vals, support_idx = rest
            grad.zero_()
            q = (support_vals - lse.unsqueeze(1)).exp()
            grad.scatter_(1, support_idx, (-g.unsqueeze(1) * q).to(grad.dtype))
        else:
            penalty, pos, first = rest
            for s, a, b in _chunks(ctx.cu_seqlens, logits.shape[0]):
                z = _penalised(logits, a, b, s, pos, first, penalty)
                z.sub_(lse[a:b].unsqueeze(1)).exp_().mul_(-g[a:b].unsqueeze(1))
                grad[a:b] = z.to(grad.dtype)
                del z
        grad.scatter_(1, labels.unsqueeze(1), label_grad.to(grad.dtype))
        return grad, None, None, None, None, None, None, None, None


def sampler_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    params: SamplerParams,
    cu_seqlens: list[int],
    prompt_lens: list[int],
    input_ids: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    inplace_backward: bool = True,
    with_entropy: bool = False,
):
    """Log-probs of `labels` under vLLM's sampling distribution, from packed logits.

    Args:
        logits: (rows, vocab), already divided by the temperature, as verl's log-prob path
            has them. Rows past `cu_seqlens[-1]` are padding.
        labels: (rows,) the token each row predicts (the packed input ids rolled by one).
        params: the penalty, top-k and top-p the rollout sampled with.
        cu_seqlens: sequence boundaries in the packed rows, as a list.
        prompt_lens: each sequence's prompt length; the rest of it is the response.
        input_ids: (cu_seqlens[-1],) the packed input ids.
        temperature: (rows,) the temperature each row's logits were divided by. vLLM
            subtracts the penalty before dividing, so it enters as penalty / temperature.
        inplace_backward: write the gradient into `logits`, as flash-attn's cross-entropy
            does; only when nothing else needs the logits in the backward.
        with_entropy: also return each row's entropy under the same distribution.

    Returns:
        (rows,) float32 log-probs; with `with_entropy`, (log-probs, entropies), the
        entropies without gradient.
    """
    num_rows, vocab_size = logits.shape
    first = None
    penalty = pos = torch.empty(0, device=logits.device)
    if params.presence_penalty != 0.0:
        first = first_seen(input_ids, cu_seqlens, prompt_lens, vocab_size)
        pos = response_positions(cu_seqlens, prompt_lens, num_rows, logits.device)
        penalty = torch.full((num_rows,), params.presence_penalty, dtype=torch.float32, device=logits.device)
        if temperature is not None:
            penalty = penalty / temperature.float()
    log_probs, entropy = SamplerLogProbs.apply(
        logits, labels, penalty, pos, first, cu_seqlens, params, inplace_backward, with_entropy
    )
    return (log_probs, entropy) if with_entropy else log_probs
