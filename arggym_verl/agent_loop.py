"""verl's single-turn agent loop with per-split sampling overrides.

verl sets one response length for training and validation and has no presence
penalty in its sampling config. The runs need both: GSM8K trains at an
8,192-token cap and evaluates dev at 32,768; Qwen3.5's dev samples with the
card's presence_penalty 1.5, its training rollouts with none. Rows name their agent loop (`agent_name`), and each name
in `config/agent_loops-*.yaml` is this class with its own `sampling`, merged over
what verl passes. Generation stays verl's token-in/token-out path:
`SingleTurnAgentLoop` sends prompt ids and keeps the ids and log-probs vLLM
returns, with no decode and re-encode.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class SamplingAgentLoop(SingleTurnAgentLoop):
    def __init__(self, *args: Any, sampling: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sampling = dict(sampling or {})
        cap = self.sampling.get("max_tokens")
        if cap is not None and cap > self.response_length:
            raise ValueError(f"max_tokens {cap} is above rollout.response_length "
                             f"{self.response_length}; the response would be cut silently")

    async def run(self, sampling_params: Dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        return await super().run({**sampling_params, **self.sampling}, **kwargs)
