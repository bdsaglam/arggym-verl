"""vLLM's server names the adapter on every request, or refuses the request.

`vLLMHttpServer.generate` used to fall back to the base model without a word
when the adapter was not loaded, and checked before a request parked behind a
weight sync's closed gate, where the old adapter is already removed.
"""
import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


class Stop(Exception):
    pass


class Engine:
    def __init__(self, loras):
        self.loras = set(loras)
        self.lora_request = "unset"

    async def list_loras(self):
        return self.loras

    def generate(self, prompt, sampling_params, request_id, lora_request, priority):
        self.lora_request = lora_request

        async def outputs():
            raise Stop
            yield

        return outputs()


def server(loras, paused=False):
    from verl.workers.rollout.vllm_rollout import vllm_async_server as v

    s = v.vLLMHttpServer.__new__(v.vLLMHttpServer)
    s._disaggregation_role = None
    s.config = OmegaConf.create({"max_model_len": 64, "response_length": 16, "prompt_length": 16,
                                 "full_determinism": False})
    s.model_config = SimpleNamespace(lora_rank=8, lora={"merge": False}, processor=None)
    s.engine = Engine(loras)
    s._submission_paused, s._rejecting, s._admitting = paused, False, 0
    s._resume_event = asyncio.Event()
    if not paused:
        s._resume_event.set()
    s.global_steps, s.replica_rank = 1, 0
    return s, v.VLLM_LORA_INT_ID


def generate(s):
    return s.generate([1, 2, 3], {"temperature": 1.0}, request_id="r0")


def test_loaded_adapter_is_named_on_the_request():
    s, lora_id = server(loras=[])
    s.engine.loras.add(lora_id)
    with pytest.raises(Stop):
        asyncio.run(generate(s))
    assert s.engine.lora_request.lora_int_id == lora_id


def test_missing_adapter_is_refused_not_served_as_the_base_model():
    s, _ = server(loras=[])
    with pytest.raises(RuntimeError, match="not loaded"):
        asyncio.run(generate(s))
    assert s.engine.lora_request == "unset"  # never reached the engine
    assert s._admitting == 0


def test_request_parked_during_a_sync_gets_the_new_adapter():
    """Arrives while the old adapter is removed and the gate is closed; admitted after the new one is added."""
    async def run():
        s, lora_id = server(loras=[], paused=True)
        task = asyncio.create_task(generate(s))
        await asyncio.sleep(0.01)
        s.engine.loras.add(lora_id)
        s._submission_paused = False
        s._resume_event.set()
        with pytest.raises(Stop):
            await task
        return s, lora_id

    s, lora_id = asyncio.run(run())
    assert s.engine.lora_request.lora_int_id == lora_id
