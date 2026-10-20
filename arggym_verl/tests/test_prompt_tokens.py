"""verl's agent loop sends vLLM the prompt token ids TRL sends for the same row."""
import pytest

from conftest import QWEN3_06B

pytestmark = pytest.mark.skipif(not QWEN3_06B.exists(), reason="Qwen3-0.6B not cached")


def test_qwen3_prompt_ids_match_trl():
    from transformers import AutoConfig, AutoTokenizer

    from rl import gsm8k
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    tok = AutoTokenizer.from_pretrained(str(QWEN3_06B))
    kwargs = {"enable_thinking": True}
    builder = create_continuous_token_builder(
        tok, hf_model_type=AutoConfig.from_pretrained(str(QWEN3_06B)).model_type,
        chat_template_kwargs=kwargs, mm_processor_kwargs={}, processor=None)
    rows = gsm8k.trl_rows([{"id": "q", "question": "Tom has 3 apples and buys 15 more. How many?",
                            "answer": "#### 18"}])
    ours = builder.build_initial_tokens(rows[0]["prompt"])
    # TRL 1.15 (`GRPOTrainer._generate`): the chat template with the run's
    # chat_template_kwargs and the generation prompt, tokenized once.
    trl = tok.apply_chat_template(rows[0]["prompt"], add_generation_prompt=True, tokenize=True,
                                  return_dict=False, **kwargs)
    assert list(ours) == list(trl)
    assert tok.decode(ours).endswith("<|im_start|>assistant\n")
