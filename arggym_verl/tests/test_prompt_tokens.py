"""verl's agent loop sends vLLM the prompt token ids TRL sends for the same row."""
import pytest

from conftest import QWEN3_06B, QWEN35_4B


@pytest.mark.skipif(not QWEN3_06B.exists(), reason="Qwen3-0.6B not cached")
def test_qwen3_prompt_ids_match_trl():
    from transformers import AutoConfig, AutoTokenizer

    from arggym_verl.vendor import rl_gsm8k as gsm8k
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


def test_qwen35_prompt_ids_match_trl_on_the_dev_rows():
    """Qwen3.5-4B is a vision-language checkpoint, so verl builds prompts through its
    processor (HFModelConfig loads one); TRL tokenizes with the chat template."""
    import pandas as pd
    from transformers import AutoConfig, AutoTokenizer

    from conftest import ROOT
    from verl.utils import hf_processor
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    tok = AutoTokenizer.from_pretrained(str(QWEN35_4B))
    processor = hf_processor(str(QWEN35_4B))
    assert processor is not None
    kwargs = {"enable_thinking": True}
    builder = create_continuous_token_builder(
        tok, hf_model_type=AutoConfig.from_pretrained(str(QWEN35_4B)).model_type,
        chat_template_kwargs=kwargs, mm_processor_kwargs={}, processor=processor)
    dev = pd.read_parquet(ROOT / "data/arggym-qwen3.5-4b/dev.parquet")
    for prompt in dev["prompt"]:
        messages = [dict(m) for m in prompt]
        trl = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, **kwargs)
        assert list(builder.build_initial_tokens(messages)) == list(trl)
    assert tok.decode(trl).endswith("<|im_start|>assistant\n<think>\n")
