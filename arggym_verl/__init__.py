"""ArgGYM's GRPO runs on verl: data, reward and sampling glue around verl's own trainer.

The data selection, the prompts and the scorers are imported from the ArgGYM
checkout (`ARGGYM_ROOT`, on PYTHONPATH), so this pipeline and the TRL one see
the same rows and score the same way.
"""
