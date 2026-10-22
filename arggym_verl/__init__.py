"""ArgGYM's GRPO runs on verl: data, reward and sampling glue around verl's own trainer.

The task generators and the grader are ArgGYM's library, `arggym`; the data
selection, prompts, scoring and curriculum code of ArgGYM's training pipeline is
vendored in `arggym_verl.vendor`, so both pipelines see the same rows and score
the same way.
"""
