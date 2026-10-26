"""verl's `main_ppo` entry point, with the trainer mixins of `arggym_verl.trainer` installed.

    python -m arggym_verl.main --config-path=... --config-name=... [overrides]

Identical to `python -m verl.trainer.main_ppo` except that the Ray task runner,
before it builds the trainer, wraps the configured V1 trainer mode with
`arggym_verl.trainer`'s `ServedPolicyCheck`, `CurriculumFeedback`, `RewardMetrics`,
`AutoStop` and `DevSelection`.
"""
from __future__ import annotations

import hydra
import ray

from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@ray.remote
class CheckedTaskRunner(TaskRunnerV1.__ray_actor_class__):
    def run(self, config):
        from arggym_verl.trainer import install

        install(config.trainer.v1.trainer_mode)
        return super().run(config)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config),
                    use_critic=need_critic(config))
    if not config.trainer.use_v1:
        raise SystemExit("arggym_verl runs verl's V1 trainer (trainer.use_v1=true)")
    run_ppo(config, task_runner_class=CheckedTaskRunner)


if __name__ == "__main__":
    main()
