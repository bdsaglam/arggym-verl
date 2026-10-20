"""The served-policy check: vLLM serving the adapter passes, vLLM serving the base fails."""
import pytest
import torch

from arggym_verl.trainer import ServedPolicyCheck, check_served, install, served_stats


def engines(serve_adapter: bool, shift: float = 0.02, noise: float = 0.08, n=(64, 512)):
    g = torch.Generator().manual_seed(0)
    base = -torch.rand(n, generator=g) * 3
    trained = base + shift * torch.randn(n, generator=g)
    served = trained if serve_adapter else base
    rollout = served + noise * torch.randn(n, generator=g)  # BF16 vs FP32 kernel gap
    mask = torch.ones(n)
    mask[:, 400:] = 0
    return rollout, trained, base, mask


def test_adapter_served_passes_even_when_the_kernel_gap_is_larger_than_the_update():
    stats = served_stats(*engines(True), is_threshold=2.0)
    assert stats["served/z"] > 5
    check_served(stats)


def test_base_served_fails():
    stats = served_stats(*engines(False), is_threshold=2.0)
    assert stats["served/z"] < -5
    with pytest.raises(RuntimeError, match="not serving the policy"):
        check_served(stats)


def test_untrained_adapter_is_neither():
    r, _, b, m = engines(True)
    stats = served_stats(r, b, b, m, is_threshold=2.0)
    assert stats["served/z"] == 0.0 and stats["served/adapter_shift_mean"] == 0.0
    check_served(stats)


def test_gap_and_truncation_are_over_masked_tokens():
    r, t, b, m = engines(True)
    r[:, 450] = t[:, 450] - 5.0  # outside the mask: must not count
    stats = served_stats(r, t, b, m, is_threshold=2.0)
    assert stats["served/tokens"] == 64 * 400
    assert stats["served/logp_gap_max"] < 1.0
    assert stats["served/is_truncated_fraction"] == pytest.approx(
        ((t - r)[:, :400].exp() > 2.0).double().mean().item())


def test_install_wraps_the_registered_mode():
    from verl.trainer.ppo.v1 import get_trainer_cls
    from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

    cls = install("colocate_async")
    assert get_trainer_cls("colocate_async") is cls
    assert issubclass(cls, ServedPolicyCheck) and issubclass(cls, PPOTrainerColocateAsync)
    assert install("colocate_async") is cls
