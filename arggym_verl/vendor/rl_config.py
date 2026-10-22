# Vendored from ArgGYM rl/config.py (see arggym_verl/vendor/__init__.py; MIT, ./LICENSE).
# Kept: the loader and overrides. Removed: run names, eval-model and vLLM helpers.
"""One YAML file holds every setting of a run; `key.sub=value` arguments override it.

An override is parsed as YAML, so `2e-5` is a float, `[1,2]` a list and `true` a
bool. A key the file does not already have is refused: a typo would otherwise
run with the default it meant to change.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Sequence

import yaml


class _Loader(yaml.SafeLoader):
    """YAML 1.1 reads `1e-5` as a string, because it wants a dot in the mantissa."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"))


def _parse(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass


def load(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    with open(path) as f:
        cfg = _parse(f.read())
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} does not hold a mapping")
    return apply(cfg, overrides)


def apply(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"override {item!r} is not key=value")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise SystemExit(f"override {key!r}: {p!r} is not a section of the config")
            node = node[p]
        if leaf not in node:
            raise SystemExit(f"override {key!r}: no such key; known here: {sorted(node)}")
        node[leaf] = _parse(raw)
    return cfg
