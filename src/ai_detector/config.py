from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = config
    for part in parts[:-1]:
        node = cursor.get(part)
        if node is None:
            node = {}
            cursor[part] = node
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override non-mapping key: {dotted_path}")
        cursor = node
    cursor[parts[-1]] = value


def dump_yaml_config(path: str | Path, config: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
