from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ACK_PATTERNS = [
    re.compile(r"^\s*致谢\s*$"),
    re.compile(r"^\s*参考文献\s*$"),
    re.compile(r"^\s*附录\s*$"),
    re.compile(r"^\s*(references|acknowledg(e)?ments?)\s*$", re.IGNORECASE),
]
_ENGLISH_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-\.,:;()]{5,}$")
_REFERENCE_LINE_RE = re.compile(r"^\s*(\[\d+\]|\d+\.)")


@dataclass(frozen=True)
class PairPaths:
    pair_id: str
    original_path: Path
    rewrite_path: Path
    forum: str
    year: int


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload: {path}")
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not an object: {path}")
            yield payload


def count_cjk(text: str) -> int:
    return len(_CJK_RE.findall(text))


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_drop_paragraph(text: str) -> bool:
    candidate = normalize_text(text)
    if not candidate:
        return True
    if len(candidate) < 2:
        return True
    if any(pattern.match(candidate) for pattern in _ACK_PATTERNS):
        return True
    if _ENGLISH_HEADING_RE.match(candidate) and count_cjk(candidate) == 0:
        return True
    if _REFERENCE_LINE_RE.match(candidate) and len(candidate) < 300:
        return True
    cjk_count = count_cjk(candidate)
    ascii_letters = sum(1 for char in candidate if char.isascii() and char.isalpha())
    if cjk_count < 10 and ascii_letters > 60:
        return True
    return False


def extract_document_text(payload: dict[str, Any]) -> str:
    fulltext = payload.get("fulltext", [])
    if not isinstance(fulltext, list):
        return ""
    paragraphs: list[str] = []
    for block in fulltext:
        if not isinstance(block, dict):
            continue
        raw_paragraphs = block.get("paragraphs", [])
        if not isinstance(raw_paragraphs, list):
            continue
        for paragraph in raw_paragraphs:
            if not isinstance(paragraph, str):
                continue
            candidate = normalize_text(paragraph)
            if should_drop_paragraph(candidate):
                continue
            paragraphs.append(candidate)
    return "\n\n".join(paragraphs).strip()


def compute_similarity_ratio(text_a: str, text_b: str, max_chars: int = 10000) -> float:
    left = text_a[:max_chars]
    right = text_b[:max_chars]
    return round(SequenceMatcher(None, left, right).ratio(), 4)


def find_pairs(
    input_root: str | Path,
    original_subdir: str = "json",
    rewrite_subdir: str = "rewrite/outputs/per_paper",
) -> tuple[list[PairPaths], list[str]]:
    root = Path(input_root)
    original_root = root / original_subdir
    rewrite_root = root / rewrite_subdir
    original_map = {path.stem: path for path in original_root.rglob("*.json")}
    rewrite_map = {path.stem: path for path in rewrite_root.rglob("*.json")}
    shared_ids = sorted(original_map.keys() & rewrite_map.keys())
    missing_ids = sorted(original_map.keys() - rewrite_map.keys())

    pairs: list[PairPaths] = []
    for pair_id in shared_ids:
        original_path = original_map[pair_id]
        rewrite_path = rewrite_map[pair_id]
        relative = original_path.relative_to(original_root)
        forum = relative.parts[0] if len(relative.parts) >= 1 else "unknown"
        year_raw = relative.parts[1] if len(relative.parts) >= 2 else "0"
        try:
            year = int(year_raw)
        except ValueError:
            year = 0
        pairs.append(
            PairPaths(
                pair_id=pair_id,
                original_path=original_path,
                rewrite_path=rewrite_path,
                forum=forum,
                year=year,
            )
        )
    return pairs, missing_ids


def bucket_similarity(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.5:
        return "medium"
    return "high"


def build_token_chunks(
    text: str,
    tokenizer: Any,
    max_length: int,
    stride: int,
    min_cjk_chars: int,
) -> list[dict[str, Any]]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if stride >= max_length:
        raise ValueError("stride must be smaller than max_length")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = encoded.get("input_ids", [])
    offsets = encoded.get("offset_mapping", [])
    if not input_ids:
        return []

    step = max_length - stride
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(input_ids), step):
        end = min(start + max_length, len(input_ids))
        if end <= start:
            break
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = normalize_text(text[char_start:char_end])
        if count_cjk(chunk_text) < min_cjk_chars:
            if end == len(input_ids):
                break
            continue
        chunks.append(
            {
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "char_count": len(chunk_text),
            }
        )
        if end == len(input_ids):
            break
    return chunks


def assign_splits(
    pairs: list[dict[str, Any]],
    val_ratio: float,
    test_years: set[int],
    seed: int,
) -> dict[str, str]:
    train_val_pairs = [row for row in pairs if row["year"] not in test_years]
    test_pairs = [row for row in pairs if row["year"] in test_years]
    split_map = {row["pair_id"]: "test" for row in test_pairs}

    if not train_val_pairs:
        return split_map
    if len(train_val_pairs) == 1:
        split_map[train_val_pairs[0]["pair_id"]] = "train"
        return split_map

    rng = random.Random(seed)
    grouped: dict[str, list[str]] = {}
    for row in train_val_pairs:
        group_key = f"{row['forum']}::{row['year']}"
        grouped.setdefault(group_key, []).append(row["pair_id"])
    for pair_ids in grouped.values():
        rng.shuffle(pair_ids)

    target_validation = max(1, round(len(train_val_pairs) * val_ratio))
    allocations: dict[str, int] = {}
    for group_key, pair_ids in grouped.items():
        if len(pair_ids) <= 1:
            allocations[group_key] = 0
            continue
        proposed = round(len(pair_ids) * val_ratio)
        allocations[group_key] = min(max(proposed, 0), len(pair_ids) - 1)

    def total_allocated() -> int:
        return sum(allocations.values())

    while total_allocated() < target_validation:
        candidates = [
            key
            for key, pair_ids in grouped.items()
            if len(pair_ids) - allocations[key] > 1
        ]
        if not candidates:
            break
        best_key = max(candidates, key=lambda item: len(grouped[item]) - allocations[item])
        allocations[best_key] += 1

    while total_allocated() > target_validation:
        candidates = [key for key, value in allocations.items() if value > 0]
        if not candidates:
            break
        best_key = max(candidates, key=lambda item: allocations[item])
        allocations[best_key] -= 1

    validation_ids: list[str] = []
    train_ids: list[str] = []
    for group_key, pair_ids in grouped.items():
        cut = allocations[group_key]
        validation_ids.extend(pair_ids[:cut])
        train_ids.extend(pair_ids[cut:])

    if not validation_ids:
        candidate_group = max(grouped.items(), key=lambda item: len(item[1]))[0]
        if len(grouped[candidate_group]) > 1:
            validation_ids.append(train_ids.pop())
    for pair_id in train_ids:
        split_map[pair_id] = "train"
    for pair_id in validation_ids:
        split_map[pair_id] = "validation"
    return split_map
