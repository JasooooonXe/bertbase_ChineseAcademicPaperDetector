from __future__ import annotations

import argparse
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ai_detector.data import (
    assign_splits,
    bucket_similarity,
    build_token_chunks,
    compute_similarity_ratio,
    extract_document_text,
    find_pairs,
    read_json,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired AI rewrite data.")
    parser.add_argument("--input-dir", required=True, help="Raw data root.")
    parser.add_argument("--output-dir", required=True, help="Processed data root.")
    parser.add_argument(
        "--original-subdir",
        default="json",
        help="Relative path to original JSON files.",
    )
    parser.add_argument(
        "--rewrite-subdir",
        default="rewrite/outputs/per_paper",
        help="Relative path to rewritten JSON files.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="hfl/chinese-roberta-wwm-ext",
        help="Tokenizer used for chunk generation.",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-cjk-chars", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=[2024, 2025],
        help="Years reserved for the test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def _build_document_record(
    pair_id: str,
    doc_id: str,
    label: int,
    source: str,
    forum: str,
    year: int,
    text: str,
    similarity: float,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "doc_id": doc_id,
        "label": label,
        "source": source,
        "forum": forum,
        "year": year,
        "text": text,
        "text_length": len(text),
        "cjk_length": sum("\u4e00" <= char <= "\u9fff" for char in text),
        "similarity": similarity,
        "similarity_bucket": bucket_similarity(similarity),
    }


def run_prepare(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, missing_ids = find_pairs(
        input_root=args.input_dir,
        original_subdir=args.original_subdir,
        rewrite_subdir=args.rewrite_subdir,
    )
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError("No paired samples found.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    documents: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []

    for pair in tqdm(pairs, desc="Preparing pairs"):
        original_payload = read_json(pair.original_path)
        rewrite_payload = read_json(pair.rewrite_path)
        original_text = extract_document_text(original_payload)
        rewrite_text = extract_document_text(rewrite_payload)

        if not original_text or not rewrite_text:
            skipped_pairs.append(
                {
                    "pair_id": pair.pair_id,
                    "reason": "empty_text_after_cleaning",
                    "original_path": str(pair.original_path),
                    "rewrite_path": str(pair.rewrite_path),
                }
            )
            continue

        similarity = compute_similarity_ratio(original_text, rewrite_text)
        pair_rows.append(
            {
                "pair_id": pair.pair_id,
                "forum": pair.forum,
                "year": pair.year,
                "similarity": similarity,
                "similarity_bucket": bucket_similarity(similarity),
            }
        )
        documents.append(
            _build_document_record(
                pair_id=pair.pair_id,
                doc_id=f"{pair.pair_id}_human",
                label=0,
                source="human",
                forum=pair.forum,
                year=pair.year,
                text=original_text,
                similarity=similarity,
            )
        )
        documents.append(
            _build_document_record(
                pair_id=pair.pair_id,
                doc_id=f"{pair.pair_id}_ai",
                label=1,
                source="ai",
                forum=pair.forum,
                year=pair.year,
                text=rewrite_text,
                similarity=similarity,
            )
        )

    split_map = assign_splits(
        pairs=pair_rows,
        val_ratio=args.val_ratio,
        test_years=set(args.test_years),
        seed=args.seed,
    )

    chunks: list[dict[str, Any]] = []
    for document in tqdm(documents, desc="Chunking documents"):
        split = split_map.get(document["pair_id"], "train")
        document["split"] = split
        token_chunks = build_token_chunks(
            text=document["text"],
            tokenizer=tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            min_cjk_chars=args.min_cjk_chars,
        )
        for chunk_index, chunk in enumerate(token_chunks):
            chunks.append(
                {
                    "pair_id": document["pair_id"],
                    "doc_id": document["doc_id"],
                    "chunk_id": f"{document['doc_id']}__{chunk_index:04d}",
                    "label": document["label"],
                    "source": document["source"],
                    "forum": document["forum"],
                    "year": document["year"],
                    "split": split,
                    "similarity": document["similarity"],
                    "similarity_bucket": document["similarity_bucket"],
                    "text": chunk["text"],
                    "char_count": chunk["char_count"],
                    "token_count": chunk["token_count"],
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                }
            )

    split_counter = Counter(row["split"] for row in documents)
    label_counter = Counter(row["label"] for row in documents)
    chunk_split_counter = Counter(row["split"] for row in chunks)
    stats = {
        "paired_ids": len(pair_rows),
        "documents": len(documents),
        "chunks": len(chunks),
        "missing_rewrites": len(missing_ids),
        "skipped_pairs": len(skipped_pairs),
        "split_counts": dict(split_counter),
        "label_counts": dict(label_counter),
        "chunk_split_counts": dict(chunk_split_counter),
        "tokenizer_name": args.tokenizer_name,
        "max_length": args.max_length,
        "stride": args.stride,
        "min_cjk_chars": args.min_cjk_chars,
    }

    pair_table = []
    for row in pair_rows:
        pair_row = dict(row)
        pair_row["split"] = split_map.get(row["pair_id"], "train")
        pair_table.append(pair_row)

    write_jsonl(output_dir / "pairs.jsonl", pair_table)
    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "missing_rewrites.json", missing_ids)
    write_json(output_dir / "skipped_pairs.json", skipped_pairs)

    print(f"Wrote paired table: {output_dir / 'pairs.jsonl'}")
    print(f"Wrote document table: {output_dir / 'documents.jsonl'}")
    print(f"Wrote chunk table: {output_dir / 'chunks.jsonl'}")
    print(f"Wrote stats: {output_dir / 'stats.json'}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_prepare(args)
    except Exception:
        traceback.print_exc()
        return 1
