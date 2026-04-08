from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ai_detector.data import build_token_chunks, iter_jsonl, normalize_text, write_jsonl
from ai_detector.logging_utils import create_run_artifacts, save_run_metadata, tee_std_streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run document inference.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument("--input-file", required=True, help="Input file or directory.")
    parser.add_argument("--output-file", required=True, help="Prediction JSONL output.")
    parser.add_argument("--tokenizer-name", default=None, help="Fallback tokenizer name.")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-cjk-chars", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-root", default="./artifacts", help="Run artifact root.")
    parser.add_argument("--experiment-name", default="infer", help="Inference run name.")
    return parser.parse_args()


def load_documents(input_path: str | Path) -> list[dict[str, str]]:
    path = Path(input_path)
    if path.is_dir():
        documents: list[dict[str, str]] = []
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() not in {".txt", ".json", ".jsonl"}:
                continue
            documents.extend(load_documents(child))
        return documents

    if path.suffix.lower() == ".txt":
        return [{"doc_id": path.stem, "text": normalize_text(path.read_text(encoding="utf-8"))}]
    if path.suffix.lower() == ".jsonl":
        docs = []
        for row in iter_jsonl(path):
            doc_id = str(row.get("doc_id") or row.get("id") or f"row_{len(docs):06d}")
            text = normalize_text(str(row.get("text", "")))
            if text:
                docs.append({"doc_id": doc_id, "text": text})
        return docs
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "text" in payload:
                return [
                    {
                        "doc_id": str(payload.get("doc_id") or path.stem),
                        "text": normalize_text(str(payload["text"])),
                    }
                ]
            raise ValueError(f"Unsupported JSON schema: {path}")
        if isinstance(payload, list):
            docs = []
            for index, row in enumerate(payload):
                if not isinstance(row, dict) or "text" not in row:
                    continue
                docs.append(
                    {
                        "doc_id": str(row.get("doc_id") or row.get("id") or f"{path.stem}_{index:06d}"),
                        "text": normalize_text(str(row["text"])),
                    }
                )
            return docs
    raise ValueError(f"Unsupported input path: {path}")


def _load_tokenizer(checkpoint: str | Path, fallback_name: str | None) -> Any:
    from transformers import AutoTokenizer

    checkpoint = Path(checkpoint)
    for candidate in [checkpoint, checkpoint.parent]:
        try:
            return AutoTokenizer.from_pretrained(str(candidate), use_fast=True)
        except Exception:
            continue
    if fallback_name:
        return AutoTokenizer.from_pretrained(fallback_name, use_fast=True)
    raise ValueError("Could not load tokenizer from checkpoint. Pass --tokenizer-name.")


def infer_documents(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification

    artifacts = create_run_artifacts(args.output_root, args.experiment_name)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tee_std_streams(artifacts.stdout_path, artifacts.stderr_path):
        save_run_metadata(
            artifacts,
            {
                "checkpoint": args.checkpoint,
                "input_file": args.input_file,
                "output_file": args.output_file,
                "max_length": args.max_length,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "threshold": args.threshold,
            },
            extra_metadata={"mode": "inference"},
        )
        documents = load_documents(args.input_file)
        if not documents:
            raise ValueError("No inference documents found.")

        tokenizer = _load_tokenizer(args.checkpoint, args.tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        predictions: list[dict[str, Any]] = []
        for document in tqdm(documents, desc="Inferring documents"):
            chunks = build_token_chunks(
                text=document["text"],
                tokenizer=tokenizer,
                max_length=args.max_length,
                stride=args.stride,
                min_cjk_chars=args.min_cjk_chars,
            )
            if not chunks:
                continue

            probabilities: list[float] = []
            for start in range(0, len(chunks), args.batch_size):
                batch = chunks[start : start + args.batch_size]
                encoded = tokenizer(
                    [row["text"] for row in batch],
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = model(**encoded).logits
                batch_scores = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
                probabilities.extend(float(score) for score in batch_scores)

            ai_rate = sum(probabilities) / len(probabilities)
            predictions.append(
                {
                    "doc_id": document["doc_id"],
                    "ai_rate": round(ai_rate, 6),
                    "pred_label": int(ai_rate >= args.threshold),
                    "max_chunk_score": round(max(probabilities), 6),
                    "num_chunks": len(probabilities),
                }
            )

        write_jsonl(output_file, predictions)
        print(f"Wrote predictions to {output_file}")
        print(f"Run logs saved under {artifacts.run_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return infer_documents(args)
    except Exception:
        traceback.print_exc()
        return 1
