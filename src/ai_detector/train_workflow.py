from __future__ import annotations

import argparse
import inspect
import json
import math
import traceback
from pathlib import Path
from typing import Any

from ai_detector.config import load_yaml_config, set_by_path
from ai_detector.data import iter_jsonl
from ai_detector.logging_utils import (
    create_run_artifacts,
    save_metrics,
    save_run_metadata,
    tee_std_streams,
)


class ChunkDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = int(row["label"])
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a document chunk classifier.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output-root", default=None, help="Override output.root.")
    parser.add_argument("--chunk-file", default=None, help="Override data.chunk_file.")
    parser.add_argument("--experiment-name", default=None, help="Override experiment.name.")
    parser.add_argument("--model-name", default=None, help="Override model.name.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override training.train_batch_size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override training.learning_rate.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "output.root": args.output_root,
        "data.chunk_file": args.chunk_file,
        "experiment.name": args.experiment_name,
        "model.name": args.model_name,
        "training.max_steps": args.max_steps,
        "training.train_batch_size": args.train_batch_size,
        "training.learning_rate": args.learning_rate,
    }
    for dotted_path, value in overrides.items():
        if value is not None:
            set_by_path(config, dotted_path, value)
    return config


def load_split_records(
    chunk_file: str | Path,
    split_name: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(chunk_file) if row.get("split") == split_name]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def _to_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _softmax_rows(logits: Any) -> list[list[float]]:
    rows = logits.tolist() if hasattr(logits, "tolist") else logits
    probabilities: list[list[float]] = []
    for row in rows:
        max_value = max(row)
        exps = [math.exp(value - max_value) for value in row]
        total = sum(exps)
        probabilities.append([value / total for value in exps])
    return probabilities


def _binary_roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_in_bucket = sum(1 for _, label in ranked[index:end] if label == 1)
        rank_sum += positive_in_bucket * average_rank
        index = end

    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def compute_metrics(eval_prediction: tuple[Any, Any]) -> dict[str, float]:
    logits, labels = eval_prediction
    probabilities = _softmax_rows(logits)
    label_values = [int(value) for value in _to_list(labels)]
    predictions = [
        1 if row[1] >= row[0] else 0
        for row in probabilities
    ]

    tp = sum(1 for pred, gold in zip(predictions, label_values) if pred == 1 and gold == 1)
    tn = sum(1 for pred, gold in zip(predictions, label_values) if pred == 0 and gold == 0)
    fp = sum(1 for pred, gold in zip(predictions, label_values) if pred == 1 and gold == 0)
    fn = sum(1 for pred, gold in zip(predictions, label_values) if pred == 0 and gold == 1)

    total = max(1, len(label_values))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    metrics = {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    auc = _binary_roc_auc(label_values, [row[1] for row in probabilities])
    if auc is not None:
        metrics["roc_auc"] = auc
    return metrics


def build_training_arguments(checkpoint_dir: Path, config: dict[str, Any]) -> Any:
    import torch
    from transformers import TrainingArguments

    training = config["training"]
    signature = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "per_device_train_batch_size": training["train_batch_size"],
        "per_device_eval_batch_size": training["eval_batch_size"],
        "learning_rate": training["learning_rate"],
        "num_train_epochs": training["epochs"],
        "max_steps": training["max_steps"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "weight_decay": training["weight_decay"],
        "warmup_ratio": training["warmup_ratio"],
        "logging_steps": training["logging_steps"],
        "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "report_to": [],
        "seed": training["seed"],
        "remove_unused_columns": False,
        "save_safetensors": True,
    }
    eval_strategy_key = "evaluation_strategy" if "evaluation_strategy" in signature else "eval_strategy"
    kwargs[eval_strategy_key] = "steps"
    kwargs["logging_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    kwargs["eval_steps"] = training["eval_steps"]

    can_use_cuda = torch.cuda.is_available()
    bf16_enabled = bool(training.get("bf16")) and can_use_cuda
    fp16_enabled = bool(training.get("fp16")) and can_use_cuda and not bf16_enabled
    if "bf16" in signature:
        kwargs["bf16"] = bf16_enabled
    if "fp16" in signature:
        kwargs["fp16"] = fp16_enabled
    return TrainingArguments(**kwargs)


def _to_serializable(payload: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                converted[key] = None
            else:
                converted[key] = value
        else:
            converted[key] = str(value)
    return converted


def train_model(config: dict[str, Any]) -> int:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        set_seed,
    )

    output_root = Path(config["output"]["root"])
    artifacts = create_run_artifacts(output_root, config["experiment"]["name"])
    with tee_std_streams(artifacts.stdout_path, artifacts.stderr_path):
        save_run_metadata(
            artifacts,
            config,
            extra_metadata={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
        )
        set_seed(config["training"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
        train_records = load_split_records(
            config["data"]["chunk_file"],
            config["data"]["train_split"],
            config["data"].get("max_train_samples"),
        )
        validation_records = load_split_records(
            config["data"]["chunk_file"],
            config["data"]["validation_split"],
            config["data"].get("max_eval_samples"),
        )
        test_records = load_split_records(
            config["data"]["chunk_file"],
            config["data"]["test_split"],
            config["data"].get("max_eval_samples"),
        )
        if not train_records:
            raise ValueError("No training records found.")
        if not validation_records:
            raise ValueError("No validation records found.")

        train_dataset = ChunkDataset(train_records, tokenizer, config["data"]["max_length"])
        validation_dataset = ChunkDataset(
            validation_records,
            tokenizer,
            config["data"]["max_length"],
        )
        test_dataset = (
            ChunkDataset(test_records, tokenizer, config["data"]["max_length"])
            if test_records
            else None
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model"]["name"],
            num_labels=2,
            id2label={0: "human", 1: "ai"},
            label2id={"human": 0, "ai": 1},
        )
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": build_training_arguments(artifacts.checkpoint_dir, config),
            "train_dataset": train_dataset,
            "eval_dataset": validation_dataset,
            "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
            "compute_metrics": compute_metrics,
        }
        trainer_signature = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Trainer(**trainer_kwargs)

        print(f"Training records: {len(train_records)}")
        print(f"Validation records: {len(validation_records)}")
        print(f"Test records: {len(test_records)}")
        print(f"Run directory: {artifacts.run_dir}")
        print(f"Checkpoint directory: {artifacts.checkpoint_dir}")

        train_result = trainer.train()
        trainer.save_model(str(artifacts.checkpoint_dir))
        tokenizer.save_pretrained(str(artifacts.checkpoint_dir))

        metrics = {
            "train": _to_serializable(train_result.metrics),
            "validation": _to_serializable(trainer.evaluate()),
            "best_checkpoint": trainer.state.best_model_checkpoint or str(artifacts.checkpoint_dir),
        }
        if test_dataset is not None:
            metrics["test"] = _to_serializable(
                trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            )
        save_metrics(artifacts.metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        config = apply_cli_overrides(load_yaml_config(args.config), args)
        return train_model(config)
    except Exception:
        traceback.print_exc()
        return 1
