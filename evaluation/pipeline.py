import json
import os
import asyncio
import importlib
import math
import subprocess
import sys
from pathlib import Path

from datasets import Dataset
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from rag import model

os.environ.setdefault("USER_AGENT", "main_dev_ragas_evaluation")


def _load_metric_class(metric_name: str):
    candidate_modules = [
        "ragas.metrics",
        "ragas.metrics.collections",
    ]

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        metric_class = getattr(module, metric_name, None)
        if metric_class is not None:
            return metric_class

    raise ImportError(f"Could not load RAGAS metric class: {metric_name}")


def _serialize_result(result):
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        try:
            return _make_json_safe(frame.to_dict(orient="records"))
        except Exception:
            return _make_json_safe(frame.to_dict())
    if hasattr(result, "to_dict"):
        return _make_json_safe(result.to_dict())
    return {"result": str(result)}


def _make_json_safe(value):
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _run_ragas_evaluation_local(records: list[dict], run_dir: Path) -> dict:
    if not records:
        raise ValueError("No interaction records available for evaluation")

    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for record in records:
        contexts = record.get("contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        dataset_rows.append(
            {
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "contexts": contexts,
                "reference": "\n\n".join(contexts),
            }
        )

    _write_jsonl(run_dir / "dataset.jsonl", dataset_rows)

    dataset = Dataset.from_list(dataset_rows)
    evaluator_llm = LangchainLLMWrapper(model)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        cache_folder=None,
        encode_kwargs={"normalize_embeddings": True},
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            _load_metric_class("AnswerRelevancy")(llm=evaluator_llm, embeddings=evaluator_embeddings),
            _load_metric_class("Faithfulness")(llm=evaluator_llm),
            _load_metric_class("ContextPrecision")(llm=evaluator_llm),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=False,
        raise_exceptions=False,
    )

    summary = {
        "sample_count": len(dataset_rows),
        "metrics": _serialize_result(result),
    }

    summary = _make_json_safe(summary)

    _write_json(run_dir / "results.json", summary)
    return summary


def run_ragas_evaluation(records: list[dict], run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.json"
    _write_json(input_path, {"records": records})

    command = [
        sys.executable,
        "-m",
        "evaluation.worker",
        str(input_path),
        str(run_dir),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"RAGAS worker failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("RAGAS worker did not produce summary.json")

    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)