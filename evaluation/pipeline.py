import json
import os
import asyncio
import importlib
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
            return frame.to_dict(orient="records")
        except Exception:
            return frame.to_dict()
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {"result": str(result)}


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def run_ragas_evaluation(records: list[dict], run_dir: Path) -> dict:
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

    _write_json(run_dir / "results.json", summary)
    return summary