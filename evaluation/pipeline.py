import json
import os
import asyncio
from pathlib import Path

from datasets import Dataset
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
from langchain_huggingface import HuggingFaceEmbeddings

from rag import model

os.environ.setdefault("USER_AGENT", "main_dev_ragas_evaluation")


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
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
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