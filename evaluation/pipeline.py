import json
import os
import asyncio
import importlib
import math
from pathlib import Path

from datasets import Dataset

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# ── Patch RAGAS before it is used ─────────────────────────────────────────────
# When LangSmith tracing is active (or when ChatBedrockConverse doesn't fire
# LangChain callbacks), ragas.callbacks.parse_run_traces receives an empty list
# and crashes with `IndexError: list index out of range` at root_traces[0].
# We replace the function with a safe version that returns {} on that path.
try:
    import ragas.callbacks as _ragas_cb

    _orig_parse_run_traces = _ragas_cb.parse_run_traces

    def _safe_parse_run_traces(run_traces, run_id):
        try:
            return _orig_parse_run_traces(run_traces, run_id)
        except (IndexError, Exception):
            return {}

    _ragas_cb.parse_run_traces = _safe_parse_run_traces

    # Also patch dataset_schema which imports parse_run_traces by reference
    import ragas.dataset_schema as _ragas_ds
    if hasattr(_ragas_ds, "parse_run_traces"):
        _ragas_ds.parse_run_traces = _safe_parse_run_traces
except Exception:
    pass  # If RAGAS internals change, fail gracefully
# ──────────────────────────────────────────────────────────────────────────────

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings

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


def run_ragas_evaluation(records: list[dict], run_dir: Path) -> dict:
    if not records:
        raise ValueError("No interaction records available for evaluation")

    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for record in records:
        contexts = record.get("contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        # skip fallback answers that had no retrieved context
        if not contexts:
            continue
        dataset_rows.append(
            {
                "user_input": record.get("question", ""),
                "response": record.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": "\n\n".join(contexts),
            }
        )

    if not dataset_rows:
        raise ValueError("No RAG interactions with retrieved context found. Ask questions after injecting a document first.")

    _write_jsonl(run_dir / "dataset.jsonl", dataset_rows)

    dataset = Dataset.from_list(dataset_rows)

    # Nova Pro follows RAGAS structured output schemas reliably.
    evaluator_model = ChatBedrockConverse(
        model="amazon.nova-pro-v1:0",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens=1024,
        temperature=0.0,
    )
    evaluator_llm = LangchainLLMWrapper(evaluator_model)
    evaluator_embeddings = BedrockEmbeddings(
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            _load_metric_class("AnswerRelevancy")(),
            _load_metric_class("Faithfulness")(),
            _load_metric_class("ContextPrecision")(),
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