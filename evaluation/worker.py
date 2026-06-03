import json
import sys
from pathlib import Path

from evaluation.pipeline import _run_ragas_evaluation_local


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m evaluation.worker <input_json> <run_dir>")

    input_path = Path(sys.argv[1])
    run_dir = Path(sys.argv[2])

    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("records") or []
    summary = _run_ragas_evaluation_local(records, run_dir)

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()