"""Synchronize synthetic cases and run the Agent evaluation in LangSmith."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.evaluators import CORE_TOOLS, EVALUATORS


DEFAULT_CASES_PATH = PROJECT_ROOT / "evals/agent_cases.jsonl"
DEFAULT_DATASET_NAME = "legal-rag-agent-synthetic-v1"
SUITE_VERSION = "1.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic LangSmith evaluations for the legal Agent.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=(
            "Run only this case within the selected split. Repeat the option to "
            "select multiple regression cases."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_ROUTING_MODEL", "gpt-5.4-nano"),
        help="Override OPENAI_ROUTING_MODEL for this evaluation run.",
    )
    parser.add_argument("--experiment-prefix", default="legal-rag-agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local cases without contacting LangSmith or model providers.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Create or update the LangSmith dataset without running the Agent.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate the version-controlled synthetic evaluation contract."""

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Evaluation case line {line_number} is not valid JSON."
                ) from error
            if not isinstance(case, dict):
                raise ValueError(
                    f"Evaluation case line {line_number} must be a JSON object."
                )

            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"Evaluation case line {line_number} has no case_id.")
            if case_id in case_ids:
                raise ValueError(f"Duplicate evaluation case_id: {case_id}")
            case_ids.add(case_id)

            inputs = case.get("inputs")
            if not isinstance(inputs, dict):
                raise ValueError(f"Evaluation case {case_id} has invalid inputs.")
            prompt = inputs.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Evaluation case {case_id} has no prompt.")

            outputs = case.get("outputs")
            if not isinstance(outputs, dict):
                raise ValueError(f"Evaluation case {case_id} has invalid outputs.")
            tool_contracts = {}
            for field in ("expected_tools", "forbidden_tools"):
                values = outputs.get(field)
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise ValueError(f"Evaluation case {case_id} has invalid {field}.")
                tool_contracts[field] = set(values)
            expected_tools = tool_contracts["expected_tools"]
            forbidden_tools = tool_contracts["forbidden_tools"]
            unknown_tools = (expected_tools | forbidden_tools) - CORE_TOOLS
            if unknown_tools:
                raise ValueError(
                    f"Evaluation case {case_id} references unknown tools: "
                    f"{sorted(unknown_tools)}"
                )
            if expected_tools & forbidden_tools:
                raise ValueError(
                    f"Evaluation case {case_id} both expects and forbids the same tool."
                )

            expected_variant = outputs.get("expected_variant")
            if expected_variant not in {None, "A", "B", "C"}:
                raise ValueError(
                    f"Evaluation case {case_id} has invalid expected_variant."
                )
            expected_count = outputs.get("expected_requested_count")
            if expected_count is not None and (
                isinstance(expected_count, bool)
                or not isinstance(expected_count, int)
                or not 1 <= expected_count <= 10
            ):
                raise ValueError(
                    f"Evaluation case {case_id} has invalid expected_requested_count."
                )
            for field in ("requires_disclaimer", "requires_source_separation"):
                if not isinstance(outputs.get(field), bool):
                    raise ValueError(f"Evaluation case {case_id} has invalid {field}.")

            metadata = case.get("metadata")
            if not isinstance(metadata, dict) or not metadata:
                raise ValueError(f"Evaluation case {case_id} has invalid metadata.")

            splits = case.get("splits")
            if (
                not isinstance(splits, list)
                or not splits
                or not all(split in {"smoke", "full"} for split in splits)
            ):
                raise ValueError(f"Evaluation case {case_id} has no supported split.")
            cases.append(case)

    if not cases:
        raise ValueError("The evaluation suite is empty.")
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    *,
    split: str,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a split or a named subset while rejecting silent empty runs."""

    split_cases = [case for case in cases if split in case["splits"]]
    requested_ids = set(case_ids or [])
    if not requested_ids:
        return split_cases

    known_ids = {case["case_id"] for case in cases}
    unknown_ids = sorted(requested_ids - known_ids)
    if unknown_ids:
        raise ValueError(f"Unknown evaluation case_id values: {unknown_ids}")

    selected = [case for case in split_cases if case["case_id"] in requested_ids]
    selected_ids = {case["case_id"] for case in selected}
    outside_split = sorted(requested_ids - selected_ids)
    if outside_split:
        raise ValueError(
            f"Evaluation cases are not in split {split!r}: {outside_split}"
        )
    return selected


def _example_id(dataset_name: str, case_id: str):
    return uuid5(NAMESPACE_URL, f"{dataset_name}:{case_id}")


def synchronize_dataset(
    client: Client,
    *,
    dataset_name: str,
    cases: list[dict[str, Any]],
):
    """Idempotently upsert versioned examples without deleting remote history."""

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name,
            description=(
                "Synthetic public-insult prompts for deterministic legal Agent "
                "routing and response-contract evaluation."
            ),
            metadata={"suite_version": SUITE_VERSION, "contains_real_cases": False},
        )

    existing_ids = {
        str(example.id)
        for example in client.list_examples(dataset_id=dataset.id, limit=None)
    }
    for case in cases:
        case_id = case["case_id"]
        example_id = _example_id(dataset_name, case_id)
        metadata = {
            **case.get("metadata", {}),
            "case_id": case_id,
            "suite_version": SUITE_VERSION,
            "synthetic": True,
        }
        payload = {
            "inputs": case["inputs"],
            "outputs": case["outputs"],
            "metadata": metadata,
            "split": case["splits"],
        }
        if str(example_id) in existing_ids:
            client.update_example(
                example_id,
                dataset_id=dataset.id,
                **payload,
            )
        else:
            client.create_example(
                dataset_id=dataset.id,
                example_id=example_id,
                **payload,
            )
    return dataset


def invoke_agent(inputs: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    """Run one isolated Agent turn and return JSON-safe evidence for evaluators."""

    from app.agent.context import LegalAgentContext
    from app.agent.legal_agent import get_legal_agent
    from app.agent.source_rendering import compose_final_answer, render_legal_sources

    prompt = str(inputs.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Evaluation input is missing prompt.")

    result = get_legal_agent(model_name=model_name).invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": str(uuid4())}},
        context=LegalAgentContext(),
    )
    answer = compose_final_answer(result)
    source_markdown = render_legal_sources(result.get("messages", []))

    tool_calls: list[dict[str, Any]] = []
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            tool_calls.append(
                {
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                }
            )

    return {
        "answer": answer,
        "source_markdown": source_markdown,
        "tool_calls": tool_calls,
        "model": model_name,
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    cases = load_cases(args.cases.resolve())
    selected_cases = select_cases(
        cases,
        split=args.split,
        case_ids=args.case_id,
    )
    selected_count = len(selected_cases)
    print(
        f"Validated {len(cases)} synthetic cases; "
        f"selected split {args.split} has {selected_count}."
    )
    if args.case_id:
        print(
            "Selected regression cases: "
            + ", ".join(case["case_id"] for case in selected_cases)
        )
    if args.dry_run:
        return

    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit(
            "LANGSMITH_API_KEY is required unless --dry-run is used. "
            "Do not upload private client matters as evaluation inputs."
        )

    client = Client()
    dataset = synchronize_dataset(
        client,
        dataset_name=args.dataset_name,
        cases=cases,
    )
    print(f"LangSmith dataset synchronized: {args.dataset_name}")
    if args.sync_only:
        return

    selected_ids = [
        _example_id(args.dataset_name, case["case_id"]) for case in selected_cases
    ]
    examples = list(
        client.list_examples(
            dataset_id=dataset.id,
            example_ids=selected_ids,
            splits=[args.split],
        )
    )
    if len(examples) != selected_count:
        raise RuntimeError(
            f"LangSmith returned {len(examples)} examples; expected {selected_count}."
        )

    # Tracing is process-local and explicit. Ordinary Streamlit runs remain opt-in.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "legal-rag-agent-evaluation")

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return invoke_agent(inputs, model_name=args.model)

    results = evaluate(
        target,
        data=examples,
        evaluators=list(EVALUATORS),
        experiment_prefix=args.experiment_prefix,
        description=(
            "Synthetic offline evaluation of tool routing, A/B/C prediction routing, "
            "source separation, response formatting and legal-risk disclosure."
        ),
        metadata={
            "suite_version": SUITE_VERSION,
            "split": args.split,
            "model": args.model,
            "contains_real_cases": False,
            "case_ids": [case["case_id"] for case in selected_cases],
        },
        max_concurrency=1,
        client=client,
        blocking=True,
    )
    print(f"LangSmith experiment completed: {results.experiment_name}")


if __name__ == "__main__":
    main()
