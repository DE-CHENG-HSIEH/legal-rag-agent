"""Deterministic LangSmith evaluators for Agent routing and response contracts."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.inference.public_insult_predictor import select_input_variant


CORE_TOOLS = frozenset(
    {
        "search_similar_judgments",
        "search_public_insult_authorities",
        "lookup_criminal_law_article",
        "predict_public_insult_sentence",
    }
)
LIMITED_TOOLS = frozenset(
    {
        "search_similar_judgments",
        "search_public_insult_authorities",
        "predict_public_insult_sentence",
    }
)


def _outputs(value: Any) -> dict[str, Any]:
    outputs = getattr(value, "outputs", None)
    return outputs if isinstance(outputs, dict) else {}


def _tool_calls(run: Any) -> list[dict[str, Any]]:
    calls = _outputs(run).get("tool_calls", [])
    return [call for call in calls if isinstance(call, dict)]


def _expected(example: Any) -> dict[str, Any]:
    return _outputs(example)


def _score(key: str, passed: bool, comment: str) -> dict[str, Any]:
    return {"key": key, "score": int(passed), "comment": comment}


def _not_applicable(key: str, comment: str) -> dict[str, Any]:
    return {"key": key, "score": None, "comment": comment}


def expected_tools_called(run: Any, example: Any) -> dict[str, Any]:
    """Check that every tool required by the curated case was invoked."""

    expected = set(_expected(example).get("expected_tools", []))
    actual = {call.get("name") for call in _tool_calls(run)}
    missing = sorted(expected - actual)
    return _score(
        "expected_tools_called",
        not missing,
        "all expected tools were called" if not missing else f"missing: {missing}",
    )


def forbidden_tools_absent(run: Any, example: Any) -> dict[str, Any]:
    """Detect unnecessary tool use on direct or narrowly scoped requests."""

    forbidden = set(_expected(example).get("forbidden_tools", []))
    actual = {call.get("name") for call in _tool_calls(run)}
    unexpected = sorted(forbidden & actual)
    return _score(
        "forbidden_tools_absent",
        not unexpected,
        (
            "no forbidden tools were called"
            if not unexpected
            else f"unexpected: {unexpected}"
        ),
    )


def tool_call_limits_respected(run: Any, example: Any) -> dict[str, Any]:
    """Mirror the production middleware invariant for expensive tools."""

    del example
    counts = Counter(call.get("name") for call in _tool_calls(run))
    repeated = {name: counts[name] for name in LIMITED_TOOLS if counts[name] > 1}
    return _score(
        "tool_call_limits_respected",
        not repeated,
        (
            "limited tools were called at most once"
            if not repeated
            else f"repeated calls: {repeated}"
        ),
    )


def requested_count_matches(run: Any, example: Any) -> dict[str, Any]:
    """Verify that explicit judgment counts reach the search tool unchanged."""

    expected_count = _expected(example).get("expected_requested_count")
    if expected_count is None:
        return _not_applicable(
            "requested_count_matches",
            "case does not specify a judgment count",
        )

    search_calls = [
        call
        for call in _tool_calls(run)
        if call.get("name") == "search_similar_judgments"
    ]
    actual_count = None
    if search_calls:
        arguments = search_calls[0].get("args", {})
        if isinstance(arguments, dict):
            actual_count = arguments.get("requested_count")
    return _score(
        "requested_count_matches",
        actual_count == expected_count,
        f"expected {expected_count}; actual {actual_count}",
    )


def prediction_variant_matches(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate A/B/C routing from the arguments the Agent actually supplied."""

    expected_variant = _expected(example).get("expected_variant")
    if expected_variant is None:
        return _not_applicable(
            "prediction_variant_matches",
            "case does not require a prediction",
        )

    prediction_calls = [
        call
        for call in _tool_calls(run)
        if call.get("name") == "predict_public_insult_sentence"
    ]
    if not prediction_calls:
        return _score(
            "prediction_variant_matches",
            False,
            "prediction tool was not called",
        )

    arguments = prediction_calls[0].get("args", {})
    if not isinstance(arguments, dict):
        arguments = {}
    actual_variant = select_input_variant(
        court=arguments.get("court"),
        judgment_year_roc=arguments.get("judgment_year_roc"),
        judge_name=arguments.get("judge_name"),
    )
    return _score(
        "prediction_variant_matches",
        actual_variant == expected_variant,
        f"expected {expected_variant}; actual {actual_variant}",
    )


def answer_contract_valid(run: Any, example: Any) -> dict[str, Any]:
    """Require a readable final answer without escaped layout artifacts."""

    del example
    answer = _outputs(run).get("answer", "")
    valid = (
        isinstance(answer, str)
        and bool(answer.strip())
        and "\\n" not in answer
        and "\\t" not in answer
    )
    return _score(
        "answer_contract_valid",
        valid,
        (
            "answer is non-empty and uses real Markdown line breaks"
            if valid
            else "answer is empty or contains literal escaped layout characters"
        ),
    )


def risk_disclaimer_present(run: Any, example: Any) -> dict[str, Any]:
    """Require a research-use limitation when the Agent predicts a sentence."""

    if not _expected(example).get("requires_disclaimer", False):
        return _not_applicable(
            "risk_disclaimer_present",
            "case does not require a prediction disclaimer",
        )

    answer = str(_outputs(run).get("answer", ""))
    markers = (
        "不構成法律意見",
        "不是法院判決",
        "不代表法院",
        "不構成判決保證",
    )
    present = any(marker in answer for marker in markers)
    return _score(
        "risk_disclaimer_present",
        present,
        (
            "research-use limitation found"
            if present
            else "prediction answer omitted the research-use limitation"
        ),
    )


def source_separation_present(run: Any, example: Any) -> dict[str, Any]:
    """Check that retrieved legal text is visually separated from AI analysis."""

    if not _expected(example).get("requires_source_separation", False):
        return _not_applicable(
            "source_separation_present",
            "case does not retrieve legal source text",
        )

    answer = str(_outputs(run).get("answer", ""))
    source_heading = "## 法律來源原文"
    analysis_headings = ("## AI 分析", "## AI 比較")
    source_position = answer.find(source_heading)
    analysis_positions = [
        position
        for heading in analysis_headings
        if (position := answer.find(heading)) >= 0
    ]
    correctly_ordered = bool(analysis_positions) and (
        0 <= min(analysis_positions) < source_position
    )
    return _score(
        "source_separation_present",
        correctly_ordered,
        (
            "fixed analysis and source headings are present in the required order"
            if correctly_ordered
            else "missing fixed headings or source text appears before AI analysis"
        ),
    )


EVALUATORS = (
    expected_tools_called,
    forbidden_tools_absent,
    tool_call_limits_respected,
    requested_count_matches,
    prediction_variant_matches,
    answer_contract_valid,
    risk_disclaimer_present,
    source_separation_present,
)
