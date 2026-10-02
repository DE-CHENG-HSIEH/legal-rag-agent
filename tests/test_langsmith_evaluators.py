"""Regression tests for the local portion of the LangSmith evaluation suite."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from evals.evaluators import (
    answer_contract_valid,
    expected_tools_called,
    forbidden_tools_absent,
    prediction_variant_matches,
    requested_count_matches,
    risk_disclaimer_present,
    source_separation_present,
    source_fidelity_valid,
    tool_call_limits_respected,
)
from evals.run_langsmith_evaluation import load_cases, select_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals/agent_cases.jsonl"


def result(*, answer="回答", tool_calls=None):
    return SimpleNamespace(outputs={"answer": answer, "tool_calls": tool_calls or []})


def example(**outputs):
    return SimpleNamespace(outputs=outputs)


class EvaluationCaseTests(unittest.TestCase):
    def test_version_controlled_cases_are_valid_and_synthetic(self) -> None:
        cases = load_cases(CASES_PATH)

        self.assertGreaterEqual(len(cases), 10)
        self.assertTrue(any("smoke" in case["splits"] for case in cases))
        self.assertTrue(all(case["metadata"] for case in cases))
        self.assertFalse(any("真實委任" in case["inputs"]["prompt"] for case in cases))

    def test_named_regression_cases_are_selected_within_a_split(self) -> None:
        cases = load_cases(CASES_PATH)

        selected = select_cases(
            cases,
            split="full",
            case_ids=["ten_similar_judgments", "three_similar_judgments"],
        )

        self.assertEqual(
            [case["case_id"] for case in selected],
            ["three_similar_judgments", "ten_similar_judgments"],
        )

    def test_named_case_must_exist_in_the_selected_split(self) -> None:
        cases = load_cases(CASES_PATH)

        with self.assertRaisesRegex(ValueError, "not in split"):
            select_cases(
                cases,
                split="smoke",
                case_ids=["ten_similar_judgments"],
            )

        with self.assertRaisesRegex(ValueError, "Unknown evaluation case_id"):
            select_cases(
                cases,
                split="full",
                case_ids=["missing_case"],
            )


class RoutingEvaluatorTests(unittest.TestCase):
    def test_expected_and_forbidden_tools_are_evaluated_separately(self) -> None:
        run = result(
            tool_calls=[
                {"name": "search_similar_judgments", "args": {"requested_count": 3}},
                {"name": "lookup_criminal_law_article", "args": {}},
            ]
        )
        reference = example(
            expected_tools=["search_similar_judgments"],
            forbidden_tools=["lookup_criminal_law_article"],
        )

        self.assertEqual(expected_tools_called(run, reference)["score"], 1)
        self.assertEqual(forbidden_tools_absent(run, reference)["score"], 0)

    def test_tool_call_limit_detects_repeated_expensive_tools(self) -> None:
        run = result(
            tool_calls=[
                {"name": "search_similar_judgments", "args": {}},
                {"name": "search_similar_judgments", "args": {}},
            ]
        )

        self.assertEqual(tool_call_limits_respected(run, example())["score"], 0)

    def test_requested_judgment_count_must_reach_tool_unchanged(self) -> None:
        reference = example(expected_requested_count=10)
        matching = result(
            tool_calls=[
                {"name": "search_similar_judgments", "args": {"requested_count": 10}}
            ]
        )
        mismatching = result(
            tool_calls=[
                {"name": "search_similar_judgments", "args": {"requested_count": 5}}
            ]
        )

        self.assertEqual(requested_count_matches(matching, reference)["score"], 1)
        self.assertEqual(requested_count_matches(mismatching, reference)["score"], 0)

    def test_prediction_variant_uses_actual_tool_arguments(self) -> None:
        variant_c = result(
            tool_calls=[
                {
                    "name": "predict_public_insult_sentence",
                    "args": {
                        "court": "臺灣臺北地方法院",
                        "judgment_year_roc": 114,
                        "judge_name": "王小明",
                    },
                }
            ]
        )
        partial_fields = result(
            tool_calls=[
                {
                    "name": "predict_public_insult_sentence",
                    "args": {"court": "臺灣臺北地方法院"},
                }
            ]
        )

        self.assertEqual(
            prediction_variant_matches(variant_c, example(expected_variant="C"))[
                "score"
            ],
            1,
        )
        self.assertEqual(
            prediction_variant_matches(partial_fields, example(expected_variant="A"))[
                "score"
            ],
            1,
        )


class AnswerEvaluatorTests(unittest.TestCase):
    def test_answer_contract_rejects_literal_escape_sequences(self) -> None:
        reference = example()

        self.assertEqual(
            answer_contract_valid(result(answer="第一段\n\n第二段"), reference)[
                "score"
            ],
            1,
        )
        self.assertEqual(
            answer_contract_valid(result(answer="第一段\\n第二段"), reference)["score"],
            0,
        )

    def test_disclaimer_and_source_separation_are_explicit(self) -> None:
        run = result(
            answer=(
                "## AI 分析\n本結果是研究模型預測，不構成法律意見。\n\n"
                "## 法律來源原文\n原文內容"
            )
        )

        self.assertEqual(
            risk_disclaimer_present(run, example(requires_disclaimer=True))["score"],
            1,
        )

        weak_disclaimer = result(answer="## 研究模型預測\n預測結果。")
        reversed_sections = result(answer="## 法律來源原文\n原文\n\n## AI 分析\n分析")
        self.assertEqual(
            risk_disclaimer_present(
                weak_disclaimer,
                example(requires_disclaimer=True),
            )["score"],
            0,
        )
        self.assertEqual(
            source_separation_present(
                reversed_sections,
                example(requires_source_separation=True),
            )["score"],
            0,
        )
        self.assertEqual(
            source_separation_present(run, example(requires_source_separation=True))[
                "score"
            ],
            1,
        )

    def test_source_fidelity_requires_deterministic_artifact_at_end(self) -> None:
        source = "## 法律來源原文\n\n### 刑事法條查詢結果\n\n法條原文"
        matching = SimpleNamespace(
            outputs={
                "answer": f"## AI 分析\n分析內容\n\n{source}",
                "source_markdown": source,
            }
        )
        modified = SimpleNamespace(
            outputs={
                "answer": "## AI 分析\n分析內容\n\n## 法律來源原文\n改寫內容",
                "source_markdown": source,
            }
        )
        reference = example(requires_source_separation=True)

        self.assertEqual(source_fidelity_valid(matching, reference)["score"], 1)
        self.assertEqual(source_fidelity_valid(modified, reference)["score"], 0)


if __name__ == "__main__":
    unittest.main()
