"""SFT 離線評估回歸測試；只使用合成預測，不載入模型或下載指標權重。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SFT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SFT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from public_insult_sft.evaluation import (  # noqa: E402
    BAND_SCHEMA,
    MISSING_LABEL,
    evaluate_prediction_records,
    parse_model_output,
    rouge_l,
    schema_errors,
)
from public_insult_sft.inference import (  # noqa: E402
    load_completed_case_ids,
    make_prediction_record,
    prepare_inference_cases,
)


EVALUATION_SCRIPT = SFT_ROOT / "scripts/05_evaluate_outputs.py"
SPEC = importlib.util.spec_from_file_location(
    "sft_output_evaluation_under_test", EVALUATION_SCRIPT
)
evaluation_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation_script
SPEC.loader.exec_module(evaluation_script)

INFERENCE_SCRIPT = SFT_ROOT / "scripts/04_run_inference.py"
INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "sft_inference_under_test", INFERENCE_SCRIPT
)
inference_script = importlib.util.module_from_spec(INFERENCE_SPEC)
sys.modules[INFERENCE_SPEC.name] = inference_script
INFERENCE_SPEC.loader.exec_module(inference_script)


def payload(sentence_type="罰金", amount=3000, unit="新臺幣元", reason="量刑理由"):
    return {
        "prediction": {
            "sentence_type": sentence_type,
            "sentence_amount_raw": "參仟元" if unit == "新臺幣元" else "拾日",
            "sentence_amount_numeric": amount,
            "sentence_amount_unit": unit,
        },
        "sentencing_reason": reason,
    }


def band_payload(band="fine_1000_3000", reason="量刑理由"):
    definitions = {
        "fine_1000_3000": ("罰金", 1000, 3000, "新臺幣元"),
        "fine_4000_5000": ("罰金", 4000, 5000, "新臺幣元"),
        "fine_6000_plus": ("罰金", 6000, None, "新臺幣元"),
        "detention_1_15": ("拘役", 1, 15, "日"),
        "detention_16_20": ("拘役", 16, 20, "日"),
        "detention_21_plus": ("拘役", 21, None, "日"),
    }
    sentence_type, minimum, maximum, unit = definitions[band]
    return {
        "prediction": {
            "sentence_type": sentence_type,
            "sentence_amount_band": band,
            "sentence_amount_min": minimum,
            "sentence_amount_max": maximum,
            "sentence_amount_unit": unit,
        },
        "sentencing_reason": reason,
    }


def prediction_record(case_id, reference, generated):
    return {
        "case_id": case_id,
        "reference": reference,
        "generated_text": generated,
        "generation_error": None,
        "input_messages": [{"role": "user", "content": f"案情 {case_id}"}],
    }


class JsonParsingTests(unittest.TestCase):
    def test_strict_json_object_is_accepted(self):
        parsed = parse_model_output(json.dumps(payload(), ensure_ascii=False))
        self.assertTrue(parsed.strict_json_valid)
        self.assertEqual(parsed.payload["prediction"]["sentence_type"], "罰金")
        self.assertIsNone(parsed.error)

    def test_markdown_json_is_recovered_but_not_strict(self):
        generated = (
            "以下為結果：\n```json\n"
            + json.dumps(payload(), ensure_ascii=False)
            + "\n```"
        )
        parsed = parse_model_output(generated)
        self.assertFalse(parsed.strict_json_valid)
        self.assertIsNotNone(parsed.payload)
        self.assertEqual(parsed.error, "json_recovered_from_markdown_fence")

    def test_schema_requires_complete_sentence_and_reason(self):
        incomplete = {"prediction": {"sentence_type": "罰金"}, "sentencing_reason": ""}
        errors = schema_errors(incomplete)
        self.assertIn("prediction.sentence_amount_numeric_invalid", errors)
        self.assertIn("sentencing_reason_missing", errors)

    def test_band_schema_accepts_a_consistent_open_ended_range(self):
        result = band_payload("fine_6000_plus")
        self.assertEqual(schema_errors(result), [])

    def test_band_schema_rejects_a_sentence_type_mismatch(self):
        result = band_payload("detention_1_15")
        result["prediction"]["sentence_type"] = "罰金"
        self.assertIn(
            "prediction.sentence_amount_band_type_mismatch", schema_errors(result)
        )


class MetricTests(unittest.TestCase):
    def test_all_cases_remain_in_metric_denominator(self):
        exact = payload(reason="審酌犯後坦承犯行")
        wrong_class = payload(
            sentence_type="拘役", amount=10, unit="日", reason="審酌犯罪情節"
        )
        records = [
            prediction_record("A", exact, json.dumps(exact, ensure_ascii=False)),
            prediction_record("B", exact, json.dumps(wrong_class, ensure_ascii=False)),
            prediction_record("C", wrong_class, "not json"),
        ]

        metrics, cases = evaluate_prediction_records(records)

        self.assertEqual(metrics["dataset"]["case_count"], 3)
        self.assertAlmostEqual(metrics["format"]["strict_json_valid_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["classification"]["accuracy"], 1 / 3)
        self.assertEqual(metrics["classification"]["missing_prediction_count"], 1)
        self.assertEqual(cases[2]["predicted_sentence_type"], MISSING_LABEL)
        self.assertEqual(cases[2]["rouge_l_f1"], 0.0)
        self.assertAlmostEqual(metrics["end_to_end"]["joint_success_rate"], 1 / 3)

    def test_exact_output_has_perfect_end_to_end_metrics(self):
        reference = payload(reason="被告坦承犯行，兼衡生活狀況。")
        metrics, cases = evaluate_prediction_records(
            [
                prediction_record(
                    "A", reference, json.dumps(reference, ensure_ascii=False)
                ),
            ]
        )
        self.assertEqual(metrics["classification"]["accuracy"], 1.0)
        self.assertEqual(metrics["classification"]["macro_f1"], 1.0)
        self.assertEqual(metrics["sentence_amount"]["exact_match_rate"], 1.0)
        self.assertEqual(metrics["sentencing_reason"]["rouge_l_f1"], 1.0)
        self.assertTrue(cases[0]["full_prediction_exact"])

    def test_duplicate_case_ids_are_rejected(self):
        reference = payload()
        generated = json.dumps(reference, ensure_ascii=False)
        duplicate = prediction_record("same", reference, generated)
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            evaluate_prediction_records([duplicate, duplicate])

    def test_band_metrics_report_exact_and_ordinal_errors(self):
        records = [
            prediction_record(
                "fine-low",
                band_payload("fine_1000_3000"),
                json.dumps(band_payload("fine_1000_3000"), ensure_ascii=False),
            ),
            prediction_record(
                "fine-middle",
                band_payload("fine_4000_5000"),
                json.dumps(band_payload("fine_6000_plus"), ensure_ascii=False),
            ),
            prediction_record(
                "detention-low",
                band_payload("detention_1_15"),
                json.dumps(band_payload("detention_21_plus"), ensure_ascii=False),
            ),
        ]

        metrics, cases = evaluate_prediction_records(records)

        self.assertEqual(metrics["dataset"]["reference_schema"], BAND_SCHEMA)
        self.assertEqual(metrics["sentence_amount"]["mode"], "band")
        self.assertAlmostEqual(metrics["sentence_amount"]["exact_match_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics["sentence_amount"]["within_one_band_rate"], 2 / 3
        )
        self.assertAlmostEqual(metrics["sentence_amount"]["two_band_error_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics["sentence_amount"]["classification"]["accuracy"], 1 / 3
        )
        self.assertIn("amount_band_confusion_matrix", metrics)
        self.assertTrue(cases[0]["full_prediction_exact"])
        self.assertFalse(cases[2]["full_prediction_exact"])

    def test_mixed_reference_schemas_are_rejected(self):
        exact = payload()
        band = band_payload()
        with self.assertRaisesRegex(ValueError, "mixes exact-amount and amount-band"):
            evaluate_prediction_records(
                [
                    prediction_record(
                        "exact", exact, json.dumps(exact, ensure_ascii=False)
                    ),
                    prediction_record(
                        "band", band, json.dumps(band, ensure_ascii=False)
                    ),
                ]
            )

    def test_chinese_rouge_ignores_layout_whitespace(self):
        score = rouge_l("審酌 被告 坦承", "審酌被告坦承")
        self.assertEqual(score["f1"], 1.0)


class ReportUtilityTests(unittest.TestCase):
    def test_manual_review_sample_balances_reference_classes(self):
        cases = [
            {"case_id": f"fine-{index}", "actual_sentence_type": "罰金"}
            for index in range(8)
        ] + [
            {"case_id": f"detention-{index}", "actual_sentence_type": "拘役"}
            for index in range(2)
        ]
        sample = evaluation_script.balanced_review_sample(
            cases, requested_size=4, seed=42
        )
        labels = [item["actual_sentence_type"] for item in sample]
        self.assertEqual(labels.count("罰金"), 2)
        self.assertEqual(labels.count("拘役"), 2)

    def test_atomic_report_writers_create_utf8_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_script.write_json(root / "metrics.json", {"標籤": "拘役"})
            evaluation_script.write_jsonl(root / "cases.jsonl", [{"case_id": "案件一"}])
            self.assertEqual(
                json.loads((root / "metrics.json").read_text(encoding="utf-8"))["標籤"],
                "拘役",
            )
            self.assertEqual(
                json.loads((root / "cases.jsonl").read_text(encoding="utf-8"))[
                    "case_id"
                ],
                "案件一",
            )


class InferenceContractTests(unittest.TestCase):
    def dataset_record(self, case_id="CASE-1"):
        reference = payload()
        return {
            "messages": [
                {"role": "system", "content": "請輸出 JSON"},
                {"role": "user", "content": "案件事實"},
                {
                    "role": "assistant",
                    "content": json.dumps(reference, ensure_ascii=False),
                },
            ],
            "metadata": {"case_id": case_id, "court": "測試法院"},
        }

    def test_dataset_is_split_into_prompt_and_reference(self):
        case = prepare_inference_cases([self.dataset_record()])[0]
        self.assertEqual(case.case_id, "CASE-1")
        self.assertEqual(
            [message["role"] for message in case.input_messages], ["system", "user"]
        )
        self.assertEqual(case.reference["prediction"]["sentence_type"], "罰金")

        output = make_prediction_record(
            case,
            generated_text="{}",
            split="validation",
            mode="adapter",
            base_model="taide/test",
            adapter_path="adapter",
            generation_seconds=1.25,
        )
        self.assertEqual(output["case_id"], "CASE-1")
        self.assertEqual(output["metadata"]["court"], "測試法院")

    def test_duplicate_dataset_case_id_is_rejected(self):
        record = self.dataset_record()
        with self.assertRaisesRegex(ValueError, "Duplicate metadata.case_id"):
            prepare_inference_cases([record, record])

    def test_resume_reads_only_successful_unique_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"case_id": "A", "generation_error": None})
                + "\n"
                + json.dumps({"case_id": "B", "generation_error": "failed"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_completed_case_ids(path), {"A"})

    def test_greedy_generation_does_not_pass_sampling_arguments(self):
        config = {
            "generation": {
                "max_new_tokens": 128,
                "do_sample": False,
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }
        arguments = inference_script.generation_arguments(config, max_new_tokens=None)
        self.assertEqual(arguments, {"max_new_tokens": 128, "do_sample": False})

    def test_generated_text_preserves_bpe_spacing(self):
        class FakeTensor:
            shape = (1, 2)

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

            def __call__(self, prompt, return_tensors, add_special_tokens):
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

            def decode(self, tokens, **kwargs):
                self.decode_kwargs = kwargs
                return "  保留標點前後空白  "

        class FakeModel:
            def generate(self, **kwargs):
                return FakeTensor()

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return InferenceMode()

        tokenizer = FakeTokenizer()
        generated = inference_script.generate_text(
            torch=FakeTorch(),
            model=FakeModel(),
            tokenizer=tokenizer,
            messages=[{"role": "user", "content": "案件事實"}],
            device="mps",
            generation_config={"max_new_tokens": 16, "do_sample": False},
        )
        self.assertEqual(generated, "保留標點前後空白")
        self.assertFalse(tokenizer.decode_kwargs["clean_up_tokenization_spaces"])

    def test_resume_rejects_mismatched_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            inference_script.prepare_output(
                output, {"split": "validation"}, resume=False, overwrite=False
            )
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                inference_script.prepare_output(
                    output, {"split": "test"}, resume=True, overwrite=False
                )


if __name__ == "__main__":
    unittest.main()
