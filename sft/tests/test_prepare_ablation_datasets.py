"""A/B/C 資料準備流程的回歸測試；只使用合成案件，不讀取正式資料。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SFT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SFT_ROOT / "scripts/07_prepare_ablation_datasets.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_ablation_datasets_under_test", SCRIPT_PATH
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def source_record(
    *, amount: int = 5000, sentence_type: str = "罰金", unit: str = "新臺幣元"
):
    answer = {
        "prediction": {
            "sentence_type": sentence_type,
            "sentence_amount_raw": "伍仟元",
            "sentence_amount_numeric": amount,
            "sentence_amount_unit": unit,
        },
        "sentencing_reason": "審酌犯罪情節及犯後態度。",
    }
    return {
        "messages": [
            {"role": "system", "content": "舊版系統提示"},
            {
                "role": "user",
                "content": "【犯罪事實】\n被告以言詞侮辱告訴人。\n\n請輸出JSON，欄位包含prediction與sentencing_reason。",
            },
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ],
        "metadata": {
            "case_id": "CASE-001",
            "case_no": "113年度易字第1號",
            "court": "臺灣測試地方法院",
            "judgment_date": "民國113年05月06日",
            "sentence_type": sentence_type,
        },
    }


def master_lookup(*, judge: str = "法官 白光華"):
    return {
        "CASE-001": {
            "case_id": "CASE-001",
            "case_no": "113年度易字第1號",
            "court": "臺灣測試地方法院",
            "judgment_date": "民國113年05月06日",
            "sentence_type": "罰金",
            "judge": judge,
        }
    }


class VariantPromptTests(unittest.TestCase):
    def transform(self, variant: str):
        return prepare.transform_record(
            source_record(),
            variant=variant,
            master_lookup=master_lookup(),
            master_filename="master.csv",
        )

    def test_a_exposes_no_court_year_or_judge(self):
        transformed = self.transform("a")
        prompt = transformed["messages"][1]["content"]
        self.assertNotIn("臺灣測試地方法院", prompt)
        self.assertNotIn("民國113年", prompt)
        self.assertNotIn("白光華", prompt)

    def test_b_adds_court_and_year_but_not_judge(self):
        transformed = self.transform("b")
        prompt = transformed["messages"][1]["content"]
        self.assertIn("法院：臺灣測試地方法院", prompt)
        self.assertIn("裁判年份：民國113年", prompt)
        self.assertNotIn("白光華", prompt)

    def test_c_adds_normalized_judge_and_v2_target(self):
        transformed = self.transform("c")
        prompt = transformed["messages"][1]["content"]
        answer = json.loads(transformed["messages"][2]["content"])
        self.assertIn("承審法官：白光華", prompt)
        self.assertEqual(answer["prediction"]["sentence_amount_band"], "fine_4000_5000")
        self.assertEqual(answer["prediction"]["sentence_amount_min"], 4000)
        self.assertEqual(answer["prediction"]["sentence_amount_max"], 5000)
        self.assertEqual(
            transformed["metadata"]["reference_sentence_amount_numeric"], 5000
        )
        self.assertEqual(
            transformed["metadata"]["judge_match_method"],
            "case_id+case_no+court+judgment_date+sentence_type",
        )


class ValidationTests(unittest.TestCase):
    def test_output_root_cannot_target_a_broad_project_directory(self):
        with self.assertRaisesRegex(ValueError, "must be a child"):
            prepare.validate_output_root(prepare.PROJECT_ROOT / "data/sft")

    def test_master_cross_check_rejects_a_different_court(self):
        lookup = master_lookup()
        lookup["CASE-001"]["court"] = "另一法院"
        with self.assertRaisesRegex(ValueError, "Master mismatch"):
            prepare.transform_record(
                source_record(),
                variant="c",
                master_lookup=lookup,
                master_filename="master.csv",
            )

    def test_message_order_must_be_single_turn_chat(self):
        record = source_record()
        record["messages"][0]["role"] = "user"
        with self.assertRaisesRegex(ValueError, "exactly system, user, assistant"):
            prepare.validated_enrichment(record, master_lookup())

    def test_all_six_band_boundaries_are_deterministic(self):
        cases = [
            ("罰金", 3000, "新臺幣元", "fine_1000_3000"),
            ("罰金", 4000, "新臺幣元", "fine_4000_5000"),
            ("罰金", 6000, "新臺幣元", "fine_6000_plus"),
            ("拘役", 15, "日", "detention_1_15"),
            ("拘役", 16, "日", "detention_16_20"),
            ("拘役", 21, "日", "detention_21_plus"),
        ]
        for sentence_type, amount, unit, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    prepare.amount_band(sentence_type, amount, unit)["id"], expected
                )


if __name__ == "__main__":
    unittest.main()
