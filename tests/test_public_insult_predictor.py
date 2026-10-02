"""驗證 TAIDE A／B／C 動態路由、輸出結構與風險警示。"""

import json
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from app.inference.public_insult_predictor import (
    DEFAULT_ADAPTER_A_ID,
    DEFAULT_ADAPTER_A_REVISION,
    DEFAULT_ADAPTER_B_ID,
    DEFAULT_ADAPTER_B_REVISION,
    DEFAULT_ADAPTER_C_ID,
    DEFAULT_ADAPTER_C_REVISION,
    PUBLISHED_ADAPTER_A_ID,
    PUBLISHED_ADAPTER_B_ID,
    PUBLISHED_ADAPTER_C_ID,
    PUBLISHED_ADAPTER_REVISION,
    PredictorSettings,
    PublicInsultPredictorError,
    build_prediction_messages,
    build_variant_a_messages,
    build_variant_b_messages,
    build_variant_c_messages,
    prediction_scope_warnings,
    select_input_variant,
    validate_generated_output,
)


class PublicInsultPredictorContractTests(TestCase):
    def test_default_release_is_pinned(self) -> None:
        settings = PredictorSettings()
        self.assertEqual(settings.adapter_c_id, DEFAULT_ADAPTER_C_ID)
        self.assertEqual(
            settings.adapter_c_revision,
            DEFAULT_ADAPTER_C_REVISION,
        )
        self.assertEqual(
            PUBLISHED_ADAPTER_A_ID,
            "a97141481/Llama-3.1-TAIDE-Public-Insult-A-LoRA",
        )
        self.assertEqual(
            PUBLISHED_ADAPTER_B_ID,
            "a97141481/Llama-3.1-TAIDE-Public-Insult-B-LoRA",
        )
        self.assertEqual(
            PUBLISHED_ADAPTER_C_ID,
            "a97141481/Llama-3.1-TAIDE-Public-Insult-C-LoRA",
        )
        self.assertEqual(PUBLISHED_ADAPTER_REVISION, "v1.0")
        self.assertNotEqual(settings.base_model_revision, "main")
        for default_id, default_revision, published_id in (
            (
                DEFAULT_ADAPTER_A_ID,
                DEFAULT_ADAPTER_A_REVISION,
                PUBLISHED_ADAPTER_A_ID,
            ),
            (
                DEFAULT_ADAPTER_B_ID,
                DEFAULT_ADAPTER_B_REVISION,
                PUBLISHED_ADAPTER_B_ID,
            ),
        ):
            if Path(default_id).is_dir():
                self.assertIsNone(default_revision)
            else:
                self.assertEqual(default_id, published_id)
                self.assertEqual(
                    default_revision,
                    PUBLISHED_ADAPTER_REVISION,
                )

    def test_variant_a_prompt_has_no_institutional_fields(self) -> None:
        messages = build_variant_a_messages(
            criminal_facts="犯罪事實",
            sentencing_factors="量刑因素",
        )
        user_prompt = messages[1]["content"]
        self.assertNotIn("案件制度資訊", user_prompt)
        self.assertNotIn("法院：", user_prompt)

    def test_variant_b_prompt_contains_court_and_year_only(self) -> None:
        messages = build_variant_b_messages(
            criminal_facts="犯罪事實",
            sentencing_factors="量刑因素",
            court="臺灣臺北地方法院",
            judgment_year_roc=114,
        )
        user_prompt = messages[1]["content"]
        self.assertIn("法院：臺灣臺北地方法院", user_prompt)
        self.assertIn("裁判年份：民國114年", user_prompt)
        self.assertNotIn("承審法官", user_prompt)

    def test_variant_c_prompt_contains_every_required_field(self) -> None:
        messages = build_variant_c_messages(
            criminal_facts="被告在公開場所以貶抑性言詞辱罵告訴人。",
            sentencing_factors="被告坦承犯行，尚未和解。",
            court="臺灣臺北地方法院",
            judgment_year_roc=114,
            judge_name="王小明",
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        user_prompt = messages[1]["content"]
        self.assertIn("法院：臺灣臺北地方法院", user_prompt)
        self.assertIn("裁判年份：民國114年", user_prompt)
        self.assertIn("承審法官：王小明", user_prompt)

    def test_blank_required_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "承審法官"):
            build_variant_c_messages(
                criminal_facts="犯罪事實",
                sentencing_factors="量刑因素",
                court="臺灣臺北地方法院",
                judgment_year_roc=114,
                judge_name=" ",
            )

    def test_dynamic_router_selects_highest_complete_variant(self) -> None:
        self.assertEqual(select_input_variant(), "A")
        self.assertEqual(
            select_input_variant(
                court="臺灣臺北地方法院",
                judgment_year_roc=114,
            ),
            "B",
        )
        self.assertEqual(
            select_input_variant(
                court="臺灣臺北地方法院",
                judgment_year_roc=114,
                judge_name="王小明",
            ),
            "C",
        )

    def test_incomplete_optional_fields_fall_back_without_placeholders(self) -> None:
        variant, messages = build_prediction_messages(
            criminal_facts="犯罪事實",
            sentencing_factors="量刑因素",
            court="臺灣臺北地方法院",
            judge_name="王小明",
        )

        self.assertEqual(variant, "A")
        self.assertNotIn(
            "案件制度資訊",
            messages[1]["content"],
        )

    def test_valid_fine_output_is_accepted(self) -> None:
        payload = {
            "prediction": {
                "sentence_type": "罰金",
                "sentence_amount_band": "fine_4000_5000",
                "sentence_amount_min": 4000,
                "sentence_amount_max": 5000,
                "sentence_amount_unit": "新臺幣元",
            },
            "sentencing_reason": "爰審酌案件情節及犯後態度等一切情狀。",
        }
        output = validate_generated_output(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(output.prediction.sentence_amount_band, "fine_4000_5000")

    def test_band_and_limits_must_match(self) -> None:
        payload = {
            "prediction": {
                "sentence_type": "罰金",
                "sentence_amount_band": "fine_4000_5000",
                "sentence_amount_min": 3001,
                "sentence_amount_max": 5000,
                "sentence_amount_unit": "新臺幣元",
            },
            "sentencing_reason": "量刑理由",
        }
        with self.assertRaisesRegex(PublicInsultPredictorError, "schema"):
            validate_generated_output(json.dumps(payload, ensure_ascii=False))

    def test_non_json_output_is_rejected_without_extraction(self) -> None:
        with self.assertRaisesRegex(PublicInsultPredictorError, "合法 JSON"):
            validate_generated_output("```json\n{}\n```")

    def test_out_of_distribution_inputs_receive_warnings(self) -> None:
        warnings = prediction_scope_warnings(
            court="臺灣高雄地方法院",
            judgment_year_roc=115,
        )
        self.assertEqual(len(warnings), 2)

    def test_environment_token_is_not_exposed_by_repr(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_for_test"}, clear=False):
            settings = PredictorSettings.from_environment()
        self.assertNotIn("hf_secret_for_test", repr(settings))


class PublicInsultToolRegistrationTests(TestCase):
    def test_agent_registers_only_core_legal_tools(self) -> None:
        from app.agent.legal_agent import TOOLS

        self.assertEqual(
            {registered_tool.name for registered_tool in TOOLS},
            {
                "search_similar_judgments",
                "search_public_insult_authorities",
                "lookup_criminal_law_article",
                "predict_public_insult_sentence",
            },
        )

    def test_optional_institutional_fields_are_not_required(self) -> None:
        from app.tools.public_insult_prediction import (
            predict_public_insult_sentence,
        )

        schema = predict_public_insult_sentence.tool_call_schema.model_json_schema()

        self.assertEqual(
            set(schema["required"]),
            {"criminal_facts", "sentencing_factors"},
        )
        self.assertNotIn("runtime", schema["properties"])
