"""SFT 回歸測試：以合成資料驗證失敗邊界，不下載模型或讀取正式憑證。

預設執行離線單元測試。SFT_RUN_MPS_SMOKE=1 另啟用兩步小型 Llama 訓練，
檢查 MPS、LoRA、Trackio 與 adapter 儲存；所有輸出皆位於暫存目錄。
"""

import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/03_train_sft_lora.py"
SPEC = importlib.util.spec_from_file_location("sft_training_under_test", SCRIPT)
sft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sft
SPEC.loader.exec_module(sft)


def record(reason="A complete reason."):
    return {
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Case facts."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "prediction": {"sentence_type": "fine"},
                        "sentencing_reason": reason,
                    }
                ),
            },
        ]
    }


class CharacterTokenizer:
    """只供邊界測試使用；一個字元對應一個 token，長度與遮罩可精確驗證。"""

    eos_token = "~"
    eos_token_id = ord("~")
    pad_token = "!"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(
            "<" + item["role"] + ">" + item["content"] + "~" for item in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens, truncation):
        if truncation:
            raise AssertionError("Training targets must never be silently truncated.")
        ids = list(map(ord, text))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class DefaultTaideConfigTests(unittest.TestCase):
    """直接讀取隨專案發布的 YAML，避免只測替身而漏掉模型切換設定。"""

    def setUp(self):
        import yaml

        self.model = sft.load_yaml(sft.DEFAULT_MODEL_CONFIG, yaml)
        self.training = sft.load_yaml(sft.DEFAULT_TRAINING_CONFIG, yaml)
        self.lora = sft.load_yaml(sft.DEFAULT_LORA_CONFIG, yaml)

    def test_default_model_and_credentials_contract(self):
        self.assertEqual(
            self.model["base_model_name_or_path"], "taide/Llama-3.1-TAIDE-LX-8B-Chat"
        )
        self.assertEqual(self.model["token_env_var"], "HF_TOKEN")
        self.assertFalse(self.model["trust_remote_code"])
        self.assertFalse(self.model["local_files_only"])
        self.assertNotIn("token", self.model)

    def test_outputs_and_trackio_are_isolated_for_taide(self):
        run_folder = "public_insult_taide_llama31_8b_lora"
        for key, parent in (
            ("checkpoint_dir", "checkpoints"),
            ("adapter_dir", "adapters"),
            ("log_dir", "logs"),
        ):
            with self.subTest(output=key):
                self.assertEqual(
                    self.training["output"][key], f"sft/outputs/{parent}/{run_folder}"
                )
        trainer = self.training["training"]
        self.assertEqual(
            trainer["run_name"], "public-insult-taide-llama3_1-8b-lora-r16-lr1e-4"
        )
        self.assertEqual(trainer["project"], "legal-rag-agent-public-insult-sft")
        self.assertEqual(trainer["report_to"], "trackio")
        self.assertIsNone(trainer["trackio_space_id"])
        self.assertTrue(trainer["resume_from_checkpoint"])

    def test_existing_data_paths_and_mps_budget_are_preserved(self):
        for split in ("train", "validation", "test"):
            self.assertEqual(
                self.training["data"][f"{split}_file"],
                f"data/sft/public_insult/{split}.jsonl",
            )
        self.assertEqual(self.training["data"]["max_seq_length"], 1536)
        self.assertEqual(self.training["training"]["per_device_train_batch_size"], 1)
        self.assertTrue(self.training["training"]["gradient_checkpointing"])
        self.assertEqual(self.model["torch_dtype"], "float16")
        self.assertIsNone(self.model["device_map"])

    def test_lora_covers_llama_projections_without_training_embeddings(self):
        self.assertEqual(
            set(self.lora["target_modules"]),
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            },
        )
        self.assertEqual(self.lora["r"], 16)
        self.assertIsNone(self.lora["modules_to_save"])

    def test_ablation_configs_are_isolated_and_use_the_same_sampler(self):
        import yaml

        for variant in ("a", "b", "c"):
            with self.subTest(variant=variant):
                path = (
                    sft.PROJECT_ROOT
                    / f"sft/configs/training_config_ablation_{variant}.yaml"
                )
                config = sft.load_yaml(path, yaml)
                self.assertIn(f"variant_{variant}", config["data"]["train_file"])
                self.assertIn(f"ablation_{variant}", config["output"]["adapter_dir"])
                sampling = config["training"]["class_balanced_sampling"]
                self.assertTrue(sampling["enabled"])
                self.assertEqual(sampling["minority_label"], "拘役")
                self.assertEqual(sampling["target_fraction"], 0.4)
                self.assertIsNone(config["training"]["resume_from_checkpoint"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_complete_json_and_eos_are_supervised(self):
        source = record()
        encoded = sft.TokenizedSftDataset([source], self.tokenizer, 1536)[0]
        target = "".join(chr(token) for token in encoded["labels"] if token != -100)
        self.assertEqual(target, source["messages"][-1]["content"] + "~")
        self.assertIn(-100, encoded["labels"])

    def test_overlong_answer_is_rejected_without_truncation(self):
        dataset = sft.TokenizedSftDataset(
            [record("reason " * 400)], self.tokenizer, 1536
        )
        with self.assertRaisesRegex(ValueError, "1536"):
            dataset[0]

    def test_prompt_consuming_budget_is_rejected(self):
        source = record()
        source["messages"][1]["content"] = "fact " * 400
        with self.assertRaisesRegex(ValueError, "1536"):
            sft.TokenizedSftDataset([source], self.tokenizer, 1536)[0]

    def test_exact_length_limit_keeps_eos(self):
        source = record()
        length = len(
            self.tokenizer.apply_chat_template(source["messages"], False, False)
        )
        encoded = sft.TokenizedSftDataset([source], self.tokenizer, length)[0]
        self.assertEqual(encoded["labels"][-1], self.tokenizer.eos_token_id)

    def test_later_invalid_record_is_reported(self):
        dataset = sft.TokenizedSftDataset(
            [record(), record("long " * 500)], self.tokenizer, 1536
        )
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            ValueError, "train 第 2 筆"
        ):
            dataset.validate_all("train")
        self.assertIsNone(dataset._encoded_records)

    def test_successful_preflight_reuses_checked_tokens(self):
        dataset = sft.TokenizedSftDataset([record()], self.tokenizer, 1536)
        with contextlib.redirect_stdout(io.StringIO()):
            dataset.validate_all("train")
        with patch.object(
            self.tokenizer,
            "apply_chat_template",
            side_effect=AssertionError("retokenized"),
        ):
            self.assertGreater(len(dataset[0]["input_ids"]), 0)

    def test_invalid_json_and_role_order_are_rejected(self):
        for invalid in (
            [],
            [{"role": "assistant", "content": "{}"}],
            [
                {"role": "user", "content": "fact"},
                {"role": "assistant", "content": "{broken"},
            ],
        ):
            with self.subTest(messages=invalid), self.assertRaises(ValueError):
                sft.validate_messages(invalid)

    def test_template_failure_is_not_silently_reformatted(self):
        with patch.object(
            self.tokenizer,
            "apply_chat_template",
            side_effect=ValueError("bad template"),
        ):
            with self.assertRaisesRegex(ValueError, "bad template"):
                sft.TokenizedSftDataset([record()], self.tokenizer, 1536)[0]

    def test_mismatched_prompt_prefix_is_rejected(self):
        original = self.tokenizer.apply_chat_template

        def mismatched(messages, tokenize, add_generation_prompt):
            return ("different" if add_generation_prompt else "") + original(
                messages, tokenize, add_generation_prompt
            )

        with patch.object(
            self.tokenizer, "apply_chat_template", side_effect=mismatched
        ):
            with self.assertRaisesRegex(ValueError, "not a prefix"):
                sft.TokenizedSftDataset([record()], self.tokenizer, 1536)[0]


class ClassBalancedSamplingTests(unittest.TestCase):
    @staticmethod
    def records(label, count):
        return [{"metadata": {"sentence_type": label}} for _ in range(count)]

    def test_plan_preserves_all_records_and_reaches_forty_percent(self):
        records = self.records("罰金", 1321) + self.records("拘役", 402)
        plan = sft.build_class_balanced_sampling_plan(
            records,
            {
                "enabled": True,
                "minority_label": "拘役",
                "target_fraction": 0.4,
                "seed": 42,
            },
            default_seed=7,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.extra_minority_count, 479)
        self.assertEqual(plan.target_minority_count, 881)
        self.assertEqual(plan.epoch_size, 2202)
        self.assertAlmostEqual(plan.achieved_fraction, 881 / 2202)

        indices = list(sft.RotatingMinorityOversampler(plan))
        self.assertEqual(len(indices), 2202)
        self.assertTrue(set(range(1723)).issubset(indices))
        self.assertEqual(sum(index >= 1321 for index in indices), 881)

    def test_sampler_is_reproducible_and_rotates_between_epochs(self):
        records = self.records("罰金", 6) + self.records("拘役", 4)
        plan = sft.build_class_balanced_sampling_plan(
            records,
            {
                "enabled": True,
                "minority_label": "拘役",
                "target_fraction": 0.5,
                "seed": 9,
            },
            default_seed=42,
        )
        first = sft.RotatingMinorityOversampler(plan)
        second = sft.RotatingMinorityOversampler(plan)
        first.set_epoch(0)
        second.set_epoch(0)
        epoch_zero = list(first)
        self.assertEqual(epoch_zero, list(second))
        first.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(first))

    def test_target_below_original_fraction_is_rejected(self):
        records = self.records("罰金", 6) + self.records("拘役", 4)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            sft.build_class_balanced_sampling_plan(
                records,
                {"enabled": True, "minority_label": "拘役", "target_fraction": 0.3},
                default_seed=42,
            )


class TrainingArgumentTests(unittest.TestCase):
    def test_new_api_maps_sampler_warmup_and_evaluation(self):
        class NewArguments:
            def __init__(self, train_sampling_strategy, warmup_steps, eval_strategy):
                self.sampler, self.warmup, self.evaluation = (
                    train_sampling_strategy,
                    warmup_steps,
                    eval_strategy,
                )

        actual = sft.compact_training_args(
            NewArguments,
            {
                "group_by_length": True,
                "warmup_ratio": 0.03,
                "evaluation_strategy": "steps",
                "save_safetensors": True,
            },
        )
        self.assertEqual(
            (actual.sampler, actual.warmup, actual.evaluation),
            ("group_by_length", 0.03, "steps"),
        )

    def test_legacy_api_preserves_supported_values(self):
        class LegacyArguments:
            def __init__(self, group_by_length, warmup_ratio, evaluation_strategy):
                self.values = group_by_length, warmup_ratio, evaluation_strategy

        actual = sft.compact_training_args(
            LegacyArguments,
            {
                "group_by_length": True,
                "warmup_ratio": 0.03,
                "evaluation_strategy": "steps",
            },
        )
        self.assertEqual(actual.values, (True, 0.03, "steps"))

    def test_unknown_option_fails_instead_of_disappearing(self):
        class Arguments:
            def __init__(self):
                pass

        with self.assertRaisesRegex(ValueError, "unknown_setting"):
            sft.compact_training_args(Arguments, {"unknown_setting": True})

    def test_installed_api_preserves_warmup_and_local_trackio(self):
        import torch
        from transformers import TrainingArguments

        with tempfile.TemporaryDirectory() as folder:
            actual = sft.build_training_arguments(
                TrainingArguments,
                torch,
                {
                    "group_by_length": True,
                    "warmup_ratio": 0.03,
                },
                Path(folder),
                "trackio",
            )
        self.assertEqual(actual.get_warmup_steps(100), 3)
        self.assertEqual(
            getattr(actual, "train_sampling_strategy", "group_by_length"),
            "group_by_length",
        )
        self.assertIsNone(actual.trackio_space_id)


class FinalEvaluationTests(unittest.TestCase):
    def test_reuses_eval_metrics_logged_at_the_final_global_step(self):
        trainer = SimpleNamespace(
            state=SimpleNamespace(
                global_step=414,
                log_history=[
                    {"step": 400, "epoch": 2.9, "eval_loss": 0.56},
                    {
                        "step": 414,
                        "epoch": 3.0,
                        "eval_loss": 0.55,
                        "eval_runtime": 410.0,
                    },
                    {"step": 414, "epoch": 3.0, "train_loss": 0.52},
                ],
            )
        )

        self.assertEqual(
            sft.final_eval_metrics_from_history(trainer),
            {
                "epoch": 3.0,
                "eval_loss": 0.55,
                "eval_runtime": 410.0,
            },
        )

    def test_stale_eval_metrics_do_not_suppress_final_evaluation(self):
        trainer = SimpleNamespace(
            state=SimpleNamespace(
                global_step=414,
                log_history=[{"step": 400, "epoch": 2.9, "eval_loss": 0.56}],
            )
        )
        self.assertIsNone(sft.final_eval_metrics_from_history(trainer))


class OutputSafetyTests(unittest.TestCase):
    def test_cli_rejects_dry_run_with_overwrite(self):
        with patch.object(
            sys, "argv", [str(SCRIPT), "--dry-run", "--overwrite-output-dir"]
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                sft.parse_args()

    def test_direct_main_call_rejects_destructive_dry_run_before_dependencies(self):
        with patch.object(
            sft,
            "parse_args",
            return_value=SimpleNamespace(dry_run=True, overwrite_output_dir=True),
        ):
            with patch.object(
                sft, "require_training_dependencies"
            ) as dependencies, self.assertRaises(ValueError):
                sft.main()
        dependencies.assert_not_called()

    def test_overwrite_rejects_broad_or_overlapping_targets(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            sft, "PROJECT_ROOT", Path(folder)
        ):
            root = Path(folder)
            run = root / "sft/outputs/checkpoints/run"
            for paths in (
                [root],
                [root / "sft/outputs/checkpoints"],
                [run, run / "nested"],
            ):
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    sft.validate_overwrite_targets(paths)

    def test_symlink_cannot_redirect_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            sft, "PROJECT_ROOT", Path(folder)
        ):
            root = Path(folder)
            outside = root / "data"
            outside.mkdir()
            run = root / "sft/outputs/checkpoints/run"
            run.parent.mkdir(parents=True)
            run.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                sft.validate_overwrite_targets([run])
            self.assertTrue(outside.exists())

    def test_symlinked_output_root_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project"
            outside = Path(folder) / "outside"
            (root / "sft").mkdir(parents=True)
            outside.mkdir()
            (root / "sft/outputs").symlink_to(outside, target_is_directory=True)
            with patch.object(sft, "PROJECT_ROOT", root), self.assertRaises(ValueError):
                sft.validate_overwrite_targets([root / "sft/outputs/checkpoints/run"])


class DownloadErrorTests(unittest.TestCase):
    def test_gated_error_identifies_model_access_without_exposing_secrets(self):
        import httpx
        from huggingface_hub.errors import GatedRepoError

        response = httpx.Response(
            403, request=httpx.Request("GET", "https://example.test/model")
        )
        cause = GatedRepoError("private-token-or-signed-url", response=response)
        wrapper = OSError("load failed")
        wrapper.__cause__ = cause
        model_name = "taide/Llama-3.1-TAIDE-LX-8B-Chat"
        message = sft.build_model_access_error_message(
            model_name, {"token_env_var": "HF_TOKEN"}, wrapper
        )
        self.assertIn(f"https://huggingface.co/{model_name}", message)
        self.assertIn("有 token 不代表已通過模型授權", message)
        self.assertIn("HF_TOKEN", message)
        self.assertNotIn("private-token-or-signed-url", message)

    def test_missing_env_token_preserves_hub_cached_login(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(sft.model_access_token({"token_env_var": "HF_TOKEN"}))

    def test_network_error_is_not_reported_as_gated(self):
        import httpx

        cause = httpx.ConnectTimeout("private signed URL should not appear")
        wrapper = OSError("load failed")
        wrapper.__cause__ = cause
        message = sft.build_model_access_error_message("test/model", {}, wrapper)
        self.assertIn("網路連線", message)
        self.assertNotIn("hf auth login", message)
        self.assertNotIn("private signed URL", message)

    def test_http_statuses_have_separate_actions(self):
        import httpx

        for status, expected in (
            (401, "hf auth login"),
            (403, "hf auth login"),
            (404, "找不到"),
            (429, "頻率受限"),
            (503, "5xx"),
        ):
            with self.subTest(status=status):
                response = httpx.Response(
                    status, request=httpx.Request("GET", "https://example.test/model")
                )
                error = httpx.HTTPStatusError(
                    "failed", request=response.request, response=response
                )
                self.assertIn(
                    expected,
                    sft.build_model_access_error_message("test/model", {}, error),
                )

    def test_cache_and_permissions_are_not_authentication_errors(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        for error, expected in (
            (LocalEntryNotFoundError("absent"), "快取"),
            (PermissionError("denied"), "本機檔案權限"),
            (OSError("unknown"), "沒有足夠資訊"),
        ):
            with self.subTest(error=error):
                message = sft.build_model_access_error_message("test/model", {}, error)
                self.assertIn(expected, message)
                self.assertNotIn("hf auth login", message)

    def test_loader_reports_start_and_preserves_network_cause(self):
        import httpx

        loader = Mock()
        cause = httpx.ConnectTimeout("unreachable")
        loader.from_pretrained.side_effect = cause
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(
            RuntimeError
        ) as caught:
            sft.load_pretrained_component(
                loader, {"base_model_name_or_path": "test/model"}, "載入 tokenizer"
            )
        self.assertIs(caught.exception.__cause__, cause)
        self.assertIn("開始載入 tokenizer", output.getvalue())

    def test_reporter_stops_on_keyboard_interrupt(self):
        with patch.object(sft, "Event") as event, patch.object(sft, "Thread") as thread:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                KeyboardInterrupt
            ):
                with sft.loading_progress("test"):
                    raise KeyboardInterrupt()
        event.return_value.set.assert_called_once()
        thread.return_value.join.assert_called_once()

    def test_cache_lock_wait_is_visible_and_log_level_is_restored(self):
        logger = logging.getLogger("huggingface_hub.utils._fixes")
        original_level = logger.level
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with sft.loading_progress("tokenizer"):
                    logger.info("Still waiting to acquire lock on tokenizer")
            self.assertIn("Still waiting to acquire lock", output.getvalue())
            self.assertEqual(logger.level, logging.WARNING)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)


class MainWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.configs = {
            "model_config.yaml": {
                "base_model_name_or_path": "synthetic/offline",
                "torch_dtype": "float16",
            },
            "lora_config.yaml": {
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            },
            "training_config.yaml": {
                "data": {
                    "train_file": "train.jsonl",
                    "validation_file": "validation.jsonl",
                    "test_file": "test.jsonl",
                    "max_seq_length": 1536,
                },
                "output": {
                    "checkpoint_dir": str(self.root / "sft/outputs/checkpoints/run"),
                    "adapter_dir": str(self.root / "sft/outputs/adapters/run"),
                    "log_dir": str(self.root / "logs"),
                },
                "training": {"report_to": "none", "group_by_length": True},
            },
        }
        self.rows = {
            name: [record()]
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
        }
        self.args = SimpleNamespace(
            model_config=Path("model_config.yaml"),
            training_config=Path("training_config.yaml"),
            lora_config=Path("lora_config.yaml"),
            dry_run=True,
            overwrite_output_dir=False,
        )

    @contextlib.contextmanager
    def environment(self, tokenizer=None, model_loader=None, output=None):
        deps = sft.require_training_dependencies()
        deps["AutoTokenizer"] = SimpleNamespace(
            from_pretrained=Mock(return_value=tokenizer or CharacterTokenizer())
        )
        deps["AutoModelForCausalLM"] = model_loader or SimpleNamespace(
            from_pretrained=Mock(side_effect=AssertionError("weights loaded"))
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sft, "PROJECT_ROOT", self.root))
            stack.enter_context(patch.object(sft, "load_project_env"))
            stack.enter_context(patch.object(sft, "parse_args", return_value=self.args))
            stack.enter_context(
                patch.object(sft, "require_training_dependencies", return_value=deps)
            )
            stack.enter_context(
                patch.object(
                    sft,
                    "load_yaml",
                    side_effect=lambda path, yaml: self.configs[path.name],
                )
            )
            stack.enter_context(
                patch.object(
                    sft, "load_jsonl", side_effect=lambda path: self.rows[path.name]
                )
            )
            stack.enter_context(contextlib.redirect_stdout(output or io.StringIO()))
            yield deps

    def test_successful_dry_run_leaves_existing_outputs_untouched(self):
        checkpoint = Path(
            self.configs["training_config.yaml"]["output"]["checkpoint_dir"]
        )
        checkpoint.mkdir(parents=True)
        sentinel = checkpoint / "existing.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.environment() as deps, patch.object(
            sft, "remove_output_dir"
        ) as remove:
            sft.main()
        remove.assert_not_called()
        deps["AutoModelForCausalLM"].from_pretrained.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_all_splits_are_checked_and_weights_are_not_loaded_on_failure(self):
        for rows in self.rows.values():
            rows.append(record("too long " * 300))
        with self.environment() as deps, patch.object(
            sft, "remove_output_dir"
        ) as remove:
            with self.assertRaises(ValueError) as caught:
                sft.main()
        for split in ("train", "validation", "test"):
            self.assertIn(split + " 第 2 筆", str(caught.exception))
        remove.assert_not_called()
        deps["AutoModelForCausalLM"].from_pretrained.assert_not_called()

    def test_resume_and_overwrite_are_rejected_before_cleanup(self):
        self.args.dry_run = False
        self.args.overwrite_output_dir = True
        self.configs["training_config.yaml"]["training"][
            "resume_from_checkpoint"
        ] = "checkpoint-10"
        with self.environment(), patch.object(sft, "remove_output_dir") as remove:
            with self.assertRaisesRegex(ValueError, "resume_from_checkpoint"):
                sft.main()
        remove.assert_not_called()

    def test_failed_model_load_does_not_clear_existing_outputs(self):
        self.args.dry_run = False
        self.args.overwrite_output_dir = True
        loader = SimpleNamespace(
            from_pretrained=Mock(side_effect=OSError("not available"))
        )
        with self.environment(model_loader=loader), patch.object(
            sft, "remove_output_dir"
        ) as remove:
            with self.assertRaises(RuntimeError):
                sft.main()
        remove.assert_not_called()

    def test_taide_tokenizer_and_weights_share_model_source_and_token(self):
        import yaml

        self.configs["model_config.yaml"] = sft.load_yaml(
            sft.DEFAULT_MODEL_CONFIG, yaml
        )
        self.args.dry_run = False
        # 權重載入處刻意停止，驗證真正 main 流程的來源與憑證傳遞，但不下載或訓練。
        loader = SimpleNamespace(
            from_pretrained=Mock(side_effect=OSError("stop before training"))
        )
        output = io.StringIO()
        fake_token = "synthetic-test-token"
        with patch.dict(os.environ, {"HF_TOKEN": fake_token}), self.environment(
            model_loader=loader, output=output
        ) as deps:
            with self.assertRaises(RuntimeError):
                sft.main()
        for component in (deps["AutoTokenizer"], loader):
            component.from_pretrained.assert_called_once()
            call = component.from_pretrained.call_args
            self.assertEqual(call.args[0], "taide/Llama-3.1-TAIDE-LX-8B-Chat")
            self.assertEqual(call.kwargs["token"], fake_token)
            self.assertFalse(call.kwargs["trust_remote_code"])
        self.assertNotIn(fake_token, output.getvalue())

    @unittest.skipUnless(
        os.environ.get("SFT_RUN_MPS_SMOKE") == "1", "Opt-in MPS integration test"
    )
    def test_two_step_mps_training_trackio_and_adapter_reload(self):
        import torch

        self.assertTrue(
            torch.backends.mps.is_available(), "Run outside the sandbox to access MPS."
        )
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from peft import PeftModel

        vocab = {
            word: index
            for index, word in enumerate(
                [
                    "[UNK]",
                    "[PAD]",
                    "[EOS]",
                    "system",
                    "user",
                    "assistant",
                    "fine",
                    "reason",
                ]
            )
        }
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        model_config = LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=1,
            eos_token_id=2,
        )
        loader = SimpleNamespace(
            from_pretrained=lambda *a, **k: LlamaForCausalLM(model_config).to(
                dtype=torch.float16
            )
        )
        self.args.dry_run = False
        self.configs["training_config.yaml"]["training"].update(
            max_steps=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            eval_steps=1,
            save_steps=1,
            report_to="trackio",
            project="sft-regression",
            run_name="tiny-mps",
        )
        self.rows["train.jsonl"] = [record(), record("A different reason.")]
        # 本機 Trackio 測試明確排除外部 Space／Server 設定，不接觸正式實驗紀錄。
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("TRACKIO_")
        }
        env.update(
            TRACKIO_DIR=str(self.root / "trackio"),
            HF_HUB_OFFLINE="1",
            GRADIO_ANALYTICS_ENABLED="False",
        )
        with patch.dict(os.environ, env, clear=True), self.environment(
            tokenizer, loader
        ):
            try:
                sft.main()
            finally:
                import trackio

                trackio.finish()
        checkpoint = Path(
            self.configs["training_config.yaml"]["output"]["checkpoint_dir"]
        )
        result = json.loads((checkpoint / "train_results.json").read_text())
        self.assertTrue(torch.isfinite(torch.tensor(result["train_loss"])))
        self.assertGreater(result["train_loss"], 0)
        adapter = Path(self.configs["training_config.yaml"]["output"]["adapter_dir"])
        self.assertTrue((adapter / "adapter_model.safetensors").exists())
        PeftModel.from_pretrained(LlamaForCausalLM(model_config), adapter)
        self.assertTrue(any((self.root / "trackio").glob("*.db")))


if __name__ == "__main__":
    unittest.main()
