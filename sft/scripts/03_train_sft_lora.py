"""公然侮辱判決結果與量刑理由的單輪對話 SFT 入口。

資料契約：每筆樣本包含可選的 system、user 與最後一則 assistant JSON。
訓練只計算 assistant 的 loss；長度不足以容納完整答案時，前置檢查必須失敗。
dry-run 可以下載 tokenizer 與使用 Hugging Face 快取，但不更動訓練輸出。
預設 base model 為 TAIDE Llama 3.1；模型來源只由 model_config.yaml 決定，
不在下載失敗時改用其他模型，以免混用 tokenizer、詞彙表與 adapter。
"""

import argparse
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
import random
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Any, Iterator


def configure_runtime_environment() -> None:
    """在載入 torch／Hub 前設定預設值，並尊重終端機與 .env 的覆寫。"""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # 停用 Xet 用戶端後仍可能轉址至同一個 CDN，不能視為網路故障的解法。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # 此值限制單次請求；Hub 自帶重試，因此整體等待時間可能更長。
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


# 專案根目錄。
#
# 實務上，訓練腳本常被從不同位置呼叫：
#   python sft/scripts/03_train_sft_lora.py
#   cd sft && python scripts/03_train_sft_lora.py
# 如果路徑都用目前工作目錄推算，很容易在第二種情境讀不到 data/config。
# 因此這裡固定用「此檔案所在位置」回推到專案根目錄。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "sft/configs/model_config.yaml"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "sft/configs/training_config.yaml"
DEFAULT_LORA_CONFIG = PROJECT_ROOT / "sft/configs/lora_config.yaml"


def load_project_env() -> None:
    # 從專案根目錄載入 .env。
    #
    # 這讓使用者可以把 HF_TOKEN / OPENAI_API_KEY 等本機憑證放在 .env，
    # 不需要每次開終端機都手動 export。override=False 表示：
    # 如果終端機已經有同名環境變數，保留終端機的值，不被 .env 覆蓋。
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("注意：找到 .env，但未安裝 python-dotenv，將略過 .env 載入。")
        return

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )


def parse_args() -> argparse.Namespace:
    # 只把真正需要從命令列調整的項目做成參數。
    # 其他訓練超參數集中放在 YAML，便於記錄與重現實驗。
    parser = argparse.ArgumentParser(
        description="Train a LoRA SFT adapter for public insult judgment prediction and sentencing-reason generation."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to model_config.yaml.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to training_config.yaml.",
    )
    parser.add_argument(
        "--lora-config",
        type=Path,
        default=DEFAULT_LORA_CONFIG,
        help="Path to lora_config.yaml.",
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all splits and tokenize every record; do not load weights or change training outputs.",
    )
    execution_mode.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete the checkpoint and adapter directories before training.",
    )
    return parser.parse_args()


def require_training_dependencies() -> dict[str, Any]:
    # 把訓練相關套件集中檢查。
    #
    # 這樣做有兩個好處：
    # 1. 使用者如果少裝套件，會看到「缺哪幾個」而不是在深層 import 爆掉。
    # 2. dry-run / 訓練主流程可以共用同一組依賴檢查，避免環境問題被誤判成資料問題。
    missing: list[str] = []

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
        missing.append("torch")

    try:
        yaml = importlib.import_module("yaml")
    except ImportError:
        yaml = None
        missing.append("pyyaml")

    try:
        peft = importlib.import_module("peft")
        LoraConfig = peft.LoraConfig
        get_peft_model = peft.get_peft_model
    except ImportError:
        LoraConfig = None
        get_peft_model = None
        missing.append("peft")

    if importlib.util.find_spec("accelerate") is None:
        missing.append("accelerate")

    try:
        transformers = importlib.import_module("transformers")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer
        Trainer = transformers.Trainer
        TrainingArguments = transformers.TrainingArguments
        set_seed = transformers.set_seed
    except ImportError:
        AutoModelForCausalLM = None
        AutoTokenizer = None
        Trainer = None
        TrainingArguments = None
        set_seed = None
        missing.append("transformers")

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing SFT training dependencies: "
            f"{unique_missing}. Install them before training, for example: "
            "pip install torch transformers peft accelerate pyyaml"
        )

    return {
        "torch": torch,
        "yaml": yaml,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def normalize_report_to(report_to: Any) -> str | list[str]:
    # TrainingArguments 接受字串或字串列表。
    #
    # YAML 裡我們通常寫：
    #   report_to: trackio
    # 或：
    #   report_to: none
    # 若未來要同時記錄到多個工具，也可以改成：
    #   report_to:
    #     - trackio
    #     - tensorboard
    if report_to is None:
        return "none"

    if isinstance(report_to, str):
        return report_to

    if isinstance(report_to, list) and all(isinstance(item, str) for item in report_to):
        return report_to

    raise ValueError("training.report_to must be a string, a list of strings, or null.")


def report_to_contains(report_to: str | list[str], name: str) -> bool:
    # report_to 可能是 "trackio"，也可能是 ["trackio", "tensorboard"]。
    # 集中處理格式差異，主流程就能保持乾淨。
    if isinstance(report_to, str):
        return report_to == name

    return name in report_to


def require_tracking_dependencies(report_to: str | list[str]) -> None:
    # Transformers 的 Trackio callback 只有在 trackio 套件已安裝時才能啟用。
    #
    # 如果使用者設定 report_to: trackio，卻沒有安裝套件，Trainer 會在較後面才報錯。
    # 這裡提前檢查，讓錯誤訊息更接近真正原因。
    if not report_to_contains(report_to, "trackio"):
        return

    if importlib.util.find_spec("trackio") is None:
        raise RuntimeError(
            "training.report_to is set to 'trackio', but the trackio package is not installed. "
            "Install it with: pip install trackio"
        )


def load_yaml(path: Path, yaml_module: Any) -> dict[str, Any]:
    # YAML 是這個訓練流程的實驗設定來源。
    # 這裡只負責讀檔與確認最外層是 mapping；
    # 欄位語意留在 main() 裡處理，避免過早把設定拆散。
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml_module.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data


def project_path(path_value: str | Path) -> Path:
    # 設定檔裡盡量使用相對路徑，讓專案搬家後仍可運作。
    # 如果使用者真的給絕對路徑，則尊重該路徑，不再接 PROJECT_ROOT。
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # 資料集採 OpenAI chat-style JSONL：一列是一筆樣本，每筆樣本都要有 messages。
    #
    # 此處檢查 JSON、單輪角色順序與答案必要欄位；分詞後才確認長度與 loss 遮罩。
    # 01_inspect_sft_dataset.py 的統計報告不能取代此入口檢查，避免繞過報告就誤訓練。
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record must be an object at {path}:{line_number}"
                )
            try:
                validate_messages(record.get("messages"))
            except ValueError as error:
                raise ValueError(
                    f"Invalid record at {path}:{line_number}: {error}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(f"No records found in dataset: {path}")

    return records


def validate_messages(messages: Any) -> None:
    """確認角色順序及答案結構，避免把多輪 user 訊息誤當成監督目標。"""
    if not isinstance(messages, list) or not all(
        isinstance(item, dict) for item in messages
    ):
        raise ValueError("messages must be a list of objects.")
    roles = [message.get("role") for message in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise ValueError("Expected [system,] user, assistant in a single turn.")
    for message in messages:
        if (
            not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("Every message must have non-empty text content.")
    try:
        answer = json.loads(messages[-1]["content"])
    except json.JSONDecodeError as error:
        # 不把案件全文帶進例外訊息，日誌只需要能定位資料與錯誤類型。
        raise ValueError("assistant content must be complete JSON.") from error
    if not isinstance(answer, dict) or not isinstance(answer.get("prediction"), dict):
        raise ValueError("assistant JSON must contain a prediction object.")
    prediction = answer["prediction"]
    sentence_type = prediction.get("sentence_type")
    reason = answer.get("sentencing_reason")
    if not isinstance(sentence_type, str) or not sentence_type.strip():
        raise ValueError("prediction.sentence_type must be non-empty text.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("sentencing_reason must be non-empty text.")

    # v1 精確刑度與 v2 區間刑度都可以由同一入口訓練，但一筆答案只能完整符合其中一種。
    # 在 tokenizer 之前驗證這些欄位，可避免模型花數小時學習一個本來就矛盾的 schema。
    if "sentence_amount_band" in prediction:
        band = prediction.get("sentence_amount_band")
        minimum = prediction.get("sentence_amount_min")
        maximum = prediction.get("sentence_amount_max")
        unit = prediction.get("sentence_amount_unit")
        if not isinstance(band, str) or not band.strip():
            raise ValueError("prediction.sentence_amount_band must be non-empty text.")
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise ValueError("prediction.sentence_amount_min must be numeric.")
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, (int, float))
        ):
            raise ValueError("prediction.sentence_amount_max must be numeric or null.")
        if maximum is not None and maximum < minimum:
            raise ValueError(
                "prediction.sentence_amount_max cannot be lower than sentence_amount_min."
            )
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("prediction.sentence_amount_unit must be non-empty text.")


@dataclass(frozen=True)
class ClassBalancedSamplingPlan:
    """動態過採樣的不可變計畫；實際索引會依 epoch 與 seed 重新排列。"""

    minority_label: str
    target_fraction: float
    seed: int
    base_size: int
    minority_indices: tuple[int, ...]
    original_minority_count: int
    target_minority_count: int
    extra_minority_count: int
    epoch_size: int

    @property
    def achieved_fraction(self) -> float:
        return self.target_minority_count / self.epoch_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "minority_label": self.minority_label,
            "target_fraction": self.target_fraction,
            "seed": self.seed,
            "base_size": self.base_size,
            "original_minority_count": self.original_minority_count,
            "target_minority_count": self.target_minority_count,
            "extra_minority_count": self.extra_minority_count,
            "epoch_size": self.epoch_size,
            "achieved_fraction": self.achieved_fraction,
        }


def build_class_balanced_sampling_plan(
    records: list[dict[str, Any]],
    sampling_config: Any,
    *,
    default_seed: int,
) -> ClassBalancedSamplingPlan | None:
    """依 metadata.sentence_type 計算過採樣量，不建立或修改任何 JSONL。"""
    if sampling_config in (None, False):
        return None
    if not isinstance(sampling_config, dict):
        raise ValueError("training.class_balanced_sampling must be a mapping or null.")
    if not bool(sampling_config.get("enabled", False)):
        return None

    minority_label = str(sampling_config.get("minority_label", "拘役")).strip()
    target_fraction = float(sampling_config.get("target_fraction", 0.4))
    seed = int(sampling_config.get("seed", default_seed))
    if not minority_label:
        raise ValueError("class_balanced_sampling.minority_label cannot be empty.")
    if not 0 < target_fraction < 1:
        raise ValueError(
            "class_balanced_sampling.target_fraction must be between 0 and 1."
        )

    labels: list[str] = []
    for index, record in enumerate(records):
        metadata = record.get("metadata")
        label = metadata.get("sentence_type") if isinstance(metadata, dict) else None
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Train record {index + 1} has no metadata.sentence_type for balanced sampling."
            )
        labels.append(label.strip())

    minority_indices = tuple(
        index for index, label in enumerate(labels) if label == minority_label
    )
    minority_count = len(minority_indices)
    majority_count = len(records) - minority_count
    if minority_count == 0 or majority_count == 0:
        raise ValueError(
            "Balanced sampling requires both the minority label and at least one other label."
        )
    original_fraction = minority_count / len(records)
    if target_fraction <= original_fraction:
        raise ValueError(
            "class_balanced_sampling.target_fraction must exceed the original minority fraction "
            f"({original_fraction:.6f})."
        )

    # 保留全部原始樣本一次，只增加少數類別曝光。ceil 確保實際比例不低於設定目標；
    # 目前 1321/402 的資料會得到 881 筆拘役、額外 479 個索引與 2202 筆 epoch。
    target_minority_count = math.ceil(
        target_fraction * majority_count / (1 - target_fraction)
    )
    extra_count = target_minority_count - minority_count
    epoch_size = len(records) + extra_count
    return ClassBalancedSamplingPlan(
        minority_label=minority_label,
        target_fraction=target_fraction,
        seed=seed,
        base_size=len(records),
        minority_indices=minority_indices,
        original_minority_count=minority_count,
        target_minority_count=target_minority_count,
        extra_minority_count=extra_count,
        epoch_size=epoch_size,
    )


class RotatingMinorityOversampler:
    """每個 epoch 保留全部原始索引，再以可重現輪替方式追加少數類索引。

    這不是 weighted sampling：多數類不會因隨機抽樣而整個 epoch 都沒被看到。
    `set_epoch` 由 Trainer/Accelerate 呼叫，恢復 checkpoint 時也能重建同一索引順序。
    """

    def __init__(self, plan: ClassBalancedSamplingPlan) -> None:
        self.plan = plan
        self.epoch = 0

    def __len__(self) -> int:
        return self.plan.epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        minority = list(self.plan.minority_indices)
        extra: list[int] = []
        remaining = self.plan.extra_minority_count
        cycle = 0
        while remaining:
            cycle_indices = minority.copy()
            random.Random(self.plan.seed + self.epoch * 1009 + cycle).shuffle(
                cycle_indices
            )
            take = min(remaining, len(cycle_indices))
            extra.extend(cycle_indices[:take])
            remaining -= take
            cycle += 1

        indices = list(range(self.plan.base_size)) + extra
        random.Random(self.plan.seed + self.epoch * 1009 + 997).shuffle(indices)
        return iter(indices)


def balanced_trainer_class(
    base_trainer: Any, plan: ClassBalancedSamplingPlan | None
) -> Any:
    """只在啟用取樣時覆寫 Trainer sampler，未啟用時保留套件原生行為。"""
    if plan is None:
        return base_trainer
    active_plan = plan

    class ClassBalancedTrainer(base_trainer):
        def _get_train_sampler(
            self, train_dataset: Any = None
        ) -> RotatingMinorityOversampler:
            if self.args.world_size != 1:
                raise RuntimeError(
                    "The rotating class-balanced sampler currently supports single-process training only."
                )
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if dataset is None or len(dataset) != active_plan.base_size:
                raise ValueError(
                    "Training dataset length no longer matches the validated sampling plan."
                )
            return RotatingMinorityOversampler(active_plan)

    return ClassBalancedTrainer


def split_prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    validate_messages(messages)
    return messages[:-1]


def render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    # 模板錯誤應在前置檢查中暴露；靜默改用自製格式會改變訓練分布與遮罩邊界。
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


@dataclass
class TokenizedSftDataset:
    """前置檢查成功後快取 token，讓每個 epoch 使用相同的完整監督目標。

    目前資料量約兩千筆，記憶體快取可避免每次取樣重新分詞；不產生磁碟快取，
    因此更換 tokenizer 或長度設定時，不會誤用舊的編碼結果。
    """

    records: list[dict[str, Any]]
    tokenizer: Any
    max_seq_length: int
    _encoded_records: list[dict[str, list[int]]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_seq_length <= 0:
            raise ValueError("data.max_seq_length must be positive.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if self._encoded_records is not None:
            return self._encoded_records[index]
        return self._encode_record(index)

    def _encode_record(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        messages = record["messages"]
        prompt_messages = split_prompt_messages(messages)

        # prompt_text 只包含 system/user，加上 assistant generation prompt。
        # 它的用途不是餵給模型訓練，而是計算「前面有多少 token 要遮住」。
        prompt_text = render_chat_text(
            tokenizer=self.tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
        )
        # full_text 包含 system/user/assistant，是實際餵給 causal LM 的完整序列。
        # causal LM 仍然會看到 prompt token，只是那些位置不計算 loss。
        full_text = render_chat_text(
            tokenizer=self.tokenizer,
            messages=messages,
            add_generation_prompt=False,
        )

        # 補上 EOS，讓模型學會回答到哪裡應該停止。
        # 若 template 已經自帶 EOS，就不重複附加。
        eos_token = self.tokenizer.eos_token
        if eos_token and not full_text.endswith(eos_token):
            full_text = f"{full_text}{eos_token}"

        # 必須先取得完整長度。右側截斷會破壞 JSON／量刑理由，左側截斷則會丟失
        # 案件事實；兩者都會改變任務語意，因此超長資料交由前置檢查回報處理。
        full_tokens = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )

        input_ids = list(full_tokens["input_ids"])
        attention_mask = list(full_tokens["attention_mask"])
        prompt_ids = list(prompt_tokens["input_ids"])
        prompt_length = len(prompt_ids)
        if len(input_ids) > self.max_seq_length:
            raise ValueError(
                f"完整樣本需要 {len(input_ids)} tokens，超過 max_seq_length={self.max_seq_length}；"
                "未截斷或略過此筆資料，請依訓練資料長度與記憶體限制調整設定。"
            )
        # 分開編碼的 prompt 必須確實是完整輸入的前綴，不能只憑長度猜測遮罩位置。
        if input_ids[:prompt_length] != prompt_ids:
            raise ValueError(
                "Prompt tokens are not a prefix of the full conversation; check the chat template."
            )
        if not input_ids or input_ids[-1] != self.tokenizer.eos_token_id:
            raise ValueError(
                "The complete assistant response must end with the tokenizer EOS token."
            )

        labels = input_ids.copy()
        # Hugging Face Trainer / PyTorch CrossEntropyLoss 會忽略 label = -100 的位置。
        # 所以 prompt 部分設成 -100，assistant 回答部分保留原 token id。
        #
        # 這件事對本任務很重要：
        # 我們要模型學「如何輸出 prediction 與 sentencing_reason」，
        # 而不是把固定 instruction 或案件摘要格式也當成要預測的答案。
        labels[:prompt_length] = [-100] * prompt_length

        if all(label == -100 for label in labels):
            raise ValueError(
                "Tokenized record has no assistant labels. "
                "Increase data.max_seq_length or inspect the source record."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def validate_all(self, split_name: str) -> None:
        """檢查每一筆，不因第一筆通過就放行，也不靜默排除失敗樣本。"""
        encoded: list[dict[str, list[int]]] = []
        errors: list[str] = []
        error_count = 0
        print(f"檢查 {split_name}：共 {len(self)} 筆，保留完整答案。", flush=True)
        for index in range(len(self)):
            try:
                encoded.append(self._encode_record(index))
            except (ValueError, TypeError) as error:
                error_count += 1
                if len(errors) < 5:
                    errors.append(f"{split_name} 第 {index + 1} 筆：{error}")
            if (index + 1) % 250 == 0:
                print(f"  {split_name}：已檢查 {index + 1}/{len(self)} 筆", flush=True)
        if error_count:
            raise ValueError(
                f"{split_name} 有 {error_count} 筆無法用目前設定訓練：\n"
                + "\n".join(errors)
            )
        if not encoded:
            raise ValueError(f"{split_name} is empty.")
        self._encoded_records = encoded
        print(f"{split_name} 檢查完成：{len(encoded)} 筆。", flush=True)


@dataclass
class CausalLmDataCollator:
    # DataCollator 負責把多筆長度不同的樣本組成 batch。
    #
    # tokenizer.pad 會處理 input_ids / attention_mask；
    # labels 不能直接交給 tokenizer.pad，因為 labels 的 padding 值必須是 -100，
    # 否則 padding token 也會被算進 loss。
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        # 動態 padding：每個 batch 只補到該 batch 的最長序列。
        # 這比全資料都補到 max_seq_length 更省記憶體，也比較適合 Mac 本機訓練。
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels: list[list[int]] = []

        for feature in features:
            labels = feature["labels"]
            padding_length = max_length - len(labels)
            padded_labels.append(labels + [-100] * padding_length)

        import torch

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def infer_torch_dtype(torch_module: Any, dtype_name: str | None) -> Any:
    # 模型權重載入精度。
    #
    # CUDA：
    #   - bf16 可用時優先 bf16，通常比 fp16 穩
    #   - 否則使用 fp16 節省 VRAM
    #
    # Apple Silicon / MPS：
    #   - auto 保留 fp32 的保守預設；8B 設定檔另指定 float16 以降低權重記憶體用量。
    #   - 模型載入精度與 Trainer 混合精度是不同設定，兩者不能直接互相推導。
    #
    # CPU：
    #   - 使用 fp32，避免半精度算子支援不足。
    if dtype_name in [None, "auto"]:
        if torch_module.cuda.is_available():
            if torch_module.cuda.is_bf16_supported():
                return torch_module.bfloat16
            return torch_module.float16
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return torch_module.float32
        return torch_module.float32

    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }

    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")

    return dtype_map[dtype_name]


def precision_flags(torch_module: Any, precision: str | None) -> dict[str, bool]:
    # TrainingArguments 的 bf16/fp16 是「Trainer 混合精度訓練旗標」，
    # 和 from_pretrained(dtype=...) 的「模型載入精度」不是同一件事。
    #
    # Transformers 的 fp16/bf16 Trainer 主要針對 CUDA；
    # 在 Mac MPS 上先保持兩者為 False，讓 PyTorch/Trainer 走較保守的路徑。
    if precision in [None, "auto"]:
        if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        if torch_module.cuda.is_available():
            return {"bf16": False, "fp16": True}
        return {"bf16": False, "fp16": False}

    if precision == "bf16":
        return {"bf16": True, "fp16": False}
    if precision == "fp16":
        return {"bf16": False, "fp16": True}
    if precision == "fp32":
        return {"bf16": False, "fp16": False}

    raise ValueError(f"Unsupported training precision: {precision}")


def compact_training_args(
    TrainingArguments: Any,
    raw_args: dict[str, Any],
) -> Any:
    """明確轉換已知的版本差異；未知參數必須報錯，避免設定看似生效卻被捨棄。"""
    signature = inspect.signature(TrainingArguments.__init__)
    supported_keys = set(signature.parameters)
    args = raw_args.copy()
    if "evaluation_strategy" in args and "eval_strategy" in supported_keys:
        args["eval_strategy"] = args.pop("evaluation_strategy")
    if "group_by_length" in args and "train_sampling_strategy" in supported_keys:
        args["train_sampling_strategy"] = (
            "group_by_length" if args.pop("group_by_length") else "random"
        )
    if (
        "warmup_ratio" in args
        and "warmup_ratio" not in supported_keys
        and "warmup_steps" in supported_keys
    ):
        # Transformers 5 使用 warmup_steps 的 [0, 1) 小數表示步數比例。
        # 1.0 會被當成一個 step，因此不能直接映射成「整段訓練都 warmup」。
        ratio = args.pop("warmup_ratio")
        if not 0 <= ratio < 1:
            raise ValueError(
                "training.warmup_ratio must be in [0, 1) for this Transformers version."
            )
        args["warmup_steps"] = ratio
    # 新版 Trainer 固定採 safetensors，移除已不接受、且語意等同預設值的舊開關。
    if (
        "save_safetensors" not in supported_keys
        and args.get("save_safetensors") is True
    ):
        args.pop("save_safetensors")
    unsupported = set(args) - supported_keys
    if unsupported:
        raise ValueError(
            "Installed Transformers does not support these training arguments: "
            + ", ".join(sorted(unsupported))
        )
    return TrainingArguments(**args)


def validate_overwrite_targets(paths: list[Path]) -> None:
    """將遞迴刪除限制在標準 checkpoint／adapter 子目錄，先驗證全部目標再執行。"""
    roots = [
        (PROJECT_ROOT / "sft/outputs" / name).resolve()
        for name in ("checkpoints", "adapters")
    ]
    resolved = [path.resolve() for path in paths]
    for path, target in zip(paths, resolved):
        if path.is_symlink() or not any(
            target != root and target.is_relative_to(root) for root in roots
        ):
            raise ValueError(
                f"Refusing to overwrite outside a checkpoint/adapter run directory: {path}"
            )
    if any(not root.is_relative_to(PROJECT_ROOT.resolve()) for root in roots):
        raise ValueError(
            "Output roots must not resolve outside the project through symlinks."
        )
    if any(
        a == b or a.is_relative_to(b) or b.is_relative_to(a)
        for index, a in enumerate(resolved)
        for b in resolved[index + 1 :]
    ):
        raise ValueError("Checkpoint and adapter directories must not overlap.")


def remove_output_dir(path: Path, overwrite: bool) -> None:
    # 呼叫端必須先通過 dry-run 分支、資料驗證及所有目標的路徑檢查。
    if path.exists() and overwrite:
        validate_overwrite_targets([path])
        shutil.rmtree(path)
        print(f"已清除舊訓練輸出：{path}（未建立備份）", flush=True)


def save_config_snapshot(
    log_dir: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    sampling_plan: ClassBalancedSamplingPlan | None = None,
) -> None:
    # 保存本次實驗設定快照。
    #
    # 訓練跑完幾天後，最常見的問題是「這個 adapter 到底用哪些參數訓練的」。
    # 因此在 log_dir 存一份 JSON snapshot，方便後續比較不同模型或 LoRA rank。
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = log_dir / "train_sft_lora_config_snapshot.json"
    snapshot = {
        "model_config": model_config,
        "training_config": training_config,
        "lora_config": lora_config,
        "computed_sampling_plan": sampling_plan.as_dict() if sampling_plan else None,
    }

    with snapshot_path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)


def apply_optional_training_arg(
    raw_args: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    # run_name 等可選文字欄位未設定時沿用套件預設。
    # trackio_space_id 的 null 代表明確要求本機記錄，必須另行直接傳入。
    if value is not None:
        raw_args[key] = value


def model_access_token(model_config: dict[str, Any]) -> str | None:
    # TAIDE 為 gated 模型：HF token 的帳號仍需先取得模型存取權。
    # 設定檔只保存環境變數名稱，不直接保存 token，避免把憑證寫進專案檔案。
    # 回傳 None 時讓 Hub 沿用自己的登入快取；不能只因環境變數缺少就判定未登入。
    token_env_var = model_config.get("token_env_var")
    if not token_env_var:
        return None

    token = os.environ.get(str(token_env_var))
    if token:
        return token

    return None


def build_model_access_error_message(
    model_name: str,
    model_config: dict[str, Any],
    error: BaseException,
) -> str:
    """依例外鏈的型別與 HTTP 狀態分類，不用錯誤字串猜測帳號是否已登入。"""
    import httpx
    from huggingface_hub.errors import (
        GatedRepoError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    causes: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not item for item in causes):
        causes.append(current)
        current = current.__cause__ or current.__context__
    statuses = {
        getattr(getattr(item, "response", None), "status_code", None) for item in causes
    }
    token_env_var = str(model_config.get("token_env_var", "HF_TOKEN"))
    if any(isinstance(item, GatedRepoError) for item in causes):
        detail = (
            f"模型存取受限，請先以 token 所屬帳號開啟 https://huggingface.co/{model_name}，"
            "同意模型條款並確認已取得存取權；有 token 不代表已通過模型授權。"
            f"再檢查 .env 的 {token_env_var} 或執行 hf auth login。"
        )
    elif statuses & {401, 403}:
        detail = f"HTTP 401/403：伺服器拒絕存取，請檢查 {token_env_var} 的有效性與權限，或執行 hf auth login。"
    elif 429 in statuses:
        detail = "HTTP 429：請求頻率受限，請依伺服器提示稍後再試。"
    elif any(status is not None and status >= 500 for status in statuses):
        detail = "HTTP 5xx：Hub 或下載服務回報錯誤，請稍後重試。"
    elif any(isinstance(item, OfflineModeIsEnabled) for item in causes):
        detail = "目前啟用離線模式；請確認快取完整，或關閉 HF_HUB_OFFLINE 後重試。"
    elif any(
        isinstance(item, (httpx.TransportError, TimeoutError, ConnectionError))
        for item in causes
    ):
        detail = "網路連線失敗或逾時；請檢查 Hub／CDN 連線、代理伺服器與網路狀態，此錯誤不能判定為未登入。"
    elif any(isinstance(item, LocalEntryNotFoundError) for item in causes):
        detail = "本機快取缺少必要檔案；請檢查 local_files_only／離線模式，並在連線恢復後補齊下載。"
    elif 404 in statuses:
        detail = "HTTP 404：找不到模型或必要檔案，請確認模型名稱、路徑與存取權限。"
    elif any(isinstance(item, PermissionError) for item in causes):
        detail = "本機檔案權限不足；請檢查模型或快取目錄的讀寫權限。"
    else:
        detail = "載入失敗，請依原始例外檢查模型檔案、快取或設定；目前沒有足夠資訊判定為授權問題。"
    # 只印模型識別名稱及分類，不輸出 token、HTTP headers 或帶簽章的下載網址。
    return f"Cannot load model assets: {model_name}\n{detail}\n原始例外類型：{type(error).__name__}"


@contextmanager
def loading_progress(stage: str, interval_seconds: float = 15) -> Iterator[None]:
    """以定時訊息呈現等待狀態；這不是下載百分比，也不另開下載或重試。"""
    stopped = Event()
    started = time.monotonic()
    # Hub 的快取鎖等待訊息預設藏在 INFO 層級。只開啟此 logger，讓重複啟動的
    # 程序能顯示等待原因；載入結束後還原，不影響其他套件或後續訓練的日誌設定。
    lock_logger = logging.getLogger("huggingface_hub.utils._fixes")
    previous_level = lock_logger.level
    if lock_logger.getEffectiveLevel() > logging.INFO:
        lock_logger.setLevel(logging.INFO)

    def report_wait() -> None:
        while not stopped.wait(interval_seconds):
            print(
                f"{stage}：已等待 {int(time.monotonic() - started)} 秒，仍在等待回應或處理檔案。",
                flush=True,
            )

    print(f"開始{stage}。", flush=True)
    reporter = Thread(target=report_wait, name="sft-loading-progress", daemon=True)
    reporter.start()
    try:
        yield
    finally:
        # 即使下載失敗或使用者按 Ctrl+C，也要停止提示，避免留下背景執行緒。
        stopped.set()
        reporter.join()
        lock_logger.setLevel(previous_level)


def load_pretrained_component(
    loader: Any, model_config: dict[str, Any], stage: str, **kwargs: Any
) -> Any:
    """共用 tokenizer／權重的載入提示與錯誤分類，保留原始例外供除錯。"""
    import httpx

    model_name = model_config["base_model_name_or_path"]
    print(f"{stage}來源：{model_name}", flush=True)
    if not kwargs.get("local_files_only", False):
        print(
            "需要的檔案會自動下載；Hub 可能重試，單次請求的逾時值不等於總等待上限。",
            flush=True,
        )
    try:
        with loading_progress(stage):
            result = loader.from_pretrained(model_name, **kwargs)
    except (OSError, httpx.HTTPError) as error:
        raise RuntimeError(
            build_model_access_error_message(model_name, model_config, error)
        ) from error
    print(f"{stage}完成。", flush=True)
    return result


def describe_runtime_device(torch_module: Any) -> str:
    # 回報目前訓練可能使用的裝置，特別是 Mac 使用者容易誤以為一定會用到 MPS。
    #
    # 注意：PyTorch 可能「有編進 MPS 支援」但「目前不可用」。
    # 例如驅動、安裝來源、執行環境限制都可能讓 is_available() 回傳 False。
    if torch_module.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def print_runtime_summary(torch_module: Any, torch_dtype: Any) -> None:
    # 啟動訓練前印出裝置與精度，讓使用者能立刻確認是否真的跑在 GPU/MPS 上。
    device = describe_runtime_device(torch_module)
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    print("=" * 80)
    print("Runtime")
    print("=" * 80)
    print(f"torch version: {torch_module.__version__}")
    print(f"selected device: {device}")
    print(f"mps built: {mps_built}")
    print(f"mps available: {mps_available}")
    print(f"mps fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '')}")
    print(f"hf hub disable xet: {os.environ.get('HF_HUB_DISABLE_XET', '')}")
    print(f"hf hub download timeout: {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '')}")
    print(f"model torch dtype: {torch_dtype}")

    if device == "cpu" and mps_built and not mps_available:
        print(
            "注意：目前 PyTorch 有 MPS 支援，但這個執行環境回報 MPS 不可用；"
            "訓練會退回 CPU，速度會慢很多。"
        )


def build_training_arguments(
    TrainingArguments: Any,
    torch: Any,
    trainer_config: dict[str, Any],
    checkpoint_dir: Path,
    report_to: str | list[str],
) -> Any:
    """在任何模型下載及清除輸出之前驗證訓練設定。"""
    precision = precision_flags(torch, trainer_config.get("precision", "auto"))
    raw_args = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": float(trainer_config.get("num_train_epochs", 3)),
        "max_steps": int(trainer_config.get("max_steps", -1)),
        "per_device_train_batch_size": int(
            trainer_config.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            trainer_config.get("per_device_eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            trainer_config.get("gradient_accumulation_steps", 16)
        ),
        "learning_rate": float(trainer_config.get("learning_rate", 1e-4)),
        "weight_decay": float(trainer_config.get("weight_decay", 0.0)),
        "warmup_ratio": float(trainer_config.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(trainer_config.get("lr_scheduler_type", "cosine")),
        "max_grad_norm": float(trainer_config.get("max_grad_norm", 0.3)),
        "logging_steps": int(trainer_config.get("logging_steps", 10)),
        "eval_steps": int(trainer_config.get("eval_steps", 100)),
        "save_steps": int(trainer_config.get("save_steps", 100)),
        "save_total_limit": int(trainer_config.get("save_total_limit", 3)),
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "group_by_length": bool(trainer_config.get("group_by_length", True)),
        "dataloader_num_workers": int(trainer_config.get("dataloader_num_workers", 0)),
        "report_to": report_to,
        "trackio_space_id": trainer_config.get("trackio_space_id"),
        "remove_unused_columns": False,
        "seed": int(trainer_config.get("seed", 42)),
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": bool(
            trainer_config.get("gradient_checkpointing", True)
        ),
        "save_safetensors": True,
    }
    # null 必須保留，避免舊版 Trackio 預設值將本機實驗指向遠端 Space。
    # CUDA 才使用 pinned memory，MPS 與 CPU 不需要這項資料傳輸最佳化。
    raw_args["dataloader_pin_memory"] = torch.cuda.is_available()
    for key in ("run_name", "project"):
        apply_optional_training_arg(raw_args, key, trainer_config.get(key))
    return compact_training_args(TrainingArguments, raw_args)


def final_eval_metrics_from_history(trainer: Any) -> dict[str, Any] | None:
    """重用 Trainer 已在最後一步完成的評估，避免相同 validation 再跑一次。

    Transformers 5 會在 step-based evaluation 的最後一步不是 eval_steps 整數倍時，
    主動補做一次 evaluation；較舊版本未必如此。這裡只接受與目前 global_step 完全
    相同的 eval 紀錄，否則回傳 None，讓呼叫端明確執行 trainer.evaluate()。
    """
    state = getattr(trainer, "state", None)
    global_step = getattr(state, "global_step", None)
    history = getattr(state, "log_history", None)
    if (
        not isinstance(global_step, int)
        or global_step <= 0
        or not isinstance(history, list)
    ):
        return None

    for record in reversed(history):
        if not isinstance(record, dict) or record.get("step") != global_step:
            continue
        metrics = {
            key: value
            for key, value in record.items()
            if key == "epoch" or key.startswith("eval_")
        }
        if "eval_loss" in metrics:
            return metrics
    return None


def main() -> None:
    # 主流程刻意維持線性，方便第一次看訓練流程的人從上往下理解：
    #   1. 讀設定
    #   2. 讀資料
    #   3. 建 tokenizer / dataset
    #   4. 建 model / LoRA
    #   5. 建 Trainer 並開始訓練
    args = parse_args()
    # 除了命令列互斥檢查，也保護由測試或其他 Python 程式直接呼叫 main 的情境。
    if args.dry_run and args.overwrite_output_dir:
        raise ValueError("--dry-run cannot be combined with --overwrite-output-dir.")
    load_project_env()
    configure_runtime_environment()
    deps = require_training_dependencies()

    torch = deps["torch"]
    yaml = deps["yaml"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    model_config = load_yaml(args.model_config, yaml)
    training_config = load_yaml(args.training_config, yaml)
    lora_config = load_yaml(args.lora_config, yaml)

    data_config = training_config.get("data", {})
    output_config = training_config.get("output", {})
    trainer_config = training_config.get("training", {})
    report_to = normalize_report_to(trainer_config.get("report_to", "none"))
    require_tracking_dependencies(report_to)

    # 將 YAML 裡的相對路徑解析成專案內的實際路徑。
    # test_file 只檢查格式與編碼契約，不交給 Trainer；正式測試集評估需另行實作。
    train_file = project_path(data_config["train_file"])
    validation_file = project_path(data_config["validation_file"])
    test_file = project_path(data_config["test_file"])
    max_seq_length = int(data_config.get("max_seq_length", 1536))

    checkpoint_dir = project_path(output_config["checkpoint_dir"])
    adapter_dir = project_path(output_config["adapter_dir"])
    log_dir = project_path(output_config["log_dir"])

    if args.overwrite_output_dir:
        if trainer_config.get("resume_from_checkpoint"):
            raise ValueError(
                "Cannot overwrite outputs when resume_from_checkpoint is enabled."
            )
        validate_overwrite_targets([checkpoint_dir, adapter_dir])

    training_args = build_training_arguments(
        TrainingArguments, torch, trainer_config, checkpoint_dir, report_to
    )
    torch_dtype = infer_torch_dtype(torch, model_config.get("torch_dtype"))
    print_runtime_summary(torch, torch_dtype)

    # 先讀三份資料，讓路徑或 JSONL 格式錯誤在模型下載前就被發現。
    # 這對第一次跑 Hugging Face 模型尤其重要，避免等下載完才發現資料路徑錯。
    train_records = load_jsonl(train_file)
    validation_records = load_jsonl(validation_file)
    # test set 不拿來訓練或調參；後續長度檢查也使用事先決定的相同上限。
    test_records = load_jsonl(test_file)

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Test records: {len(test_records)}", flush=True)

    sampling_plan = build_class_balanced_sampling_plan(
        train_records,
        trainer_config.get("class_balanced_sampling"),
        default_seed=int(trainer_config.get("seed", 42)),
    )
    if sampling_plan is not None:
        print(
            "Class-balanced sampling: "
            f"{sampling_plan.minority_label} {sampling_plan.original_minority_count} -> "
            f"{sampling_plan.target_minority_count}; extra={sampling_plan.extra_minority_count}; "
            f"epoch_size={sampling_plan.epoch_size}; "
            f"fraction={sampling_plan.achieved_fraction:.4f}; seed={sampling_plan.seed}",
            flush=True,
        )

    local_files_only = bool(model_config.get("local_files_only", False))
    token = model_access_token(model_config)

    # Tokenizer、chat template 與權重必須來自同一個 base model。
    # TAIDE 的詞彙表與其他 Llama 版本可能不同；不能以已快取的舊 tokenizer 代替。
    tokenizer = load_pretrained_component(
        AutoTokenizer,
        model_config,
        "載入 tokenizer",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_fast=bool(model_config.get("use_fast_tokenizer", True)),
        local_files_only=local_files_only,
        token=token,
    )

    # 部分 causal LM 沒有獨立 pad token；訓練時用 eos token 當 pad token 是常見做法。
    # padding_side 設成 right，對 causal LM batch training 較直覺，也符合多數 SFT 範例。
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token before training.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 三個集合都驗證相同的資料契約；test 不進入 Trainer，也不拿來選參數或算訓練 loss。
    # 逐一彙整各集合的錯誤，避免 train 有問題時完全看不到 validation／test 的狀態。
    datasets = {}
    validation_errors = []
    for split_name, records in (
        ("train", train_records),
        ("validation", validation_records),
        ("test", test_records),
    ):
        dataset = TokenizedSftDataset(records, tokenizer, max_seq_length)
        try:
            dataset.validate_all(split_name)
        except ValueError as error:
            validation_errors.append(str(error))
        datasets[split_name] = dataset
    if validation_errors:
        raise ValueError(
            "SFT 前置檢查失敗，尚未載入權重或更動訓練輸出：\n"
            + "\n".join(validation_errors)
        )
    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]
    del datasets["test"], dataset

    preview = train_dataset[0]
    print(
        "First tokenized sample: "
        f"{len(preview['input_ids'])} tokens, "
        f"{sum(label != -100 for label in preview['labels'])} trainable labels"
    )

    if args.dry_run:
        print(
            "Dry run completed: all splits validated; no model weights loaded or training outputs changed."
        )
        return

    # Trainer 建立時才設 seed 會晚於 LoRA 初始化；先設 seed 才能重現 adapter 初始值。
    deps["set_seed"](training_args.seed)

    # from_pretrained 只負責載入 base model。
    # LoRA adapter 會在後面用 get_peft_model 包上去，因此此時還不會改動原模型權重。
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "dtype": torch_dtype,
        "low_cpu_mem_usage": bool(model_config.get("low_cpu_mem_usage", True)),
        "local_files_only": local_files_only,
    }
    if token is not None:
        model_kwargs["token"] = token

    device_map = model_config.get("device_map")
    if device_map is not None:
        # device_map 主要給 CUDA / Accelerate 分散載入使用。
        # 單機 MPS 訓練通常保持 null，交由 Trainer/PyTorch 處理裝置搬移。
        model_kwargs["device_map"] = device_map

    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        # 例如 flash_attention_2；Mac MPS 通常不要開，CUDA 環境確認支援後再調。
        model_kwargs["attn_implementation"] = attn_implementation

    model = load_pretrained_component(
        AutoModelForCausalLM, model_config, "載入模型權重", **model_kwargs
    )

    if bool(trainer_config.get("gradient_checkpointing", True)):
        # gradient checkpointing 會在反向傳播時重算部分 activation，以換取較低記憶體占用。
        # 對 M3 Max 這類 unified memory 機器，這通常能降低記憶體壓力；
        # 代價是訓練會慢一些。初版保守開啟。
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA 只訓練少量低秩矩陣，不更新 base model 大部分權重。
    # target_modules 對不同模型家族可能不同。
    # TAIDE Llama 3.1 採 Llama-family 命名：q/k/v/o projection 與 gate/up/down MLP projection。
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("lora_alpha", 32)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        task_type=str(lora_config.get("task_type", "CAUSAL_LM")),
        fan_in_fan_out=bool(lora_config.get("fan_in_fan_out", False)),
        target_modules=list(lora_config.get("target_modules", [])),
        modules_to_save=lora_config.get("modules_to_save"),
        inference_mode=bool(lora_config.get("inference_mode", False)),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 到此設定、全部資料與模型都已準備完成，才允許清除使用者明確指定的舊輸出。
    # dry-run 已在上方返回；resume 與 overwrite 也已在任何刪除前檢查為互斥。
    remove_output_dir(checkpoint_dir, args.overwrite_output_dir)
    remove_output_dir(adapter_dir, args.overwrite_output_dir)

    TrainerClass = balanced_trainer_class(Trainer, sampling_plan)
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLmDataCollator(tokenizer=tokenizer),
    )

    save_config_snapshot(
        log_dir=log_dir,
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        sampling_plan=sampling_plan,
    )

    resume_from_checkpoint = trainer_config.get("resume_from_checkpoint")
    # checkpoint_dir 會保存 Trainer 的中間 checkpoint；
    # adapter_dir 則保存最終 LoRA adapter，之後 inference/merge 會主要使用 adapter_dir。
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = final_eval_metrics_from_history(trainer)
    if eval_metrics is None:
        eval_metrics = trainer.evaluate()
    else:
        print(
            "Final validation was already evaluated by Trainer; reusing its metrics.",
            flush=True,
        )
    trainer.save_metrics("eval", eval_metrics)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"LoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
