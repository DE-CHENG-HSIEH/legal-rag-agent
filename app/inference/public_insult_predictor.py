"""載入 TAIDE LoRA adapters，執行公然侮辱量刑群組預測。

重型相依套件與模型權重都採延遲載入。一般法律檢索不會初始化 8B 模型；
第一次真正呼叫預測工具時才會下載缺少的檔案，後續呼叫則重用同一個模型實例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


DEFAULT_BASE_MODEL_ID = "taide/Llama-3.1-TAIDE-LX-8B-Chat"
# 此 commit 來自訓練機 Hugging Face cache 的 refs/main；固定它可避免上游 main 更新後改變結果。
DEFAULT_BASE_MODEL_REVISION = "6738923786667dfe00363b1e2d225536817b429e"
PUBLISHED_ADAPTER_A_ID = "a97141481/Llama-3.1-TAIDE-Public-Insult-A-LoRA"
PUBLISHED_ADAPTER_B_ID = "a97141481/Llama-3.1-TAIDE-Public-Insult-B-LoRA"
PUBLISHED_ADAPTER_C_ID = "a97141481/Llama-3.1-TAIDE-Public-Insult-C-LoRA"
PUBLISHED_ADAPTER_REVISION = "v1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ADAPTER_A_PATH = PROJECT_ROOT / "sft/outputs/adapters/public_insult_v2_ablation_a"
LOCAL_ADAPTER_B_PATH = PROJECT_ROOT / "sft/outputs/adapters/public_insult_v2_ablation_b"
LOCAL_ADAPTER_C_PATH = PROJECT_ROOT / "sft/outputs/adapters/public_insult_v2_ablation_c"
DEFAULT_ADAPTER_A_ID = (
    str(LOCAL_ADAPTER_A_PATH)
    if LOCAL_ADAPTER_A_PATH.is_dir()
    else PUBLISHED_ADAPTER_A_ID
)
DEFAULT_ADAPTER_A_REVISION = (
    None if LOCAL_ADAPTER_A_PATH.is_dir() else PUBLISHED_ADAPTER_REVISION
)
DEFAULT_ADAPTER_B_ID = (
    str(LOCAL_ADAPTER_B_PATH)
    if LOCAL_ADAPTER_B_PATH.is_dir()
    else PUBLISHED_ADAPTER_B_ID
)
DEFAULT_ADAPTER_B_REVISION = (
    None if LOCAL_ADAPTER_B_PATH.is_dir() else PUBLISHED_ADAPTER_REVISION
)
DEFAULT_ADAPTER_C_ID = (
    str(LOCAL_ADAPTER_C_PATH)
    if LOCAL_ADAPTER_C_PATH.is_dir()
    else PUBLISHED_ADAPTER_C_ID
)
DEFAULT_ADAPTER_C_REVISION = (
    None if LOCAL_ADAPTER_C_PATH.is_dir() else PUBLISHED_ADAPTER_REVISION
)
TRAINING_MAX_SEQUENCE_LENGTH = 1536

InputVariant = Literal["A", "B", "C"]
INPUT_VARIANTS: tuple[InputVariant, ...] = ("A", "B", "C")
ADDITIONAL_INPUT_VARIANTS: tuple[InputVariant, ...] = ("B", "C")

TRAINING_COURTS = frozenset(
    {
        "臺灣臺北地方法院",
        "臺灣新北地方法院",
        "臺灣士林地方法院",
    }
)
TRAINING_YEAR_RANGE = range(104, 115)

AMOUNT_BANDS: dict[str, dict[str, Any]] = {
    "fine_1000_3000": {
        "sentence_type": "罰金",
        "minimum": 1000,
        "maximum": 3000,
        "unit": "新臺幣元",
    },
    "fine_4000_5000": {
        "sentence_type": "罰金",
        "minimum": 4000,
        "maximum": 5000,
        "unit": "新臺幣元",
    },
    "fine_6000_plus": {
        "sentence_type": "罰金",
        "minimum": 6000,
        "maximum": None,
        "unit": "新臺幣元",
    },
    "detention_1_15": {
        "sentence_type": "拘役",
        "minimum": 1,
        "maximum": 15,
        "unit": "日",
    },
    "detention_16_20": {
        "sentence_type": "拘役",
        "minimum": 16,
        "maximum": 20,
        "unit": "日",
    },
    "detention_21_plus": {
        "sentence_type": "拘役",
        "minimum": 21,
        "maximum": None,
        "unit": "日",
    },
}

SYSTEM_PROMPT = """你是臺灣刑事公然侮辱案件之判決結果預測與量刑理由生成模型。請依輸入之客觀犯罪事實與客觀量刑因素，輸出合法JSON，不得宣稱結果代表法院必然判決。

prediction必須包含sentence_type、sentence_amount_band、sentence_amount_min、sentence_amount_max與sentence_amount_unit。sentence_type只能是「拘役」或「罰金」。sentence_amount_band只能是fine_1000_3000、fine_4000_5000、fine_6000_plus、detention_1_15、detention_16_20、detention_21_plus之一，且必須與刑種、上下限及單位一致；無上限區間的sentence_amount_max使用null。最外層另須包含非空白的sentencing_reason。"""


class PublicInsultPredictorError(RuntimeError):
    """模型載入、生成或輸出契約失敗。"""


class PublicInsultPredictionCancelled(PublicInsultPredictorError):
    """使用者停止本輪 TAIDE LoRA 推論。"""


class SentencePrediction(BaseModel):
    """LoRA adapter 的 prediction 欄位契約。"""

    model_config = ConfigDict(extra="forbid")

    sentence_type: Literal["拘役", "罰金"]
    sentence_amount_band: str
    sentence_amount_min: int
    sentence_amount_max: int | None
    sentence_amount_unit: Literal["日", "新臺幣元"]

    @model_validator(mode="after")
    def validate_amount_band_contract(self) -> "SentencePrediction":
        expected = AMOUNT_BANDS.get(self.sentence_amount_band)
        if expected is None:
            raise ValueError(f"未知刑度群組：{self.sentence_amount_band}")

        actual = {
            "sentence_type": self.sentence_type,
            "minimum": self.sentence_amount_min,
            "maximum": self.sentence_amount_max,
            "unit": self.sentence_amount_unit,
        }
        if actual != expected:
            raise ValueError(
                f"刑度群組 {self.sentence_amount_band} 與刑種、上下限或單位不一致。"
            )
        return self


class PublicInsultModelOutput(BaseModel):
    """模型最外層 JSON 契約。"""

    model_config = ConfigDict(extra="forbid")

    prediction: SentencePrediction
    sentencing_reason: str = Field(min_length=1)

    @field_validator("sentencing_reason")
    @classmethod
    def reject_blank_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sentencing_reason 不得為空白。")
        return value


@dataclass(frozen=True)
class PredictorSettings:
    """可由環境變數覆寫的 A、B、C 組部署設定。"""

    base_model_id: str = DEFAULT_BASE_MODEL_ID
    base_model_revision: str = DEFAULT_BASE_MODEL_REVISION
    adapter_a_id: str = DEFAULT_ADAPTER_A_ID
    adapter_a_revision: str | None = DEFAULT_ADAPTER_A_REVISION
    adapter_b_id: str = DEFAULT_ADAPTER_B_ID
    adapter_b_revision: str | None = DEFAULT_ADAPTER_B_REVISION
    adapter_c_id: str = DEFAULT_ADAPTER_C_ID
    adapter_c_revision: str | None = DEFAULT_ADAPTER_C_REVISION
    device: str = "auto"
    max_input_tokens: int = 1024
    max_new_tokens: int = 512
    hf_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "PredictorSettings":
        """Load deployment overrides and reject token budgets unseen in training."""

        settings = cls(
            base_model_id=os.getenv(
                "PUBLIC_INSULT_BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID
            ),
            base_model_revision=os.getenv(
                "PUBLIC_INSULT_BASE_MODEL_REVISION",
                DEFAULT_BASE_MODEL_REVISION,
            ),
            adapter_a_id=os.getenv(
                "PUBLIC_INSULT_ADAPTER_A_ID",
                DEFAULT_ADAPTER_A_ID,
            ),
            adapter_a_revision=(
                os.getenv("PUBLIC_INSULT_ADAPTER_A_REVISION")
                or DEFAULT_ADAPTER_A_REVISION
            ),
            adapter_b_id=os.getenv(
                "PUBLIC_INSULT_ADAPTER_B_ID",
                DEFAULT_ADAPTER_B_ID,
            ),
            adapter_b_revision=(
                os.getenv("PUBLIC_INSULT_ADAPTER_B_REVISION")
                or DEFAULT_ADAPTER_B_REVISION
            ),
            adapter_c_id=(
                os.getenv("PUBLIC_INSULT_ADAPTER_C_ID") or DEFAULT_ADAPTER_C_ID
            ),
            adapter_c_revision=(
                os.getenv("PUBLIC_INSULT_ADAPTER_C_REVISION")
                or DEFAULT_ADAPTER_C_REVISION
            ),
            device=os.getenv("PUBLIC_INSULT_DEVICE", "auto").strip().lower(),
            max_input_tokens=_positive_int_env("PUBLIC_INSULT_MAX_INPUT_TOKENS", 1024),
            max_new_tokens=_positive_int_env("PUBLIC_INSULT_MAX_NEW_TOKENS", 512),
            hf_token=os.getenv("HF_TOKEN") or None,
        )
        if (
            settings.max_input_tokens + settings.max_new_tokens
            > TRAINING_MAX_SEQUENCE_LENGTH
        ):
            raise PublicInsultPredictorError(
                "PUBLIC_INSULT_MAX_INPUT_TOKENS 與 PUBLIC_INSULT_MAX_NEW_TOKENS 的總和"
                f"不得超過訓練長度 {TRAINING_MAX_SEQUENCE_LENGTH}。"
            )
        return settings

    def adapter_source(
        self,
        variant: InputVariant,
    ) -> tuple[str, str | None]:
        """Return the repository or local path and immutable revision for a variant."""

        return {
            "A": (
                self.adapter_a_id,
                self.adapter_a_revision,
            ),
            "B": (
                self.adapter_b_id,
                self.adapter_b_revision,
            ),
            "C": (
                self.adapter_c_id,
                self.adapter_c_revision,
            ),
        }[variant]


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise PublicInsultPredictorError(f"{name} 必須是正整數。") from error
    if value <= 0:
        raise PublicInsultPredictorError(f"{name} 必須是正整數。")
    return value


def _required_text(label: str, value: str, *, maximum_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不得為空白。")
    normalized = value.strip()
    if len(normalized) > maximum_length:
        raise ValueError(f"{label}超過允許長度 {maximum_length} 個字元。")
    return normalized


def _validated_optional_year(
    judgment_year_roc: int | None,
) -> int | None:
    if judgment_year_roc is None:
        return None
    if isinstance(judgment_year_roc, bool) or not isinstance(judgment_year_roc, int):
        raise ValueError("裁判年份必須是民國年整數，例如 114。")
    if not 1 <= judgment_year_roc <= 999:
        raise ValueError("裁判年份必須是 1 到 999 的民國年整數。")
    return judgment_year_roc


def _optional_text(
    label: str,
    value: str | None,
    *,
    maximum_length: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label}必須是文字。")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > maximum_length:
        raise ValueError(f"{label}超過允許長度 {maximum_length} 個字元。")
    return normalized


def select_input_variant(
    *,
    court: str | None = None,
    judgment_year_roc: int | None = None,
    judge_name: str | None = None,
) -> InputVariant:
    """依已知制度資訊選擇最高且完整的 A、B 或 C 組輸入。"""

    normalized_court = _optional_text(
        "法院",
        court,
        maximum_length=100,
    )
    normalized_judge = _optional_text(
        "承審法官",
        judge_name,
        maximum_length=100,
    )
    normalized_year = _validated_optional_year(judgment_year_roc)

    if (
        normalized_court is not None
        and normalized_year is not None
        and normalized_judge is not None
    ):
        return "C"

    if normalized_court is not None and normalized_year is not None:
        return "B"

    return "A"


def _build_messages(
    *,
    criminal_facts: str,
    sentencing_factors: str,
    institutional_lines: list[str] | None = None,
) -> list[dict[str, str]]:
    """建立與 A、B、C 組訓練資料一致的共同訊息模板。"""

    facts = _required_text("犯罪事實", criminal_facts, maximum_length=8000)
    factors = _required_text("量刑因素", sentencing_factors, maximum_length=8000)

    user_prompt = f"""請根據以下資料，產生本件公然侮辱案件的判決結果與量刑理由段落。

【客觀犯罪事實摘要】
{facts}

【客觀量刑因素摘要】
{factors}"""

    if institutional_lines:
        user_prompt += "\n\n【案件制度資訊】\n" + "\n".join(institutional_lines)

    user_prompt += "\n\n請輸出JSON，欄位包含prediction與sentencing_reason。"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_variant_a_messages(
    *,
    criminal_facts: str,
    sentencing_factors: str,
) -> list[dict[str, str]]:
    """以訓練資料的 A 組模板建立訊息。"""

    return _build_messages(
        criminal_facts=criminal_facts,
        sentencing_factors=sentencing_factors,
    )


def build_variant_b_messages(
    *,
    criminal_facts: str,
    sentencing_factors: str,
    court: str,
    judgment_year_roc: int,
) -> list[dict[str, str]]:
    """以訓練資料的 B 組模板建立訊息。"""

    normalized_court = _required_text(
        "法院",
        court,
        maximum_length=100,
    )
    normalized_year = _validated_optional_year(judgment_year_roc)
    assert normalized_year is not None

    return _build_messages(
        criminal_facts=criminal_facts,
        sentencing_factors=sentencing_factors,
        institutional_lines=[
            f"法院：{normalized_court}",
            f"裁判年份：民國{normalized_year}年",
        ],
    )


def build_variant_c_messages(
    *,
    criminal_facts: str,
    sentencing_factors: str,
    court: str,
    judgment_year_roc: int,
    judge_name: str,
) -> list[dict[str, str]]:
    """以訓練資料的 C 組模板建立訊息。"""

    normalized_court = _required_text(
        "法院",
        court,
        maximum_length=100,
    )
    normalized_year = _validated_optional_year(judgment_year_roc)
    assert normalized_year is not None
    normalized_judge = _required_text(
        "承審法官",
        judge_name,
        maximum_length=100,
    )

    return _build_messages(
        criminal_facts=criminal_facts,
        sentencing_factors=sentencing_factors,
        institutional_lines=[
            f"法院：{normalized_court}",
            f"裁判年份：民國{normalized_year}年",
            f"承審法官：{normalized_judge}",
        ],
    )


def build_prediction_messages(
    *,
    criminal_facts: str,
    sentencing_factors: str,
    court: str | None = None,
    judgment_year_roc: int | None = None,
    judge_name: str | None = None,
) -> tuple[InputVariant, list[dict[str, str]]]:
    """依資料完整度路由並建立對應訓練版本的訊息。"""

    variant = select_input_variant(
        court=court,
        judgment_year_roc=judgment_year_roc,
        judge_name=judge_name,
    )

    if variant == "C":
        assert court is not None
        assert judgment_year_roc is not None
        assert judge_name is not None
        messages = build_variant_c_messages(
            criminal_facts=criminal_facts,
            sentencing_factors=sentencing_factors,
            court=court,
            judgment_year_roc=judgment_year_roc,
            judge_name=judge_name,
        )
    elif variant == "B":
        assert court is not None
        assert judgment_year_roc is not None
        messages = build_variant_b_messages(
            criminal_facts=criminal_facts,
            sentencing_factors=sentencing_factors,
            court=court,
            judgment_year_roc=judgment_year_roc,
        )
    else:
        messages = build_variant_a_messages(
            criminal_facts=criminal_facts,
            sentencing_factors=sentencing_factors,
        )

    return variant, messages


def validate_generated_output(generated_text: str) -> PublicInsultModelOutput:
    """只接受完整且嚴格符合 schema 的 JSON；不從說明文字中猜測或擷取片段。"""
    try:
        payload = json.loads(generated_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise PublicInsultPredictorError("量刑模型未輸出合法 JSON。") from error
    try:
        return PublicInsultModelOutput.model_validate(payload)
    except ValidationError as error:
        raise PublicInsultPredictorError(
            "量刑模型輸出不符合既定 JSON schema。"
        ) from error


def prediction_scope_warnings(
    *,
    court: str | None = None,
    judgment_year_roc: int | None = None,
) -> list[str]:
    """標示超出訓練法院或年度的輸入，但保留研究用途的預測能力。"""

    warnings: list[str] = []
    normalized_court = _optional_text(
        "法院",
        court,
        maximum_length=100,
    )
    normalized_year = _validated_optional_year(judgment_year_roc)

    if normalized_court is not None and normalized_court not in TRAINING_COURTS:
        warnings.append("法院不在三所訓練法院內，屬於分布外輸入。")
    if normalized_year is not None and normalized_year not in TRAINING_YEAR_RANGE:
        warnings.append("裁判年份不在民國 104 至 114 年的訓練範圍內，屬於分布外輸入。")
    return warnings


def routing_warnings(
    *,
    variant: InputVariant,
    court: str | None = None,
    judgment_year_roc: int | None = None,
    judge_name: str | None = None,
) -> list[str]:
    """說明因欄位組合不完整而未納入模型的選填資訊。"""

    warnings: list[str] = []
    has_court = (
        _optional_text(
            "法院",
            court,
            maximum_length=100,
        )
        is not None
    )
    has_year = _validated_optional_year(judgment_year_roc) is not None
    has_judge = (
        _optional_text(
            "承審法官",
            judge_name,
            maximum_length=100,
        )
        is not None
    )

    if variant == "A" and (has_court or has_year):
        warnings.append(
            "法院與裁判年份必須同時提供才能使用 B／C 組；"
            "本次未將零散制度資訊輸入模型。"
        )
    if variant != "C" and has_judge:
        warnings.append(
            "承審法官必須與法院及裁判年份一併提供才能使用 C 組；"
            "本次未將法官姓名輸入模型。"
        )

    return warnings


class PublicInsultPredictor:
    """執行緒安全的延遲載入推論服務。"""

    def __init__(self, settings: PredictorSettings | None = None) -> None:
        """Create an unloaded service; heavyweight model state is initialized on demand."""

        self.settings = settings or PredictorSettings.from_environment()
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Report whether the shared TAIDE base and all adapters are ready."""

        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            self._load_model()

    def _load_model(self) -> None:
        # 必須在 import torch 前設定 fallback，否則部分 MPS 尚未支援的運算無法回退 CPU。
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise PublicInsultPredictorError(
                "缺少 TAIDE 推論套件，請安裝 torch、transformers、peft、accelerate 與 safetensors。"
            ) from error

        device = _select_device(torch, self.settings.device)
        dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
        hub_kwargs: dict[str, Any] = {}
        if self.settings.hf_token:
            hub_kwargs["token"] = self.settings.hf_token

        try:
            adapter_configs = {}
            for variant in INPUT_VARIANTS:
                adapter_id, adapter_revision = self.settings.adapter_source(variant)
                adapter_kwargs = dict(hub_kwargs)
                if adapter_revision is not None:
                    adapter_kwargs["revision"] = adapter_revision
                adapter_config = PeftConfig.from_pretrained(
                    adapter_id,
                    **adapter_kwargs,
                )
                if (
                    adapter_config.base_model_name_or_path
                    != self.settings.base_model_id
                ):
                    raise PublicInsultPredictorError(
                        f"{variant} 組 Adapter 宣告的基礎模型與部署設定不一致，"
                        "已停止載入以避免混用權重。"
                    )
                adapter_configs[variant] = adapter_config

            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.base_model_id,
                revision=self.settings.base_model_revision,
                use_fast=True,
                trust_remote_code=False,
                **hub_kwargs,
            )
            if tokenizer.eos_token_id is None:
                raise PublicInsultPredictorError(
                    "TAIDE tokenizer 未定義 eos_token_id。"
                )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            base_model = AutoModelForCausalLM.from_pretrained(
                self.settings.base_model_id,
                revision=self.settings.base_model_revision,
                dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                **hub_kwargs,
            )
            adapter_a_id, adapter_a_revision = self.settings.adapter_source("A")
            adapter_a_kwargs = dict(hub_kwargs)
            if adapter_a_revision is not None:
                adapter_a_kwargs["revision"] = adapter_a_revision

            # 單一 TAIDE base 掛載三個具名 adapter；路由只切換 adapter，
            # 不複製 8B base weights，兼顧記憶體用量與切換延遲。
            model = PeftModel.from_pretrained(
                base_model,
                adapter_a_id,
                adapter_name="A",
                config=adapter_configs["A"],
                is_trainable=False,
                **adapter_a_kwargs,
            )

            for variant in ADDITIONAL_INPUT_VARIANTS:
                adapter_id, adapter_revision = self.settings.adapter_source(variant)
                adapter_kwargs = dict(hub_kwargs)
                if adapter_revision is not None:
                    adapter_kwargs["revision"] = adapter_revision
                model.load_adapter(
                    adapter_id,
                    adapter_name=variant,
                    is_trainable=False,
                    **adapter_kwargs,
                )

            model.set_adapter("A")
            model.to(device)
            model.eval()
            model.config.use_cache = True
        except PublicInsultPredictorError:
            raise
        except Exception as error:
            raise PublicInsultPredictorError(
                "無法載入 TAIDE 基礎模型或 LoRA adapter；請檢查 HF_TOKEN、"
                "TAIDE 存取權、網路連線與本機記憶體。"
            ) from error

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def predict(
        self,
        *,
        criminal_facts: str,
        sentencing_factors: str,
        court: str | None = None,
        judgment_year_roc: int | None = None,
        judge_name: str | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Route one case to A/B/C, generate greedily, and validate the JSON contract."""

        variant, messages = build_prediction_messages(
            criminal_facts=criminal_facts,
            sentencing_factors=sentencing_factors,
            court=court,
            judgment_year_roc=judgment_year_roc,
            judge_name=judge_name,
        )
        if cancellation_event is not None and cancellation_event.is_set():
            raise PublicInsultPredictionCancelled("TAIDE LoRA 量刑預測已停止。")

        self._ensure_loaded()
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens > self.settings.max_input_tokens:
            raise ValueError(
                f"模型輸入共有 {input_tokens} tokens，超過上限 "
                f"{self.settings.max_input_tokens}；系統不會自動截斷法律事實，請先精簡內容。"
            )
        encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}

        generation_kwargs: dict[str, Any] = {}
        if cancellation_event is not None:
            active_cancellation_event = cancellation_event
            from transformers import (
                StoppingCriteria,
                StoppingCriteriaList,
            )

            class CancellationCriteria(StoppingCriteria):
                def __call__(
                    self,
                    input_ids,
                    scores,
                    **kwargs,
                ) -> bool:
                    """Stop generation at the next token boundary after a UI request."""

                    return active_cancellation_event.is_set()

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [CancellationCriteria()]
            )

        started_at = time.monotonic()
        with self._generation_lock, self._torch.inference_mode():
            self._model.set_adapter(variant)
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                **generation_kwargs,
            )

        if cancellation_event is not None and cancellation_event.is_set():
            raise PublicInsultPredictionCancelled("TAIDE LoRA 量刑預測已停止。")
        elapsed = time.monotonic() - started_at
        continuation = generated[0, input_tokens:]
        generated_text = self._tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        validated = validate_generated_output(generated_text)

        adapter_id, adapter_revision = self.settings.adapter_source(variant)
        routing_reason = {
            "A": ("未取得完整法院與裁判年份，使用僅需犯罪事實及量刑因素的訴前模式。"),
            "B": ("已取得法院與裁判年份，但未取得承審法官，使用法院年份模式。"),
            "C": ("法院、裁判年份與承審法官均已提供，使用完整制度資訊模式。"),
        }[variant]

        return {
            "status": "ok",
            "model": {
                "base_model": self.settings.base_model_id,
                "base_revision": self.settings.base_model_revision,
                "adapter": adapter_id,
                "adapter_revision": adapter_revision,
                "input_variant": variant,
                "device": str(self._device),
            },
            "routing": {
                "input_variant": variant,
                "reason": routing_reason,
                "used_optional_fields": {
                    "court": variant in {"B", "C"},
                    "judgment_year_roc": variant in {"B", "C"},
                    "judge_name": variant == "C",
                },
            },
            **validated.model_dump(),
            "generation": {
                "input_tokens": input_tokens,
                "max_new_tokens": self.settings.max_new_tokens,
                "seconds": round(elapsed, 3),
                "decoding": "greedy",
            },
            "scope_warnings": [
                *prediction_scope_warnings(
                    court=(court if variant in {"B", "C"} else None),
                    judgment_year_roc=(
                        judgment_year_roc if variant in {"B", "C"} else None
                    ),
                ),
                *routing_warnings(
                    variant=variant,
                    court=court,
                    judgment_year_roc=(judgment_year_roc),
                    judge_name=judge_name,
                ),
            ],
            "limitations": [
                "此結果是研究模型預測，不是法院判決、法律意見或結果保證。",
                "模型只預測罰金金額群組或拘役日數區間，不預測精確刑度。",
                "罰金群組依訓練資料的實際金額集合建立，不是連續金額區間。",
                f"本次依可用欄位自動選擇 {variant} 組 LoRA adapter。",
            ],
        }


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise PublicInsultPredictorError(
                "已指定 MPS，但目前 PyTorch 無法使用 MPS。"
            )
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise PublicInsultPredictorError(
                "已指定 CUDA，但目前 PyTorch 無法使用 CUDA。"
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise PublicInsultPredictorError(
        "PUBLIC_INSULT_DEVICE 只能是 auto、mps、cuda 或 cpu。"
    )


@lru_cache(maxsize=1)
def get_public_insult_predictor() -> PublicInsultPredictor:
    """每個 Python process 僅建立一個 8B 模型實例。"""
    return PublicInsultPredictor()
