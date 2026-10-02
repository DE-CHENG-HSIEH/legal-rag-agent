"""公然侮辱 SFT 的離線任務指標。

此模組刻意不載入 Transformers、PEFT 或基礎模型。生成與計分分成兩個階段，
讓昂貴的 MPS 推論只執行一次；之後調整 JSON 解析或指標定義時，可直接重算結果。

指標契約：
1. 格式指標以全部案件為分母，避免只計算成功輸出的樣本而高估品質。
2. 分類指標把缺漏或未知類別視為錯誤，Macro-F1 僅平均資料集定義的真實類別。
3. 中文量刑理由使用字元級 ROUGE-L。中文句子通常沒有空白分詞，直接套用以空白
   分詞的英文 ROUGE 會低估重疊程度；此處移除空白但保留標點與文字。
4. BERTScore 屬於選配且需要額外模型，由命令列腳本在訓練結束後才載入。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Sequence


MISSING_LABEL = "__MISSING__"
EXACT_SCHEMA = "exact_amount_v1"
BAND_SCHEMA = "amount_band_v2"

AMOUNT_BANDS: dict[str, dict[str, Any]] = {
    "fine_1000_3000": {
        "sentence_type": "罰金",
        "minimum": 1000.0,
        "maximum": 3000.0,
        "unit": "新臺幣元",
        "rank": 0,
    },
    "fine_4000_5000": {
        "sentence_type": "罰金",
        "minimum": 4000.0,
        "maximum": 5000.0,
        "unit": "新臺幣元",
        "rank": 1,
    },
    "fine_6000_plus": {
        "sentence_type": "罰金",
        "minimum": 6000.0,
        "maximum": None,
        "unit": "新臺幣元",
        "rank": 2,
    },
    "detention_1_15": {
        "sentence_type": "拘役",
        "minimum": 1.0,
        "maximum": 15.0,
        "unit": "日",
        "rank": 0,
    },
    "detention_16_20": {
        "sentence_type": "拘役",
        "minimum": 16.0,
        "maximum": 20.0,
        "unit": "日",
        "rank": 1,
    },
    "detention_21_plus": {
        "sentence_type": "拘役",
        "minimum": 21.0,
        "maximum": None,
        "unit": "日",
        "rank": 2,
    },
}


@dataclass(frozen=True)
class ParsedModelOutput:
    """模型文字的 JSON 解析結果，分開保留嚴格格式與可恢復格式。"""

    payload: dict[str, Any] | None
    strict_json_valid: bool
    error: str | None


@dataclass(frozen=True)
class PredictionValues:
    sentence_type: str
    amount: float | None
    amount_band: str
    amount_min: float | None
    amount_max: float | None
    unit: str
    reason: str
    schema: str


def _decode_json_object(candidate: str) -> dict[str, Any] | None:
    """只有最外層為 JSON object 才接受，避免陣列或純量通過資料契約。"""
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def parse_model_output(text: Any) -> ParsedModelOutput:
    """解析模型輸出，並辨識 Markdown fence 或前後說明中的 JSON object。

    ``strict_json_valid`` 僅在完整輸出本身就是 object 時為 True。為了分析模型內容，
    fence 或額外前綴中的 object 仍會被恢復並納入後續指標，但不算嚴格 JSON 成功。
    """
    if not isinstance(text, str) or not text.strip():
        return ParsedModelOutput(None, False, "empty_output")

    stripped = text.strip()
    strict_payload = _decode_json_object(stripped)
    if strict_payload is not None:
        return ParsedModelOutput(strict_payload, True, None)

    # 模型常把正確 JSON 包在 ```json ... ``` 中。先處理 fence，可提供較明確的恢復路徑。
    for fenced in re.findall(
        r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL
    ):
        payload = _decode_json_object(fenced.strip())
        if payload is not None:
            return ParsedModelOutput(
                payload, False, "json_recovered_from_markdown_fence"
            )

    # raw_decode 可在說明文字後找到第一個完整 object，且正確處理字串內的大括號與跳脫。
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return ParsedModelOutput(
                value, False, "json_recovered_from_surrounding_text"
            )

    return ParsedModelOutput(None, False, "json_object_not_found")


def prediction_schema(payload: dict[str, Any] | None) -> str:
    """依欄位辨識 v1 精確數字或 v2 區間；缺少兩者時沿用 v1 供舊報表診斷。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("prediction"), dict):
        return EXACT_SCHEMA
    prediction = payload["prediction"]
    if any(
        name in prediction
        for name in (
            "sentence_amount_band",
            "sentence_amount_min",
            "sentence_amount_max",
        )
    ):
        return BAND_SCHEMA
    return EXACT_SCHEMA


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def schema_errors(
    payload: dict[str, Any] | None,
    *,
    expected_schema: str | None = None,
) -> list[str]:
    """回傳完整預測 schema 的錯誤清單，不把錯誤悄悄轉成預設值。"""
    if not isinstance(payload, dict):
        return ["root_not_object"]

    errors: list[str] = []
    prediction = payload.get("prediction")
    if not isinstance(prediction, dict):
        errors.append("prediction_not_object")
        prediction = {}

    for field_name in ("sentence_type", "sentence_amount_unit"):
        value = prediction.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"prediction.{field_name}_missing")

    actual_schema = prediction_schema(payload)
    if expected_schema is not None and actual_schema != expected_schema:
        errors.append(f"prediction.schema_mismatch_expected_{expected_schema}")

    if actual_schema == BAND_SCHEMA:
        band_value = prediction.get("sentence_amount_band")
        band = band_value.strip() if isinstance(band_value, str) else ""
        if not band:
            errors.append("prediction.sentence_amount_band_missing")
        elif band not in AMOUNT_BANDS:
            errors.append("prediction.sentence_amount_band_unknown")

        minimum = _finite_number(prediction.get("sentence_amount_min"))
        maximum_raw = prediction.get("sentence_amount_max")
        maximum = _finite_number(maximum_raw)
        if minimum is None:
            errors.append("prediction.sentence_amount_min_invalid")
        if maximum_raw is not None and maximum is None:
            errors.append("prediction.sentence_amount_max_invalid")
        if minimum is not None and maximum is not None and maximum < minimum:
            errors.append("prediction.sentence_amount_range_invalid")

        if band in AMOUNT_BANDS:
            definition = AMOUNT_BANDS[band]
            sentence_type = prediction.get("sentence_type")
            unit = prediction.get("sentence_amount_unit")
            if sentence_type != definition["sentence_type"]:
                errors.append("prediction.sentence_amount_band_type_mismatch")
            if unit != definition["unit"]:
                errors.append("prediction.sentence_amount_band_unit_mismatch")
            if minimum != definition["minimum"]:
                errors.append("prediction.sentence_amount_band_min_mismatch")
            if maximum != definition["maximum"]:
                errors.append("prediction.sentence_amount_band_max_mismatch")
    else:
        raw = prediction.get("sentence_amount_raw")
        if not isinstance(raw, str) or not raw.strip():
            errors.append("prediction.sentence_amount_raw_missing")
        if _finite_number(prediction.get("sentence_amount_numeric")) is None:
            errors.append("prediction.sentence_amount_numeric_invalid")

    reason = payload.get("sentencing_reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("sentencing_reason_missing")

    return errors


def _reference_payload(record: dict[str, Any]) -> dict[str, Any]:
    reference = record.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("Each prediction record must contain a reference JSON object.")
    if schema_errors(reference):
        raise ValueError(
            f"Reference output violates the expected schema: {schema_errors(reference)}"
        )
    return reference


def _prediction_values(payload: dict[str, Any] | None) -> PredictionValues:
    if not isinstance(payload, dict):
        return PredictionValues(
            MISSING_LABEL, None, MISSING_LABEL, None, None, "", "", EXACT_SCHEMA
        )
    prediction = payload.get("prediction")
    if not isinstance(prediction, dict):
        prediction = {}

    sentence_type = prediction.get("sentence_type")
    label = (
        sentence_type.strip()
        if isinstance(sentence_type, str) and sentence_type.strip()
        else MISSING_LABEL
    )

    amount = _finite_number(prediction.get("sentence_amount_numeric"))
    band_value = prediction.get("sentence_amount_band")
    band = (
        band_value.strip()
        if isinstance(band_value, str) and band_value.strip()
        else MISSING_LABEL
    )
    minimum = _finite_number(prediction.get("sentence_amount_min"))
    maximum = _finite_number(prediction.get("sentence_amount_max"))

    unit_value = prediction.get("sentence_amount_unit")
    unit = unit_value.strip() if isinstance(unit_value, str) else ""
    reason_value = payload.get("sentencing_reason")
    reason = reason_value.strip() if isinstance(reason_value, str) else ""
    return PredictionValues(
        sentence_type=label,
        amount=amount,
        amount_band=band,
        amount_min=minimum,
        amount_max=maximum,
        unit=unit,
        reason=reason,
        schema=prediction_schema(payload),
    )


def classification_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
) -> tuple[dict[str, Any], list[str], list[list[int]]]:
    """計算 Accuracy、Macro-F1、各類別指標及 confusion matrix。"""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length.")
    if not references:
        raise ValueError("At least one prediction record is required.")

    expected_labels = sorted(set(references))
    observed_extra = sorted(set(predictions) - set(expected_labels) - {MISSING_LABEL})
    matrix_labels = expected_labels + observed_extra
    if MISSING_LABEL in predictions:
        matrix_labels.append(MISSING_LABEL)

    label_index = {label: index for index, label in enumerate(matrix_labels)}
    matrix = [[0 for _ in matrix_labels] for _ in matrix_labels]
    for actual, predicted in zip(references, predictions):
        matrix[label_index[actual]][label_index[predicted]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in expected_labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(references, predictions)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(references, predictions)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(references, predictions)
        )
        support = sum(actual == label for actual in references)

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = sum(
        actual == predicted for actual, predicted in zip(references, predictions)
    )
    summary: dict[str, Any] = {
        "accuracy": correct / len(references),
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(item["recall"] for item in per_class.values())
        / len(per_class),
        "correct_count": correct,
        "unknown_prediction_count": sum(
            predicted not in expected_labels and predicted != MISSING_LABEL
            for predicted in predictions
        ),
        "missing_prediction_count": predictions.count(MISSING_LABEL),
        "per_class": per_class,
    }
    return summary, matrix_labels, matrix


def _normalized_reason(text: str) -> str:
    """中文 ROUGE-L 以字元計算；只移除排版空白，不改寫原始文字。"""
    return "".join(text.split())


def _lcs_length(left: str, right: str) -> int:
    """使用單列動態規劃計算 LCS，將記憶體複雜度降為 O(min(m, n))。"""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_character in left:
        current = [0]
        for index, right_character in enumerate(right, start=1):
            if left_character == right_character:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, candidate: str) -> dict[str, float]:
    """回傳字元級 ROUGE-L precision、recall 與 F1。"""
    normalized_reference = _normalized_reason(reference)
    normalized_candidate = _normalized_reason(candidate)
    if not normalized_reference or not normalized_candidate:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = _lcs_length(normalized_reference, normalized_candidate)
    precision = overlap / len(normalized_candidate)
    recall = overlap / len(normalized_reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_prediction_records(
    records: Iterable[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """以全部案件為分母計算格式、分類、刑度、理由與端到端指標。"""
    materialized = list(records)
    if not materialized:
        raise ValueError("Prediction file is empty.")

    case_ids: set[str] = set()
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    actual_bands: list[str] = []
    predicted_bands: list[str] = []
    case_results: list[dict[str, Any]] = []
    strict_json_count = 0
    parseable_json_count = 0
    schema_valid_count = 0
    joint_success_count = 0
    full_prediction_exact_count = 0
    amount_absolute_errors: list[float] = []
    rouge_scores: list[dict[str, float]] = []
    error_counts: Counter[str] = Counter()
    reference_schema: str | None = None

    for position, record in enumerate(materialized, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Prediction record {position} must be a JSON object.")
        case_id_value = record.get("case_id")
        case_id = str(case_id_value).strip() if case_id_value is not None else ""
        if not case_id:
            raise ValueError(f"Prediction record {position} has no case_id.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate case_id in prediction file: {case_id}")
        case_ids.add(case_id)

        reference = _reference_payload(record)
        current_reference_schema = prediction_schema(reference)
        if reference_schema is None:
            reference_schema = current_reference_schema
        elif reference_schema != current_reference_schema:
            raise ValueError(
                "Prediction file mixes exact-amount and amount-band reference schemas."
            )
        generated_text = record.get("generated_text", "")
        generation_error = record.get("generation_error")
        parsed = parse_model_output(generated_text)
        errors = schema_errors(parsed.payload, expected_schema=current_reference_schema)
        if generation_error:
            errors.insert(0, "generation_error")

        strict_json_count += int(parsed.strict_json_valid)
        parseable_json_count += int(parsed.payload is not None)
        schema_valid = parsed.payload is not None and not errors
        schema_valid_count += int(schema_valid)
        if parsed.error:
            error_counts[parsed.error] += 1
        error_counts.update(errors)

        actual = _prediction_values(reference)
        predicted = _prediction_values(parsed.payload)
        actual_labels.append(actual.sentence_type)
        predicted_labels.append(predicted.sentence_type)

        label_correct = predicted.sentence_type == actual.sentence_type
        if current_reference_schema == BAND_SCHEMA:
            actual_bands.append(actual.amount_band)
            predicted_bands.append(predicted.amount_band)
            amount_exact = (
                schema_valid
                and label_correct
                and predicted.amount_band == actual.amount_band
            )
        else:
            amount_exact = (
                predicted.amount is not None
                and actual.amount is not None
                and predicted.unit == actual.unit
                and predicted.amount == actual.amount
            )
            if (
                predicted.amount is not None
                and actual.amount is not None
                and predicted.unit == actual.unit
            ):
                amount_absolute_errors.append(abs(predicted.amount - actual.amount))

        # Joint success 對應產品可直接消費的最低條件：可解析、完整 schema、類別正確。
        # 刑度數值正確另以 full_prediction_exact_match_rate 呈現，避免混淆錯誤來源。
        joint_success = schema_valid and label_correct
        full_prediction_exact = joint_success and amount_exact
        joint_success_count += int(joint_success)
        full_prediction_exact_count += int(full_prediction_exact)

        reason_score = rouge_l(actual.reason, predicted.reason)
        rouge_scores.append(reason_score)
        case_results.append(
            {
                "case_id": case_id,
                "strict_json_valid": parsed.strict_json_valid,
                "parseable_json": parsed.payload is not None,
                "schema_valid": schema_valid,
                "format_errors": errors,
                "parse_note": parsed.error,
                "actual_sentence_type": actual.sentence_type,
                "predicted_sentence_type": predicted.sentence_type,
                "sentence_type_correct": label_correct,
                "actual_sentence_amount_numeric": actual.amount,
                "predicted_sentence_amount_numeric": predicted.amount,
                "actual_sentence_amount_band": actual.amount_band,
                "predicted_sentence_amount_band": predicted.amount_band,
                "actual_sentence_amount_min": actual.amount_min,
                "predicted_sentence_amount_min": predicted.amount_min,
                "actual_sentence_amount_max": actual.amount_max,
                "predicted_sentence_amount_max": predicted.amount_max,
                "actual_sentence_amount_unit": actual.unit,
                "predicted_sentence_amount_unit": predicted.unit,
                "sentence_amount_exact": amount_exact,
                "reference_reason": actual.reason,
                "generated_reason": predicted.reason,
                "rouge_l_precision": reason_score["precision"],
                "rouge_l_recall": reason_score["recall"],
                "rouge_l_f1": reason_score["f1"],
                "joint_success": joint_success,
                "full_prediction_exact": full_prediction_exact,
            }
        )

    classification, matrix_labels, matrix = classification_metrics(
        actual_labels, predicted_labels
    )
    total = len(materialized)
    sentence_amount_metrics: dict[str, Any]
    amount_band_confusion: dict[str, Any] | None = None
    if reference_schema == BAND_SCHEMA:
        band_classification, band_labels, band_matrix = classification_metrics(
            actual_bands, predicted_bands
        )
        within_one_band_count = 0
        two_band_error_count = 0
        for actual_label, predicted_label, actual_band, predicted_band in zip(
            actual_labels, predicted_labels, actual_bands, predicted_bands
        ):
            if (
                actual_label == predicted_label
                and actual_band in AMOUNT_BANDS
                and predicted_band in AMOUNT_BANDS
            ):
                distance = abs(
                    AMOUNT_BANDS[actual_band]["rank"]
                    - AMOUNT_BANDS[predicted_band]["rank"]
                )
                within_one_band_count += int(distance <= 1)
                two_band_error_count += int(distance >= 2)
        sentence_amount_metrics = {
            "mode": "band",
            "exact_match_rate": full_prediction_exact_count / total,
            "exact_match_count": full_prediction_exact_count,
            "within_one_band_rate": within_one_band_count / total,
            "within_one_band_count": within_one_band_count,
            "two_band_error_rate": two_band_error_count / total,
            "two_band_error_count": two_band_error_count,
            "classification": band_classification,
        }
        amount_band_confusion = {"labels": band_labels, "values": band_matrix}
    else:
        sentence_amount_metrics = {
            "mode": "exact_numeric",
            "exact_match_rate": full_prediction_exact_count / total,
            "exact_match_count": full_prediction_exact_count,
            "mae_on_comparable_outputs": (
                sum(amount_absolute_errors) / len(amount_absolute_errors)
                if amount_absolute_errors
                else None
            ),
            "comparable_output_count": len(amount_absolute_errors),
        }
    metrics: dict[str, Any] = {
        "dataset": {
            "case_count": total,
            "reference_schema": reference_schema,
            "reference_class_distribution": dict(
                sorted(Counter(actual_labels).items())
            ),
            "reference_amount_band_distribution": dict(
                sorted(Counter(actual_bands).items())
            ),
        },
        "format": {
            "strict_json_valid_rate": strict_json_count / total,
            "strict_json_valid_count": strict_json_count,
            "parseable_json_rate": parseable_json_count / total,
            "parseable_json_count": parseable_json_count,
            "schema_compliance_rate": schema_valid_count / total,
            "schema_compliance_count": schema_valid_count,
            "error_counts": dict(sorted(error_counts.items())),
        },
        "classification": classification,
        "sentence_amount": sentence_amount_metrics,
        "sentencing_reason": {
            "rouge_l_precision": sum(score["precision"] for score in rouge_scores)
            / total,
            "rouge_l_recall": sum(score["recall"] for score in rouge_scores) / total,
            "rouge_l_f1": sum(score["f1"] for score in rouge_scores) / total,
            "scored_case_count": total,
        },
        "end_to_end": {
            "joint_success_rate": joint_success_count / total,
            "joint_success_count": joint_success_count,
            "full_prediction_exact_match_rate": full_prediction_exact_count / total,
            "full_prediction_exact_match_count": full_prediction_exact_count,
        },
        "confusion_matrix": {
            "labels": matrix_labels,
            "values": matrix,
        },
    }
    if amount_band_confusion is not None:
        metrics["amount_band_confusion_matrix"] = amount_band_confusion
    return metrics, case_results


def compute_bertscore(
    case_results: Sequence[dict[str, Any]],
    *,
    model_type: str | None = None,
    batch_size: int = 8,
    device: str | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """選配 BERTScore；空白生成以 0 分納入總體平均，避免 coverage bias。"""
    try:
        from bert_score import score
    except ImportError as error:
        raise RuntimeError(
            "BERTScore evaluation requires the optional 'bert-score' package. "
            "Install it after the current TAIDE training has finished."
        ) from error

    valid_indices = [
        index
        for index, result in enumerate(case_results)
        if result["generated_reason"] and result["reference_reason"]
    ]
    all_f1 = [0.0] * len(case_results)
    if not valid_indices:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage_rate": 0.0,
            "model_type": model_type or "bert-score-default-for-zh",
        }, all_f1

    candidates = [case_results[index]["generated_reason"] for index in valid_indices]
    references = [case_results[index]["reference_reason"] for index in valid_indices]
    keyword_arguments: dict[str, Any] = {
        "cands": candidates,
        "refs": references,
        "lang": "zh",
        "batch_size": batch_size,
        "verbose": True,
    }
    if model_type:
        keyword_arguments["model_type"] = model_type
    if device:
        keyword_arguments["device"] = device

    precision, recall, f1 = score(**keyword_arguments)
    precision_values = [float(value) for value in precision]
    recall_values = [float(value) for value in recall]
    f1_values = [float(value) for value in f1]
    for index, value in zip(valid_indices, f1_values):
        all_f1[index] = value

    total = len(case_results)
    return {
        # 分母固定為全部案件；缺少量刑理由的輸出以 0 分計算。
        "precision": sum(precision_values) / total,
        "recall": sum(recall_values) / total,
        "f1": sum(f1_values) / total,
        "coverage_rate": len(valid_indices) / total,
        "model_type": model_type or "bert-score-default-for-zh",
    }, all_f1
