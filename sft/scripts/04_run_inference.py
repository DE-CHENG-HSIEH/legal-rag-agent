"""使用原始 TAIDE 或 LoRA adapter 逐案生成 validation/test 預測。

生成結果逐列寫入 JSONL，並提供 --resume。這能把昂貴的模型推論與便宜的離線指標
分開；評估公式變更時不必再次載入 8B 模型。預設使用 validation，避免不小心在模型
選擇階段反覆查看 test set。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


def configure_runtime_environment() -> None:
    """與訓練入口維持相同 Hub/MPS 預設，但尊重使用者現有環境變數。"""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


configure_runtime_environment()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_SRC = PROJECT_ROOT / "sft/src"
if str(SFT_SRC) not in sys.path:
    sys.path.insert(0, str(SFT_SRC))

from public_insult_sft.inference import (  # noqa: E402
    load_completed_case_ids,
    make_prediction_record,
    prepare_inference_cases,
)


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "sft/configs/model_config.yaml"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "sft/configs/training_config.yaml"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "sft/outputs/predictions"


def load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("注意：找到 .env，但未安裝 python-dotenv，將沿用目前終端機環境。")
        return
    load_dotenv(dotenv_path=env_path, override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validation/test predictions from TAIDE base or its LoRA adapter."
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use validation for model selection; reserve test for the final frozen experiment.",
    )
    parser.add_argument("--mode", choices=("base", "adapter"), default="adapter")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Adapter/checkpoint to evaluate. Defaults to training_config output.adapter_dir.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases for a smoke run.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data/output naming without loading tokenizer, model, adapter or MPS.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume", action="store_true", help="Continue a compatible partial JSONL."
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this prediction JSONL and manifest.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Inference requires pyyaml. Install it with: pip install pyyaml"
        ) from error
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a YAML mapping: {resolved}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Each dataset line must be a JSON object: {path}:{line_number}"
                )
            records.append(value)
    return records


def safe_run_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "sft-evaluation")).strip("-._")
    return name or "sft-evaluation"


def default_output_path(training_config: dict[str, Any], mode: str, split: str) -> Path:
    run_name = training_config.get("training", {}).get("run_name", "sft-evaluation")
    return DEFAULT_PREDICTION_DIR / f"{safe_run_name(run_name)}-{mode}-{split}.jsonl"


def select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested but is not available in this Python environment."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this Python environment."
        )
    return torch.device(requested)


def resolve_torch_dtype(torch: Any, configured: Any, device: Any) -> Any:
    if device.type == "cpu":
        # CPU 半精度算子支援不完整；評估雖較慢，仍以 float32 的正確性優先。
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": torch.float32,
        None: torch.float32,
    }
    if configured not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {configured}")
    return mapping[configured]


def require_runtime_dependencies(mode: str) -> dict[str, Any]:
    missing: list[str] = []
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
        missing.append("torch")
    try:
        transformers = importlib.import_module("transformers")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer
    except ImportError:
        AutoModelForCausalLM = None
        AutoTokenizer = None
        missing.append("transformers")

    PeftModel: Any = None
    if mode == "adapter":
        try:
            PeftModel = importlib.import_module("peft").PeftModel
        except ImportError:
            missing.append("peft")
    if missing:
        raise RuntimeError(
            "Missing inference dependencies: " + ", ".join(sorted(missing))
        )
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
    }


def model_access_token(model_config: dict[str, Any]) -> str | None:
    variable_name = model_config.get("token_env_var")
    if variable_name is None:
        return None
    if not isinstance(variable_name, str) or not variable_name.strip():
        raise ValueError(
            "model_config token_env_var must be a non-empty environment variable name."
        )
    value = os.environ.get(variable_name)
    return value.strip() if value and value.strip() else None


def static_manifest(
    *,
    model_config: dict[str, Any],
    split: str,
    mode: str,
    dataset_path: Path,
    adapter_path: Path | None,
    limit: int | None,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """續跑前比對所有會改變輸出的條件，避免把不同實驗混在同一個 JSONL。"""
    return {
        "format_version": 1,
        "base_model": model_config["base_model_name_or_path"],
        "split": split,
        "mode": mode,
        "dataset_path": str(dataset_path.resolve()),
        "adapter_path": str(adapter_path.resolve()) if adapter_path else None,
        "limit": limit,
        "generation": generation_config,
    }


def write_json_atomically(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prepare_output(
    output_path: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
    overwrite: bool,
) -> set[str]:
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output_path.write_text("", encoding="utf-8")
        write_json_atomically(manifest_path, manifest)
        return set()
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Prediction output already exists: {output_path}. Use --resume or --overwrite."
        )
    if resume and output_path.exists():
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                "Existing prediction manifest does not match the requested experiment."
            )
        return load_completed_case_ids(output_path)

    output_path.touch()
    write_json_atomically(manifest_path, manifest)
    return set()


def generation_arguments(
    model_config: dict[str, Any], max_new_tokens: int | None
) -> dict[str, Any]:
    configured = model_config.get("generation", {})
    if not isinstance(configured, dict):
        raise ValueError("model_config generation must be a mapping.")
    arguments: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens or configured.get("max_new_tokens", 512)),
        "do_sample": bool(configured.get("do_sample", False)),
    }
    if arguments["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")
    # temperature/top_p 對 greedy decoding 無效，也會觸發 Transformers 警告；只在抽樣時傳入。
    if arguments["do_sample"]:
        arguments["temperature"] = float(configured.get("temperature", 1.0))
        arguments["top_p"] = float(configured.get("top_p", 1.0))
    return arguments


def load_model_and_tokenizer(
    dependencies: dict[str, Any],
    model_config: dict[str, Any],
    mode: str,
    adapter_path: Path | None,
    device: Any,
) -> tuple[Any, Any]:
    torch = dependencies["torch"]
    source = model_config["base_model_name_or_path"]
    token = model_access_token(model_config)
    common: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "local_files_only": bool(model_config.get("local_files_only", False)),
    }
    if token:
        common["token"] = token

    print(f"載入 tokenizer：{source}", flush=True)
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        source,
        use_fast=bool(model_config.get("use_fast_tokenizer", True)),
        **common,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for generation.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = resolve_torch_dtype(torch, model_config.get("torch_dtype"), device)
    model_kwargs = {
        **common,
        "dtype": dtype,
        "low_cpu_mem_usage": bool(model_config.get("low_cpu_mem_usage", True)),
    }
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    # 本專案使用的 Transformers 版本接受 dtype；若未來版本改名，直接報錯比靜默忽略安全。
    signature = inspect.signature(dependencies["AutoModelForCausalLM"].from_pretrained)
    if "dtype" not in signature.parameters and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        raise RuntimeError(
            "Installed Transformers does not support the expected dtype loading argument."
        )

    print(f"載入基礎模型：{source}（device={device}, dtype={dtype}）", flush=True)
    model = dependencies["AutoModelForCausalLM"].from_pretrained(source, **model_kwargs)
    if mode == "adapter":
        if adapter_path is None:
            raise ValueError("adapter mode requires an adapter path.")
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Adapter directory has no adapter_config.json: {adapter_path}"
            )
        print(f"套用 LoRA adapter：{adapter_path}", flush=True)
        model = dependencies["PeftModel"].from_pretrained(
            model, adapter_path, is_trainable=False
        )

    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_text(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    device: Any,
    generation_config: dict[str, Any],
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, prompt_length:]
    # TAIDE 使用 BPE tokenizer。Transformers 的通用 cleanup 主要為 WordPiece 設計，
    # 可能移除標點前空白並改變原始輸出；評估必須保留模型真正生成的文字。
    return tokenizer.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")
    if args.dry_run and (args.resume or args.overwrite):
        raise ValueError("--dry-run cannot be combined with --resume or --overwrite.")

    load_project_env()
    model_config = load_yaml(args.model_config)
    training_config = load_yaml(args.training_config)
    if not isinstance(model_config.get("base_model_name_or_path"), str):
        raise ValueError("model_config must define base_model_name_or_path.")

    data_config = training_config.get("data", {})
    output_config = training_config.get("output", {})
    dataset_path = resolve_project_path(data_config[f"{args.split}_file"])
    records = load_jsonl(dataset_path)
    cases = prepare_inference_cases(records)
    if args.limit is not None:
        cases = cases[: args.limit]

    adapter_path: Path | None = None
    if args.mode == "adapter":
        configured_adapter = args.adapter_path or output_config.get("adapter_dir")
        if configured_adapter is None:
            raise ValueError(
                "No adapter path was provided and output.adapter_dir is missing."
            )
        adapter_path = resolve_project_path(configured_adapter)

    output_path = resolve_project_path(
        args.output or default_output_path(training_config, args.mode, args.split)
    )
    generation_config = generation_arguments(model_config, args.max_new_tokens)
    manifest = static_manifest(
        model_config=model_config,
        split=args.split,
        mode=args.mode,
        dataset_path=dataset_path,
        adapter_path=adapter_path,
        limit=args.limit,
        generation_config=generation_config,
    )

    print(f"Split: {args.split} ({len(cases)} cases)")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print(
            "Inference dry run completed: config/data validated; tokenizer, model, adapter and MPS were not loaded."
        )
        return

    completed_case_ids = prepare_output(
        output_path,
        manifest,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    pending_cases = [case for case in cases if case.case_id not in completed_case_ids]
    if not pending_cases:
        print(
            "All requested cases already exist in the prediction file; no model was loaded."
        )
        return
    if completed_case_ids:
        print(
            f"Resume: {len(completed_case_ids)} completed, {len(pending_cases)} remaining."
        )

    dependencies = require_runtime_dependencies(args.mode)
    device = select_device(dependencies["torch"], args.device)
    model, tokenizer = load_model_and_tokenizer(
        dependencies,
        model_config,
        args.mode,
        adapter_path,
        device,
    )

    with output_path.open("a", encoding="utf-8") as output_file:
        for index, case in enumerate(pending_cases, start=1):
            started_at = time.monotonic()
            generated_text = generate_text(
                torch=dependencies["torch"],
                model=model,
                tokenizer=tokenizer,
                messages=case.input_messages,
                device=device,
                generation_config=generation_config,
            )
            elapsed = time.monotonic() - started_at
            result = make_prediction_record(
                case,
                generated_text=generated_text,
                split=args.split,
                mode=args.mode,
                base_model=model_config["base_model_name_or_path"],
                adapter_path=str(adapter_path) if adapter_path else None,
                generation_seconds=elapsed,
            )
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            # 每十筆同步一次磁碟；即使程序意外中止，最多只需重跑少量尚未落盤資料。
            if index % 10 == 0:
                os.fsync(output_file.fileno())
            print(
                f"Generated {index}/{len(pending_cases)}: {case.case_id} ({elapsed:.1f}s)",
                flush=True,
            )
        os.fsync(output_file.fileno())

    print(f"Inference completed: {output_path}")


if __name__ == "__main__":
    main()
