"""Deterministically render legal-source artifacts outside the language model."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.schemas import normalize_markdown_text


LEGAL_SOURCE_ARTIFACT_TYPE = "legal_sources_v1"


def make_legal_source_artifact(
    source_kind: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON-safe artifact contract shared by tools, UI, and evals."""

    return {
        "artifact_type": LEGAL_SOURCE_ARTIFACT_TYPE,
        "source_kind": source_kind,
        "entries": entries,
    }


def _current_turn_messages(messages: list[Any]) -> list[Any]:
    """Return messages produced after the latest user input."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return messages[index + 1 :]
        if isinstance(message, dict) and message.get("role") == "user":
            return messages[index + 1 :]
    return messages


def collect_legal_source_artifacts(messages: list[Any]) -> list[dict[str, Any]]:
    """Collect trusted source artifacts from the current Agent turn only."""

    artifacts: list[dict[str, Any]] = []
    for message in _current_turn_messages(messages):
        if not isinstance(message, ToolMessage):
            continue
        artifact = getattr(message, "artifact", None)
        if not isinstance(artifact, dict):
            continue
        if artifact.get("artifact_type") != LEGAL_SOURCE_ARTIFACT_TYPE:
            continue
        entries = artifact.get("entries")
        if isinstance(entries, list) and entries:
            artifacts.append(artifact)
    return artifacts


def _render_metadata(metadata: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for field in metadata:
        label = str(field.get("label", "")).strip()
        value = str(field.get("value", "")).strip()
        if not label or not value:
            continue
        if field.get("format") == "url" and value.startswith(("https://", "http://")):
            value = f"[{value}]({value})"
        lines.append(f"- **{label}：** {value}")
    return lines


def render_legal_sources(messages: list[Any]) -> str:
    """Render source artifacts without asking the model to reproduce source text."""

    artifacts = collect_legal_source_artifacts(messages)
    if not artifacts:
        return ""

    lines = ["## 法律來源原文"]
    for artifact in artifacts:
        for entry in artifact["entries"]:
            title = str(entry.get("title", "法律來源")).strip() or "法律來源"
            lines.extend(["", f"### {title}"])

            metadata = entry.get("metadata", [])
            if isinstance(metadata, list):
                rendered_metadata = _render_metadata(metadata)
                if rendered_metadata:
                    lines.extend(["", *rendered_metadata])

            sections = entry.get("sections", [])
            if isinstance(sections, list):
                for section in sections:
                    heading = str(section.get("heading", "")).strip()
                    text = str(section.get("text", "")).strip()
                    if not heading or not text:
                        continue
                    lines.extend(["", f"#### {heading}", "", text])

    return "\n".join(lines).strip()


def remove_model_source_sections(answer: str) -> str:
    """Discard model-authored source sections before trusted artifacts are appended."""

    lines = normalize_markdown_text(answer).splitlines()
    output: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## 法律來源原文":
            skipping = True
            continue
        if skipping and stripped.startswith("## "):
            skipping = False
        if not skipping:
            output.append(line)
    return normalize_markdown_text("\n".join(output))


def _structured_answer(result: dict[str, Any]) -> str:
    structured_response = result.get("structured_response")
    if hasattr(structured_response, "answer_markdown"):
        return str(structured_response.answer_markdown)
    if isinstance(structured_response, dict):
        return str(structured_response.get("answer_markdown", ""))

    messages = result.get("messages", [])
    if not messages:
        return ""
    final_message = messages[-1]
    if isinstance(final_message, dict):
        return str(final_message.get("content", ""))
    return str(getattr(final_message, "content", ""))


def compose_final_answer(result: dict[str, Any]) -> str:
    """Combine model analysis with source text rendered from trusted tool artifacts."""

    analysis = remove_model_source_sections(_structured_answer(result))
    sources = render_legal_sources(result.get("messages", []))
    return "\n\n".join(part for part in (analysis, sources) if part).strip()
