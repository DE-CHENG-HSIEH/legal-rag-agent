"""Warm the legal-authority index before starting the UI."""

from pathlib import Path
import sys


# Keep the script usable as either ``python scripts/...`` or ``python -m scripts...``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.vector_store import get_authority_vector_store  # noqa: E402


def main() -> None:
    vector_store = get_authority_vector_store()
    record_count = len(vector_store.get(include=[]).get("ids", []))
    print(f"重要法律見解向量索引已就緒：{record_count} 筆。")


if __name__ == "__main__":
    main()
