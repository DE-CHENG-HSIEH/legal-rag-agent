"""實際載入 Hugging Face v1.0 adapter，執行一筆端到端 MPS smoke test。

這不是一般單元測試：第一次執行會從 Hub 下載本機尚未快取的檔案，並將完整
TAIDE 8B 基礎模型載入記憶體。執行期間不要同時啟動另一份訓練或推論程序。
"""

import json
from pathlib import Path
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.inference.public_insult_predictor import get_public_insult_predictor


def main() -> None:
    load_dotenv()
    predictor = get_public_insult_predictor()
    result = predictor.predict(
        criminal_facts=(
            "被告於晚間騎乘機車時與告訴人發生行車糾紛，"
            "並在道路旁以粗俗、貶抑性言詞公開辱罵告訴人。"
        ),
        sentencing_factors=(
            "被告於準備程序坦承犯行，大學肄業，月收入約新臺幣3萬元，"
            "無需扶養他人；雙方因賠償金額認知差距而未能和解。"
        ),
        court="臺灣臺北地方法院",
        judgment_year_roc=107,
        judge_name="王惟琪",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
