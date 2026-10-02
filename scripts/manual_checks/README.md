# 手動整合檢查

這些腳本會實際呼叫模型、向量資料庫或外部服務，因此可能需要憑證、網路、
大型模型下載與較多記憶體，不應混入快速且可重現的自動測試。

設定 `.env` 後，從專案根目錄以 module 方式個別執行，確保專案 package
可以被正確匯入。例如：

```bash
python -m scripts.manual_checks.check_reranked_search
python -m scripts.manual_checks.check_public_insult_prediction
python -m scripts.manual_checks.check_agent_routing
```

自動測試位於 `tests/` 與 `sft/tests/`。
