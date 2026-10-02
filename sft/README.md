# 公然侮辱判決 SFT

使用 `taide/Llama-3.1-TAIDE-LX-8B-Chat` 訓練 LoRA adapter，
監督目標為判決預測 JSON 與量刑理由。預設設定保留第一版精確刑度實驗；
第二版另提供 A／B／C 消融設定、類別平衡取樣與三段式刑度區間。

## 模型切換與授權

TAIDE 模型需要先取得 Hugging Face 存取權：

1. 登入 [TAIDE 模型頁](https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat)，
   閱讀並同意條款，確認帳號已能存取模型檔案。
2. 在專案根目錄 `.env` 設定該帳號的 `HF_TOKEN`，或使用 `hf auth login`。
   Token 必須有讀取此 gated repository 的權限；不要將實際 token 寫入 YAML 或日誌。
3. 重新執行 dry-run。舊模型檢查通過不代表 TAIDE 通過：tokenizer、詞彙表與模板均可能不同。

程式仍使用 Hugging Face 自動下載與快取，不需要手動搬模型，也不會在失敗時切回其他模型。
取得授權只能解決存取限制，不能保證 Hub／CDN 下載不逾時。
下載中斷後是否保留未完成檔案取決於 Hub 版本，不要假定重新啟動一定能接續所有進度。

TAIDE 與先前模型的訓練成果分開保存，不刪除既有輸出或下載快取：

- checkpoint：`sft/outputs/checkpoints/public_insult_taide_llama31_8b_lora`
- 最終 adapter／tokenizer：`sft/outputs/adapters/public_insult_taide_llama31_8b_lora`
- 設定快照：`sft/outputs/logs/public_insult_taide_llama31_8b_lora`
- Trackio project：`legal-rag-agent-public-insult-sft`（本機記錄，沿用同一比較專案）
- Trackio run：`public-insult-taide-llama3_1-8b-lora-r16-lr1e-4`

預設的第一版 TAIDE 實驗已有 checkpoint，因此 `resume_from_checkpoint` 設為 `true`，
Trainer 會從 checkpoint 目錄自動選擇最新編號續跑。不得把其他 base model 的 checkpoint
放進此目錄。已經啟動的程序不會自動讀取新設定；此設定只影響之後啟動的訓練。

## 執行方式

在專案根目錄啟用 `.venv`，安裝包含訓練與離線評估工具的選用依賴：

```bash
pip install -r requirements-sft.txt
```

這會安裝 `torch`、`transformers`、`peft`、`accelerate`、`pyyaml`、
`python-dotenv`、`trackio` 與 BERTScore；一般 Agent 使用者只需安裝根目錄的
`requirements.txt`。
`.env` 可以設定 `HF_TOKEN`；終端機已設定的同名變數優先。

```bash
python sft/scripts/03_train_sft_lora.py --dry-run
```

dry-run 會讀取三份 JSONL、驗證訓練參數、下載或載入 tokenizer，並檢查全部樣本。
它不載入 8B 權重，也不寫入 checkpoint、adapter 或訓練日誌。
Hugging Face 自己的下載快取仍可能更新。

前置檢查通過後，正式訓練使用：

```bash
python sft/scripts/03_train_sft_lora.py
```

`--overwrite-output-dir` 會清除指定的舊 checkpoint 與 adapter，無自動備份。
這個參數不能和 `--dry-run` 或 `resume_from_checkpoint` 一起使用；
清除範圍限於專案 `sft/outputs/checkpoints/`、`sft/outputs/adapters/` 下的單次實驗目錄，
且設定、全部資料及模型載入成功後才會執行。

## 第二版 A／B／C 消融實驗

第二版資料由 `07_prepare_ablation_datasets.py` 產生。程式用 `case_id` 對應完整 master CSV，
並再核對案號、法院、裁判日期與刑種；任一欄不一致就整批停止。法官姓名只來自本機
已清理的 master CSV，不以模糊搜尋或模型猜測補值。重新產生資料時執行：

> 訓練用 master CSV 與產生後的 train／validation／test JSONL 屬於本機資料準備成果，
> 不隨公開儲存庫發布。下列路徑需替換成使用者自己的合法資料來源位置。

```bash
python sft/scripts/07_prepare_ablation_datasets.py \
  --master-csv "/path/to/public_insult_cleaned_master_all.csv" \
  --overwrite
```

產物位於 `data/sft/public_insult_ablation/`，`manifest.json` 記錄來源雜湊、筆數、
法官覆蓋率與各刑度區間分布。三種輸入只差下列制度資訊，案件切分及答案完全相同：

- A：犯罪事實與客觀量刑因素。
- B：A 加上法院與裁判年份。
- C：B 加上承審法官。

三個設定都保留 1,723 筆原始 train 資料一次，再於每個 epoch 動態加入 479 個拘役
案件索引。拘役曝光次數因此由 402 增為 881，epoch 共 2,202 筆，實際占比為
40.01%。額外索引依 epoch 輪替，`seed: 42` 可重現；JSONL 不會寫入重複案件。

三組前置檢查：

```bash
python sft/scripts/03_train_sft_lora.py --dry-run \
  --training-config sft/configs/training_config_ablation_a.yaml
python sft/scripts/03_train_sft_lora.py --dry-run \
  --training-config sft/configs/training_config_ablation_b.yaml
python sft/scripts/03_train_sft_lora.py --dry-run \
  --training-config sft/configs/training_config_ablation_c.yaml
```

正式實驗請依 A、B、C 順序逐一執行，不要在同一台 Mac 同時載入多份 8B 模型：

```bash
caffeinate -i python sft/scripts/03_train_sft_lora.py \
  --training-config sft/configs/training_config_ablation_a.yaml
```

完成 A 的訓練、validation 生成與評估後，將檔名中的 `a` 改成 `b`、再改成 `c`。
三組 checkpoint、adapter、日誌及 Trackio run 均已隔離，且都從相同 base model 開始，
不會沿用第一版或另一組消融實驗的 checkpoint。

## 資料與錯誤處理

- 每筆樣本為 `[system,] user, assistant` 的單輪對話。assistant 必須是完整 JSON，
  包含 `prediction.sentence_type` 與 `sentencing_reason`。
- `max_seq_length` 包含提示、答案及特殊 token。超長樣本直接報錯，列出集合與筆數位置，
  不截斷案件事實、答案或 EOS，也不靜默略過。長度上限應依訓練資料與硬體預算決定。
- train、validation、test 都檢查資料契約；只有 train 與 validation 交給 Trainer。
  test 不用於訓練或選參數；生成式評估方式見下一節。
- 分詞結果在通過前置檢查後保存在記憶體，訓練時重用；prompt 與 padding 的標籤為 `-100`。
- `group_by_length` 與 `warmup_ratio` 會依 Transformers 版本轉換名稱與表示方式。
  未知參數直接報錯，不以靜默捨棄的方式相容。
- tokenizer 與權重載入期間每 15 秒顯示等待狀態；單次請求的逾時值不等於總等待上限。
  同時啟動的下載若佔用相同快取鎖，會顯示 `Still waiting to acquire lock`；
  請先確認先前的程序是否仍在下載，不要直接刪除它持有的鎖檔。
  網路、401/403、429、5xx、本機快取及檔案權限錯誤分別提供提示。
  HF Token 無法修復 CDN 連線故障；停用 Xet 用戶端也不保證繞過 CDN。

## 回歸測試

```bash
python -B -m unittest discover -s sft/tests -v
```

預設測試使用合成資料與替代載入器，不下載模型、不讀取 `.env`、不更動正式成果。
在可使用 MPS 的 Mac 終端機中，可另外啟用兩步小型隨機 Llama 整合測試：

```bash
SFT_RUN_MPS_SMOKE=1 python -B -m unittest discover -s sft/tests -v
```

整合測試涵蓋 MPS、LoRA、Trackio 本機記錄及 adapter 儲存／重載。
小模型通過不代表完整 8B 模型已驗證記憶體需求、訓練品質或數值穩定性。

## 生成式任務評估

評估分成「模型生成」與「離線計分」兩階段。生成結果逐筆寫入 JSONL，之後修改
JSON/schema 或指標定義時，不需要重新載入 8B 模型。模型仍在訓練時只能執行
`--dry-run`；不要同時在 MPS 載入第二份 TAIDE。

先檢查 validation 資料與預設輸出路徑，不載入任何模型：

```bash
python sft/scripts/04_run_inference.py --dry-run
```

訓練結束後，以 adapter 生成 validation 預測：

```bash
python sft/scripts/04_run_inference.py \
  --split validation \
  --mode adapter
```

若中斷，可用完全相同的參數續跑；manifest 不相容時會拒絕混合兩次實驗：

```bash
python sft/scripts/04_run_inference.py \
  --split validation \
  --mode adapter \
  --resume
```

預設 validation 預測檔為：

```text
sft/outputs/predictions/public-insult-taide-llama3_1-8b-lora-r16-lr1e-4-adapter-validation.jsonl
```

離線計分不載入 TAIDE：

```bash
python sft/scripts/05_evaluate_outputs.py \
  --predictions sft/outputs/predictions/public-insult-taide-llama3_1-8b-lora-r16-lr1e-4-adapter-validation.jsonl
```

報表包含：

- `metrics.json`：JSON/schema、Accuracy、Macro-F1、各類別 precision/recall/F1、
  刑度一致率、字元級 ROUGE-L、joint success rate。
- `case_results.jsonl`：逐案格式、分類、刑度及理由分數，供錯誤分析。
- `confusion_matrix.csv`：真實類別與預測類別的混淆矩陣。
- `human_review.csv`：固定 seed、類別平衡的人工法律品質抽查表。

第二版消融實驗在推論時必須帶入相同設定檔。例如 A 組：

```bash
caffeinate -i python sft/scripts/04_run_inference.py \
  --training-config sft/configs/training_config_ablation_a.yaml \
  --split validation \
  --mode adapter

python sft/scripts/05_evaluate_outputs.py \
  --predictions sft/outputs/predictions/public-insult-v2-ablation-a-band-balanced40-adapter-validation.jsonl
```

第二版的 `metrics.json` 另包含六類刑度區間 Accuracy、balanced accuracy、Macro-F1、
精確區間命中率、相鄰一級容許率與跨兩級錯誤率；`amount_band_confusion_matrix.csv`
則呈現刑種內三段區間的混淆情況。validation 用於選擇 A／B／C；決策凍結後才對勝出組
執行一次 test 評估。

BERTScore 需要額外載入中文 encoder；請等 TAIDE 訓練與生成結束、MPS 記憶體釋放後再執行：

```bash
python sft/scripts/05_evaluate_outputs.py \
  --predictions sft/outputs/predictions/public-insult-taide-llama3_1-8b-lora-r16-lr1e-4-adapter-validation.jsonl \
  --bertscore \
  --overwrite
```

checkpoint 與超參數只能依 validation 結果選擇。全部決策凍結後，才將 `--split`
改成 `test` 執行一次正式測試；不要反覆查看 test 指標調參。原始 TAIDE baseline 使用
`--mode base` 產生另一份獨立輸出，再以相同計分腳本比較。

## Agent 使用已發布的 adapter

正式 Agent 透過 `app/inference/public_insult_predictor.py` 載入同一個 TAIDE 8B
基礎模型，並在其上掛載三個具名 LoRA adapter：

```text
a97141481/Llama-3.1-TAIDE-Public-Insult-A-LoRA@v1.0
a97141481/Llama-3.1-TAIDE-Public-Insult-B-LoRA@v1.0
a97141481/Llama-3.1-TAIDE-Public-Insult-C-LoRA@v1.0
```

基礎模型也固定到訓練時使用的 Hugging Face commit，避免 adapter 雖鎖定版本、
上游 TAIDE `main` 更新後卻產生不同結果。只有第一次呼叫量刑工具時才載入 8B 模型；
同一個 Python process 的後續預測會重用基礎模型並切換 adapter，不會複製三份
8B 權重。犯罪事實與量刑因素為必要欄位；法院與民國裁判年份同時存在時使用 B 組，
再有法官姓名時使用 C 組，制度資訊未知或不完整時使用 A 組且不填入猜測值。

不載入模型的契約測試：

```bash
python -m unittest tests.test_public_insult_predictor -v
```

實際載入 Hub adapter 並生成一筆結果：

```bash
caffeinate -i python -m scripts.manual_checks.check_public_insult_prediction
```

推論服務不會截斷過長的案件內容。預設輸入上限為 1024 tokens、輸出上限為
512 tokens，合計等於訓練的 `max_seq_length=1536`；超過時會要求先整理客觀摘要。
環境變數及固定 revision 範例列在專案根目錄 `.env.example`。

## 註解維護原則

註解使用繁體中文，保留 API 與設定欄位的原名。優先說明資料契約、設計取捨、
版本差異及失敗邊界；不逐行重述語法。修改行為時同步更新註解與對應測試，
避免留下已不適用的模型大小、參數名稱或網路故障假設。
