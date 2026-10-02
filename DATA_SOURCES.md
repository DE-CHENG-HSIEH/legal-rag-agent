# Data Sources and Governance

## Public runtime corpus

`data/public/public_insult_judgments.jsonl` is the version-controlled corpus used
by the similar-judgment retrieval tool. It contains 2,154 public-insult judgments:

| Court | Records |
| --- | ---: |
| 臺灣臺北地方法院 | 879 |
| 臺灣新北地方法院 | 991 |
| 臺灣士林地方法院 | 284 |

The decisions span ROC years 104–114. Source decisions were obtained from the
[Judicial Yuan Judgment System](https://judgment.judicial.gov.tw/) and the
[Judicial Yuan Open Data Platform](https://opendata.judicial.gov.tw/).
The directory-level usage and attribution notice is available in
[`data/public/README.md`](data/public/README.md).

`data/public/public_insult_authorities.csv` contains 21 manually verified
constitutional judgments, Supreme Court criminal judgments, and Judicial Yuan
interpretations used by the legal-authority retrieval tool. Each row retains
its citation, source type, original location, official URL, and verification
status so the UI can keep source text separate from AI analysis.

The public snapshot was finalized on **2026-10-02**. Release checksums are:

| File | SHA-256 |
| --- | --- |
| `public_insult_judgments.jsonl` | `3b886dd61c43b3af0e2e7a2edb6204e753058b1a4458da438061f09c81d1457c` |
| `public_insult_authorities.csv` | `d428434d940f7325fb7d8e86c59d7d825fdc6379c08e28a8e2ea08258505aa67` |

## Processing

The public corpus keeps the Judicial Yuan public version of each judgment's
crime-fact, sentencing, and disposition text verbatim so retrieval results can
quote the legal source rather than an AI rewrite. The build script:

- removes local source filenames and spreadsheet row numbers;
- retains public institutional metadata such as court, case number, judgment
  date, and presiding judge;
- preserves the train, validation, and test split labels for reproducibility.

The project does not restore information hidden by the Judicial Yuan and does
not claim to reproduce a non-public case file. It republishes only the public
judgment version present in the source snapshot and performs no additional
paraphrasing or identity masking on that text.

The retrieval summaries are used only for embedding and reranking. Tools return
the reviewed source fields as structured artifacts, and the UI renders those
artifacts directly instead of asking a language model to reproduce the text.

The runtime RAG demo searches all 2,154 published records, including records
whose original SFT split labels are `train`, `validation`, or `test`. TAIDE
validation and test evaluation does not call RAG or consume retrieval results;
the RAG demo must not be presented as out-of-sample evidence for TAIDE.

## Live statute source

Current statutory text is retrieved at runtime from the Ministry of Justice
[Laws & Regulations Database](https://law.moj.gov.tw/) and is not copied into
the version-controlled corpus. Each result includes its official source URL.
The website publishes its materials under its
[Government Website Open Data Declaration](https://law.moj.gov.tw/Service/Copyright.aspx),
which adopts the Government Open Data License, Version 1.0 and requires source
attribution.

The committed public artifact is sufficient to run the GitHub project. Project
maintainers can rebuild it from `data/rag/public_insult_sft_retrieval.jsonl`, a
local preparation file intentionally excluded from Git, with:

```bash
python scripts/build_public_judgment_dataset.py
```

## First-run indexing

The Chroma vector database is a generated local cache and is not committed to
GitHub. On the first similar-judgment or legal-authority search, the application
loads the embedding model and builds the required collection automatically.
This can take several minutes depending on the machine; later searches reuse
the local cache. A digest covering the source content, index schema, embedding
model, and pinned revision rebuilds the collection when any input changes.

The indexes can also be prepared before opening the UI:

```bash
python scripts/ingest_judgments.py
python scripts/ingest_authorities.py
```

## Attribution and terms

Judicial Yuan materials are used under its
[Open Data License Terms](https://opendata.judicial.gov.tw/LicenseTerms). This
repository is an independently developed research project and is not produced,
endorsed, or certified by the Judicial Yuan or any court. The corpus is a
processed snapshot; the limited field selection and text-preservation policy
are described above.

The repository-level [MIT License](LICENSE) covers the project's original
software and documentation; it does not relicense the Judicial Yuan or
Ministry of Justice source materials. Principal third-party model and data
terms are summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Corrections and removals

Judgments may later be corrected or withdrawn by the source. When a source
decision is no longer public, the corresponding record should be removed from
the public corpus. Add its `case_id` to
`data/public/withdrawn_case_ids.txt`, rebuild the public artifact, and publish
the resulting corpus change. The source-content hash causes each local Chroma
index to rebuild automatically on its next use. Corrections or removal requests
can be submitted through the repository issue tracker with the court and case
number. Do not include additional personal data in the request.

## Files intentionally excluded

The following are local preparation or generated artifacts and are not required
to run the public demo:

- raw spreadsheet exports and provenance tables;
- intermediate SFT datasets and training outputs;
- Chroma database files, which are generated from the public corpus;
- model checkpoints and adapters, which are distributed separately on Hugging
  Face.
