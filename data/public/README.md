# Public Runtime Data

This directory contains the public, version-controlled data required by the
runtime retrieval tools. The files are intentionally small enough to support a
reproducible local demo without committing private preparation files or a
generated vector database.

## Files

- `public_insult_judgments.jsonl`: 2,154 public-insult judgments used for
  similar-judgment retrieval.
- `public_insult_authorities.csv`: 21 manually verified constitutional
  judgments, Supreme Court criminal judgments, and Judicial Yuan
  interpretations.
- `withdrawn_case_ids.txt`: case identifiers excluded after a source judgment
  is corrected, withdrawn, or no longer publicly available.

Detailed coverage, processing rules, and the removal workflow are documented
in [DATA_SOURCES.md](../../DATA_SOURCES.md).

## License and attribution

The data files in this directory are **not licensed under the repository's MIT
License**. Their official source text remains subject to the
[Judicial Yuan Open Data License Terms](https://opendata.judicial.gov.tw/LicenseTerms).

When using or redistributing these data files, retain a visible attribution
equivalent to:

> Data source: Judicial Yuan Judgment System and Judicial Yuan Open Data
> Platform, Republic of China (Taiwan). Fields selected and arranged by this
> project.

The files are a processed snapshot of public source material. They are not an
official publication of the Judicial Yuan, do not imply endorsement, and may
not reflect later corrections or withdrawals. Users remain responsible for
respecting applicable privacy, personal-data, and third-party-rights
requirements.

Snapshot date and SHA-256 checksums are recorded in
[DATA_SOURCES.md](../../DATA_SOURCES.md).
