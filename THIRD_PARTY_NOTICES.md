# Third-Party Notices

This file records the principal third-party data, models, and online services
used by this project. It is an attribution summary, not a replacement for the
upstream license texts.

The repository-level [MIT License](LICENSE) applies only to the original source
code, project documentation, and project-owned UI assets created for this
repository. It does not relicense the public legal data, third-party model
weights, Python packages, or online services listed below.

## Public legal sources

### Judicial Yuan judgments and legal authorities

- Used by:
  `data/public/public_insult_judgments.jsonl` and
  `data/public/public_insult_authorities.csv`
- Provider: Judicial Yuan, Republic of China (Taiwan)
- Sources:
  [Judgment System](https://judgment.judicial.gov.tw/) and
  [Open Data Platform](https://opendata.judicial.gov.tw/)
- Terms:
  [Judicial Yuan Open Data License Terms](https://opendata.judicial.gov.tw/LicenseTerms)

The files preserve selected text from the Judicial Yuan's public versions of
judgments and authorities. They are not covered by this repository's MIT
License. The source must be attributed, third-party rights must be respected,
and use of public personal data remains the user's responsibility under
applicable law. See [DATA_SOURCES.md](DATA_SOURCES.md) and
[data/public/README.md](data/public/README.md) for corpus scope and governance.

### Laws and regulations queried at runtime

- Provider: Ministry of Justice, Republic of China (Taiwan)
- Source: [Laws & Regulations Database](https://law.moj.gov.tw/)
- Terms:
  [Government Website Open Data Declaration](https://law.moj.gov.tw/Service/Copyright.aspx),
  which adopts the Government Open Data License, Version 1.0

The application retrieves current statutory text at runtime and does not
bundle a copy of the Ministry of Justice database. Official source links and
attribution are displayed with each result.

## Models

### TAIDE and Meta Llama 3.1

- Base model:
  [`taide/Llama-3.1-TAIDE-LX-8B-Chat`](https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat)
- Upstream foundation model: Meta Llama 3.1
- Governing terms:
  the TAIDE model license presented on the model page and the applicable
  [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)

The TAIDE repository is gated. Each user must review and accept its current
terms before downloading the model. This GitHub repository does not include
TAIDE or Meta Llama weights.

### Public-insult LoRA adapters

- [Adapter A](https://huggingface.co/a97141481/Llama-3.1-TAIDE-Public-Insult-A-LoRA)
- [Adapter B](https://huggingface.co/a97141481/Llama-3.1-TAIDE-Public-Insult-B-LoRA)
- [Adapter C](https://huggingface.co/a97141481/Llama-3.1-TAIDE-Public-Insult-C-LoRA)

These project adapters are distributed separately on Hugging Face and are not
included in this repository. Use of an adapter also requires access to the
TAIDE base model and compliance with the current terms displayed on both the
adapter page and the TAIDE model page.

### Retrieval models

- [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3): MIT License; pinned to commit `5617a9f61b028005a4858fdac845db406aefb181`
- [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3): Apache License 2.0; pinned to commit `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

The weights are downloaded from Hugging Face on first use and are not committed
to this repository.

## Dependencies and hosted services

Python dependencies are listed in `requirements*.txt` and remain subject to
their respective upstream licenses. The project also integrates with OpenAI,
LangSmith, and Hugging Face services; use of those services is governed by the
terms associated with each user's own account. No API keys or access tokens are
included in the repository.

No upstream provider, court, or government agency endorses or certifies this
independent research project.
