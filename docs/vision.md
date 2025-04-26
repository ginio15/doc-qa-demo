## Core user stories (MVP)

| # | As a … | I want to … | So that … |
|---|---------|-------------|-----------|
| 1 | ⚖️  Lawyer | upload one or more documents | I can question them without reading everything |
| 2 | ⚡️  … | ask a natural-language question | the system reasons over full text |
| 3 | 🔍  … | see the exact quoted passage | I trust the answer |
| 4 | 🗂️  … | switch between docs in one search | I don’t repeat queries per file |


## Success metrics (demo scope)

| Metric | Target | Measurement method |
|--------|--------|--------------------|
| Citation accuracy | ≥ 85 % | Manual spot-check of 20 Q&A |
| Answer latency    | ≤ 4 s  p95 | cURL timing in local dev |
| Query cost        | ≤ $0.01 per Q | OpenAI usage dashboard |
