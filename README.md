# Face ID for Tone — research prototype

This repository consolidates the N-step SVD proposal and the previously uncommitted
`unsupervised-learning` companion application. It explores learning communication
preferences through comparisons between examples, then using those preferences
for retrieval and generation.

**Status:** source handoff, not a validated tone model or production application.
SVD guarantees geometric orthogonality, not a semantic topic/tone split. The
calibration paths have known defects, and the full application needs external
artifacts that are deliberately not included. Start with the offline example.

## Start here: offline example

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/), then run from this repo:

```bash
uv run --no-project --with numpy python examples/offline_decomposition.py
```

This generates a tiny synthetic matrix in memory and checks reconstruction,
orthogonality, and equivalence to blocks of one SVD. It also demonstrates that
orthogonal components can still mix topic and tone and remain statistically
dependent. It makes no network calls, reads no corpus, and writes no data files.
The first uv invocation may download NumPy.

## Application entrypoints

| Path | Role | Requirements / status |
| --- | --- | --- |
| `residualized_tone_space/api/main.py` | FastAPI backend used by the React UI | External arrays/corpus; known calibration failures |
| `frontend/` | React interface | Node/npm; backend at localhost:8000; prototype flows remain incomplete |
| `scripts/utilities/tone_calibration_system_fixed.py` | API's Ridge-based comparison learner | Known zero-vector/NaN defect despite the historical filename |
| `run_pipeline.py` | Experimental interactive CLI | External artifacts, optional Supabase/OpenAI; separate confidence heuristic |
| `residualized_tone_space/core/` | Decomposition, projection and alternative preference learner | Unit tests included; logistic learner has a known one-class error |
| `scripts/analysis/`, `scripts/investigation/`, `notebooks/` | Research utilities | Some need private data, GPU dependencies or services beyond the base environment |
| `netlify/`, Docker files | Historical deployment experiments | Preserved for inspection, not validated deployment instructions |

### Install and check the source

```bash
uv sync --locked
uv run pytest
uv build
cd frontend
npm ci --no-audit --no-fund
npm run build
```

The lockfiles are included. Results and remaining limitations are recorded in
[the handoff](docs/HANDOFF.md). Passing geometry tests does not validate semantic
separation or client preference accuracy.

### Explore the web application locally

After supplying the data described in [the handoff](docs/HANDOFF.md), run the
backend **from the repository root**:

```bash
uv run uvicorn residualized_tone_space.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm start
```

Visit localhost:3000 and localhost:8000/docs. Optional service configuration is
listed in `.env.example`; export values into the server process when needed.
The API does not automatically load this file. Do not interpret a running HTTP
server as successful calibration: startup logs data-loading failures and can
leave the calibration system unavailable.

## What to work on next

Repair preference learning and calibration, implement held-out human preference
evaluation, and compare raw embeddings with residualized and supervised
baselines. Test whether residualization improves style matching across unfamiliar
topics without discarding useful tone information, and whether that translates
into fewer client revisions and better approval of agent responses.

For a Hobbes pilot, start with client-approved response comparisons and examples;
factual accuracy, competitor policy, voice delivery and agent actions require
separate evaluation.

## Handoff and provenance

- [Current status, known failures and next steps](docs/HANDOFF.md)
- [Original mathematical proposal, with historical claims marked](docs/history/original-proposal.md)
- [Historical companion README](docs/history/companion-readme.md)
- [Import manifest and source checksums](docs/import-manifest.json)

Verify the consolidated snapshot with `python scripts/verify_import.py`. The
manifest records the source bytes separately from the reviewed handoff bytes;
documentation edits and packaging adjustments are explicitly identified. The
original local working tree was left intact. Private datasets, model artifacts,
credentials and generated output were not imported.
