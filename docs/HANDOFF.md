# Consolidation handoff — 2026-10-14

## Two-sentence brief

Next, repair preference learning and calibration, replace placeholder validation
with held-out human preference tests, and compare residualized embeddings against
raw embeddings and simple supervised baselines. The central hypothesis still to
test is whether removing selected SVD directions improves style matching across
unfamiliar topics without discarding useful tone information—and whether that
improvement reduces client review effort and increases approval of agent responses.

## What was consolidated

`n-step-svd-decomp` at `5bd0fd6c90941d290517514efff292b61dce64bb` contained the
proposal, an ignore file and a short research note. The companion directory's
remote is `Unobtainiumrock/unsupervised-learning-svd-social-media`, but its local
`master` branch had no commits. This handoff imports selected staged and untracked
source files, not Git history. The original working tree remains untouched.

The manifest records source checksums, destination paths, handoff checksums and
reasons for edits. Historical documentation is retained under `docs/history/`;
its claims and deployment commands are not current acceptance criteria. The
previous research notes discuss UX tradeoffs and per-subspace VAEs. Those remain
exploratory, downstream of validating the basic preference model.

No private corpus, real content examples, fitted matrices, generated reports,
credentials, build output or nested Git repository is included. `golden-example.md`
was deliberately omitted because its content examples are unnecessary for the
source handoff. Notebook outputs were empty in the imported notebook.

## Required data and services

The offline example and core unit tests need no service credentials or real data.
The web calibration loader and interactive CLI expect these local artifacts:

- `data/embeddings/ugc-posts_COMPLETE_embeddings.h5`: `embeddings` is an N×1536
  numeric matrix; `ids` are byte strings; optional `texts` are byte strings.
- `data/reports/optimal_160_projection_matrix.npy`: a 1536×1536 projection matrix.
- `data/reports/optimal_130_tone_axes.npy`: 130×1536 axis vectors.

Other experimental paths use additional files under `data/models/` and
`data/production/`; inspect their loaders before running them. No authenticated
artifact download location is supplied. A recipient needs authorized data from
the owner or must generate a compatible dataset and projections; the offline
example intentionally does not fabricate a production corpus.

Supabase supplies optional content lookup from `ugc_posts` (`post_text`) and
`ugc_videos` (`script`). OpenAI is used by embedding/generation experiments;
Pinecone by download utilities. Configuring those services may incur costs.
Nothing in the offline checks calls them. The API/frontend and CLI use different
calibration paths. The home-page onboarding currently navigates directly to
results; the calibration route is a separate prototype surface.

## Confirmed limitations

| Finding | Evidence / consequence |
| --- | --- |
| Semantic labels are unproven | Removing leading singular directions does not guarantee removing topic or preserving tone. An orthogonal split can contain mixed semantics and dependent scores. |
| Iterative SVD is not automatic semantic discovery | With exact SVD and distinct singular values it extracts successive blocks from one decomposition. An elbow plot does not label those blocks. |
| API calibration can succeed with NaN preferences | `scripts/utilities/tone_calibration_system_fixed.py` trains Ridge on all-one targets with an intercept, then normalizes zero coefficients. Synthetic reproduction reported calibrated after 5 choices while every coefficient and content score was NaN. |
| Alternative logistic learner fails | `residualized_tone_space/core/tone_axis_learning.py` feeds only class 1 to LogisticRegression; the actual method raised a two-class requirement error. |
| CLI confidence is heuristic | `scripts/core/interactive_post_finder.py` derives confidence from rounds and shrinking radius; candidate quality is not measured by that score. |
| Semantic validation is unfinished | `residualized_tone_space/utils/validation.py` returns placeholder zero metrics regardless of data/labels. |
| Five selections or nine rounds are not validated guarantees | No held-out preference accuracy or calibrated uncertainty establishes such a stopping rule. |
| Corpus transfer and context remain open | Social-post embeddings may not generalize to sales dialogue; style may vary by client, role and conversation stage. Text embeddings do not measure audio delivery or tool trajectories. |

The review used synthetic inputs and isolated method calls, not production data or
an end-to-end HTTP test. Core algorithms are preserved in this consolidation;
known failures are not silently marked fixed. Reproduce the API NaNs, logistic
learner error and placeholder metrics with:

```bash
uv run python examples/reproduce_known_failures.py
```

This executes the actual learner methods with synthetic inputs, bypassing only
external-artifact initialization. Its successful exit confirms the documented
defects still exist; it is not a preference-quality test.

## Prioritized follow-up

1. Fix pairwise preference training, finite-vector handling and success criteria.
   Add regressions that reproduce the original failures and fail if the defects
   return. Test conflicting choices, identical candidates and empty input.
2. Implement human-labelled evaluation with content-controlled style pairs.
   Hold out whole topics, conversations and speakers to reduce leakage. Compare
   raw embeddings, residualized embeddings, a simple supervised preference model,
   and a rubric-based judge. Report uncertainty and reviewer disagreement.
3. Replace scheduled confidence with demonstrated reliability on unseen choices;
   allow insufficient evidence and keep factual/policy requirements separate from
   stylistic preference.
4. Preserve client, reviewer, stage, chosen/rejected examples and feedback reasons.
   Test context-specific preferences before relying on one averaged vector.
5. Pilot client calibration and example retrieval for Hobbes. Measure approval on
   unseen agent responses and review minutes/revision rounds. Consider bandits and
   fine-tuning only after establishing useful feedback and data permissions.

## Verification record

Checked on Python 3.12.3, Node 24.19.0 and npm 11.17.0:

| Check | Result |
| --- | --- |
| `uv sync --locked` | Pass using the imported lockfile (NumPy 2.3.2, scikit-learn 1.7.1). |
| `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q` | 16 passed; existing tests cover 12% of the Python package and do not establish calibration quality. |
| `uv build` | Wheel and source distribution build successfully. The wheel includes the `scripts` package required by the API. |
| Installed-wheel API import from outside checkout | Pass; API and companion learner import without relying on the source tree. |
| `npm ci --no-audit --no-fund` and `npm run build` in `frontend/` | Pass; build retains three unused-variable warnings in ResultsPage and a stale Browserslist-data notice. |
| `uv run --no-project --with numpy python examples/offline_decomposition.py` | Pass: reconstruction, orthogonality, one-SVD equivalence and synthetic dependence counterexample. |
| `uv run python examples/reproduce_known_failures.py` | Reproduces API NaNs despite successful calibration, logistic one-class error and placeholder validation metrics with the locked dependencies. |
| `python scripts/verify_import.py --source ../unsupervised-learning` | All 84 imported files match their documented source/handoff hashes. |
| `git diff --cached --check` | Reports existing trailing whitespace/blank EOF lines in imported source and historical notes; retained to preserve original bytes, not introduced by application logic changes. |
| Public-snapshot inspection | No credential matches or unresolved detect-secrets findings. The final scan flags 91 SHA-256 entries and the base commit ID in the provenance manifest; reviewed as provenance-only false positives. No private data, generated arrays, installed dependencies or credential files in the selected snapshot or build artifacts. |

The original source tree's Git status and imported file bytes are unchanged. The
only application packaging adjustment is including `scripts` in the wheel;
application source files retain their original bytes. No authenticated service,
full web calibration, deployment or real-data benchmark was run. Dependency
security upgrades and production hardening are outside this source consolidation.
