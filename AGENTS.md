# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- `JsonlDatasetRepository` (`src/chsa_triage/infrastructure/adapters/jsonl_dataset_repository.py`) rereads and rewrites the ENTIRE JSONL file on every `sauvegarder()`/`sauvegarder_plusieurs()` call. Never call `sauvegarder()` inside a per-item loop over more than a few hundred items — it's O(n²) and infeasible on real corpus sizes (confirmed on the real 147k-example pivot dataset). Always accumulate into a list and call `sauvegarder_plusieurs()` once.
- `LecteurCorpusFichierLocal` loads the whole source file into a pandas DataFrame at once unless constructed with `taille_bloc=N` (chunked pandas read, CSV/JSONL only). Real files here can be ~1GB (`ultramedical_preference.jsonl`, 966MB/109k records) on a ~5.8GB-RAM box — pass `--taille-bloc` (both `profiler_corpus.py --bloque` and `construire_dataset_pivot.py --taille-bloc`) for any source that large, or expect an OOM kill.
- Real Presidio/spaCy anonymization throughput measured on the real dataset: ~130-550ms per text field depending on length/language, dominated by spaCy NLP inference, not I/O. Full-dataset anonymization of the 147k-example pivot is estimated at ~19h wall time with the current per-field `AnalyzerEngine.analyze` calls (no batching). Any request to "run anonymisation on the real dataset" needs a product decision on scope/approach first — see `docs/02_etape1_donnees/00_couverture_exigences_officielles.md` for the measured breakdown.
- `data/raw/`, `data/processed/`, `data/splits/` are gitignored — a fresh worktree/clone has none of the 6 source files. Regenerate them with the `telecharger_corpus.py` commands in the README before running any later pipeline step; no network access means the pipeline literally cannot be exercised end-to-end.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
