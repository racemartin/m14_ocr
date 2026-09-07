# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- `JsonlDatasetRepository` (`src/chsa_triage/infrastructure/adapters/jsonl_dataset_repository.py`) rereads and rewrites the ENTIRE JSONL file on every `sauvegarder()`/`sauvegarder_plusieurs()` call. Never call `sauvegarder()` inside a per-item loop over more than a few hundred items — it's O(n²) and infeasible on real corpus sizes (confirmed on the real 147k-example pivot dataset). Always accumulate into a list and call `sauvegarder_plusieurs()` once.
- `LecteurCorpusFichierLocal` loads the whole source file into a pandas DataFrame at once unless constructed with `taille_bloc=N` (chunked pandas read, CSV/JSONL only). Real files here can be ~1GB (`ultramedical_preference.jsonl`, 966MB/109k records) on a ~5.8GB-RAM box — pass `--taille-bloc` (both `profiler_corpus.py --bloque` and `construire_dataset_pivot.py --taille-bloc`) for any source that large, or expect an OOM kill.
- Real Presidio/spaCy anonymization throughput measured on the real dataset: ~130-550ms per text field depending on length/language, dominated by spaCy NLP inference, not I/O. Full-dataset anonymization of the 147k-example pivot is estimated at ~19h wall time with the current per-field `AnalyzerEngine.analyze` calls (no batching). `anonymiser_dataset.py --limite N` (default 5000) makes this incremental/resumable via `ExemplePivot.anonymise` — it never reprocesses `anonymise=True` records, so repeated runs with a larger N (or `full`) pick up where the last one left off. `decouper_splits.py` only splits `anonymise=True` records — rerun it after every new anonymization wave. See `docs/02_etape1_donnees/00_couverture_exigences_officielles.md` for the measured breakdown and wave history.
- `data/raw/`, `data/processed/`, `data/splits/` are gitignored — a fresh worktree/clone has none of the 6 source files. Regenerate them with the `telecharger_corpus.py` commands in the README before running any later pipeline step; no network access means the pipeline literally cannot be exercised end-to-end.
- Regenerating `docs/diagrams/**/*.puml` outputs: this box has no `plantuml` CLI and no matching-JDK PlantUML jar preinstalled (system Java is 17; the latest PlantUML jar needs Java 21 for its PDF path, throwing `UnsupportedClassVersionError`). Workaround that worked: download an older jar built for Java 17 (e.g. `plantuml-1.2024.8.jar` from GitHub releases) for `-tpng -tsvg` (run those two together, not combined with `-tpdf` — a PDF failure aborts the whole invocation before png/svg get written), then convert the generated `.svg` to `.pdf` with `cairosvg` (`uv pip install --python <venv> cairosvg`, needs system `libcairo2` which is present) since PlantUML's own PDF export needs Apache Batik+FOP transcoders not bundled in the jar.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
