# Table layout benchmark (standalone)

Compares **layout/table extraction** approaches on hard MSFT FY2025 10-K cases **without modifying** the main chunking pipeline.

## What this measures

Golden cases in `cases/msft_fy2025_hard_tables.json`:

| Case | Pages | Why it's hard |
|------|-------|----------------|
| Unearned Revenue schedule | 42–43 | Header-only `(In millions)` on p42; data on p43 |
| Dividend declaration | 32–33 | Cross-page column header |
| Segment Total Operating Income | 37–38 | Subtotal row on page break |
| Summary revenue/EPS | 33 | Wide same-page table (sanity) |

Scoring checks:

- Required cell text present (e.g. `67,265`, `128,528`)
- Cross-page linkage signal (when applicable)
- Latency + table count

## Run (minimal — baseline + find_tables only)

```bash
cd benchmarks/table_layout
python run_benchmark.py --only baseline,find_tables
```

Uses existing `assets.json` for baseline (your `table_pipeline_v2` output).

## Run with optional tools

```bash
pip install -r requirements-benchmark.txt
python run_benchmark.py
```

Adapters:

| Key | Tool | Notes |
|-----|------|-------|
| `baseline` | `assets.json` | Your production path |
| `find_tables` | PyMuPDF | Raw detector used before heuristics |
| `pymupdf4llm` | `pip install pymupdf4llm` | Layout auto-on; closed weights |
| `docling` | `pip install docling` | MIT, TableFormer; first run downloads models |
| `doclayout_yolo` | stub | MinerU stack not run by default (heavy) |

Results written to `results/benchmark_*.json` and `*.md`.

## Expected conclusion for **this** repo

These tools are **not drop-in replacements** for `table_pipeline`:

1. They detect layout/tables; they do **not** map tables to **Item 7 > Unearned Revenue** subsections.
2. **Cross-page header-only bands** still need explicit merge logic (or you accept broken markdown tables at page breaks).
3. Best use: replace **detection/enrichment** layer only, keep TOC sectioning + chunk refs.

**PyMuPDF4LLM Layout** — good for markdown RAG export; weights not inspectable.

**Docling** — best open-source table structure; add section linker + cross-page pass.

**DocLayout-YOLO / MinerU** — fast bbox detector; same linker gap as Docling.
