"""Resolve filing source identity (ticker, fiscal year) from filenames and workspace metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GENERIC_SOURCE_FILES = frozenset({"source.pdf"})
SOURCE_ID_RE = re.compile(r"(?P<ticker>[A-Z]+)_FY(?P<year>\d{4})_", re.IGNORECASE)


def parse_source_id(source_file: str) -> tuple[str | None, str | None]:
    match = SOURCE_ID_RE.match(source_file or "")
    if not match:
        return None, None
    return match.group("ticker").upper(), f"FY{match.group('year')}"


def normalize_ticker(value: str) -> str:
    return (value or "").strip().upper()


def normalize_fiscal_year(value: str) -> str:
    cleaned = (value or "").strip().upper()
    if not cleaned:
        return cleaned
    if cleaned.startswith("FY"):
        return cleaned
    if cleaned.isdigit() and len(cleaned) == 4:
        return f"FY{cleaned}"
    return cleaned


def normalize_filter_values(values: str | list[str] | None, *, kind: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    normalizer = normalize_ticker if kind == "ticker" else normalize_fiscal_year
    normalized = [normalizer(value) for value in values if str(value or "").strip()]
    return normalized or None


def resolve_source_file(
    source_file: str | None,
    *,
    original_filename: str | None = None,
    json_path: Path | None = None,
) -> str:
    source_file = (source_file or "").strip()
    original_filename = (original_filename or "").strip()

    if source_file and source_file.lower() not in GENERIC_SOURCE_FILES:
        return source_file
    if original_filename:
        return original_filename
    if json_path is not None:
        meta_path = json_path.parent / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resolved = (meta.get("original_filename") or "").strip()
            if resolved:
                return resolved
    return source_file or "unknown.pdf"


def filing_identity(
    source_file: str | None,
    *,
    original_filename: str | None = None,
    json_path: Path | None = None,
) -> tuple[str, str | None, str | None]:
    resolved = resolve_source_file(
        source_file,
        original_filename=original_filename,
        json_path=json_path,
    )
    ticker, fiscal_year = parse_source_id(resolved)
    return resolved, ticker, fiscal_year


def replace_source_file_value(value: Any, old_source: str, new_source: str) -> Any:
    if isinstance(value, str):
        if value == old_source:
            return new_source
        if old_source and value.startswith(f"{old_source}::"):
            return f"{new_source}{value[len(old_source):]}"
        return value
    if isinstance(value, list):
        return [replace_source_file_value(item, old_source, new_source) for item in value]
    if isinstance(value, dict):
        return {key: replace_source_file_value(item, old_source, new_source) for key, item in value.items()}
    return value


def patch_json_source_file(path: Path, new_source: str, *, old_source: str = "source.pdf") -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    current = payload.get("source_file")
    if current not in {old_source, new_source}:
        return False
    if current == new_source and not any(
        chunk.get("source_file") == old_source for chunk in payload.get("chunks", [])
    ):
        return False
    patched = replace_source_file_value(payload, old_source, new_source)
    patched["source_file"] = new_source
    path.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def patch_workspace_source_files(workspace_dir: Path) -> dict[str, Any]:
    meta_path = workspace_dir / "metadata.json"
    if not meta_path.exists():
        return {"workspace": workspace_dir.name, "patched": False, "reason": "missing metadata.json"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    original_filename = (meta.get("original_filename") or "").strip()
    if not original_filename:
        return {"workspace": workspace_dir.name, "patched": False, "reason": "missing original_filename"}

    changed_files: list[str] = []
    for filename in ("sections.json", "assets.json", "chunks.json"):
        path = workspace_dir / filename
        if patch_json_source_file(path, original_filename):
            changed_files.append(filename)

    return {
        "workspace": workspace_dir.name,
        "patched": bool(changed_files),
        "source_file": original_filename,
        "ticker": parse_source_id(original_filename)[0],
        "fiscal_year": parse_source_id(original_filename)[1],
        "changed_files": changed_files,
    }
