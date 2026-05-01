"""CSV / structured data ingestion — row serialisation + schema embedding."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from config.settings import get_settings

cfg = get_settings()

ROWS_PER_CHUNK = 10  # embed this many rows together


@dataclass
class CSVChunk:
    content: str           # serialised row(s) as natural language
    source: str
    doc_hash: str
    row_start: int
    row_end: int
    columns: list[str]
    summary: str           # one-line schema description
    doc_type: str = "csv"
    metadata: dict = field(default_factory=dict)

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "source": self.source,
                "doc_hash": self.doc_hash,
                "doc_type": self.doc_type,
                "row_start": self.row_start,
                "row_end": self.row_end,
                "columns": ", ".join(self.columns),
                "summary": self.summary,
                **self.metadata,
            },
        )


class CSVIngestionPipeline:
    """
    Converts tabular data into searchable text chunks.

    Strategy
    --------
    - Auto-detect dtypes (numeric, categorical, date)
    - Generate a human-readable schema summary (used in every chunk)
    - Serialise rows as "Column: Value, Column: Value …" sentences
    - Group ROWS_PER_CHUNK rows per chunk to give the LLM context
    - Include aggregate stats (min/max/mean) as a separate header chunk
    """

    def ingest(self, file_path: str | Path) -> list[CSVChunk]:
        path = Path(file_path)
        df = pd.read_csv(path)
        return self._process(df, path.name, open(path, "rb").read())

    def ingest_bytes(self, data: bytes, filename: str) -> list[CSVChunk]:
        df = pd.read_csv(io.BytesIO(data))
        return self._process(df, filename, data)

    # ── Private ────────────────────────────────────────────────────────

    def _process(
        self, df: pd.DataFrame, filename: str, raw: bytes
    ) -> list[CSVChunk]:
        doc_hash = hashlib.sha256(raw).hexdigest()[:16]
        df = self._clean(df)
        summary = self._schema_summary(df)
        chunks: list[CSVChunk] = []

        # 1. Stats header chunk
        stats_text = self._stats_chunk(df, summary)
        chunks.append(
            CSVChunk(
                content=stats_text,
                source=filename,
                doc_hash=doc_hash,
                row_start=0,
                row_end=len(df),
                columns=df.columns.tolist(),
                summary=summary,
                metadata={"chunk_kind": "stats"},
            )
        )

        # 2. Row chunks
        for start in range(0, len(df), ROWS_PER_CHUNK):
            end = min(start + ROWS_PER_CHUNK, len(df))
            rows_text = self._serialise_rows(df.iloc[start:end], summary)
            chunks.append(
                CSVChunk(
                    content=rows_text,
                    source=filename,
                    doc_hash=doc_hash,
                    row_start=start,
                    row_end=end,
                    columns=df.columns.tolist(),
                    summary=summary,
                    metadata={"chunk_kind": "rows"},
                )
            )

        return chunks

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how="all").reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def _schema_summary(df: pd.DataFrame) -> str:
        parts = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                parts.append(f"{col} (numeric)")
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                parts.append(f"{col} (date)")
            else:
                n_unique = df[col].nunique()
                parts.append(f"{col} (text, {n_unique} unique)")
        return "Table with columns: " + ", ".join(parts)

    @staticmethod
    def _stats_chunk(df: pd.DataFrame, summary: str) -> str:
        lines = [f"Dataset overview — {summary}", f"Total rows: {len(df)}", ""]
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna()
            lines.append(
                f"{col}: min={s.min():.2f}, max={s.max():.2f}, "
                f"mean={s.mean():.2f}, std={s.std():.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _serialise_rows(rows: pd.DataFrame, summary: str) -> str:
        lines = [f"Context: {summary}", ""]
        for _, row in rows.iterrows():
            pairs = ", ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if pd.notna(val)
            )
            lines.append(f"Record — {pairs}.")
        return "\n".join(lines)