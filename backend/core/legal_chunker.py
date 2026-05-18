"""legal_chunker.py – Legal document normalizer for Vietnamese Legal RAG.

Design philosophy
-----------------
LightRAG handles its own internal chunking (default: ~1200 token windows with
overlap).  This module does NOT replace that chunking — it **normalizes** the
raw extracted text so that LightRAG's natural ``\\n\\n`` split boundaries align
with legal article (Điều) boundaries rather than landing arbitrarily in the
middle of an article.

Primary API
-----------
``normalize_for_lightrag(text, source_file)``
    Returns a single normalized text string ready to be passed directly to
    ``rag.ainsert(text, file_paths=[filename], split_by_character="\\n\\n")``.

What the normalizer does
------------------------
1. Detects Điều markers and ensures each article is surrounded by ``\\n\\n``.
2. Prepends a breadcrumb ``[Chương … > Mục …]`` to each Điều block so that
   citation context is embedded in the text LightRAG indexes.
3. Collapses accidental double-newlines *inside* an article (between Khoản
   lines) to single newlines — preventing LightRAG from splitting mid-article.
4. Normalizes Chương / Mục header formatting for consistent entity extraction.
5. Falls back gracefully: if no Điều markers are detected the original text is
   returned unchanged (LightRAG handles generic splitting).

Internal helpers (kept for potential standalone use)
----------------------------------------------------
``LegalChunk``, ``chunk_markdown()``, ``chunk_text()``, ``chunks_to_strings()``
are preserved but are no longer called from the upload route.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Regex patterns for Vietnamese legal structure markers
# ---------------------------------------------------------------------------

# CHƯƠNG I / CHƯƠNG II / CHƯƠNG 1 …  (with optional Markdown heading prefix)
_RE_CHUONG = re.compile(
    r"^(?:#{1,6}\s*)?(CHƯƠNG\s+[IVXLCDM\d]+(?:\s+.*)?)",
    re.IGNORECASE | re.MULTILINE,
)

# MỤC 1 / MỤC I …
_RE_MUC = re.compile(
    r"^(?:#{1,6}\s*)?(MỤC\s+[IVXLCDM\d]+(?:\s+.*)?)",
    re.IGNORECASE | re.MULTILINE,
)

# Điều 1. / Điều 12. / **Điều 3.** / # Điều 5:
_RE_DIEU = re.compile(
    r"^(?:#{1,6}\s*|\*{1,2})?(?:Điều|ĐIỀU)\s+(\d+)[\.\:]",
    re.MULTILINE,
)

# Minimum characters for a chunk to be considered valid (avoid orphan headers)
_MIN_CHUNK_LEN = 30

# Maximum chunk size guard (≈ 6000 chars ≈ 1500 tokens)
_MAX_CHUNK_CHARS = 6000


# ---------------------------------------------------------------------------
# Primary public API  ← use this in routes.py
# ---------------------------------------------------------------------------

def normalize_for_lightrag(text: str, source_file: str = "") -> str:
    """Normalize a legal document text for optimal LightRAG ingestion.

    Returns a **single string** ready for::

        rag.ainsert(normalized, file_paths=[filename], split_by_character="\\n\\n")

    Normalization steps:
    - Ensures ``\\n\\n`` boundaries at every Điều so LightRAG chunks on article
      edges rather than at arbitrary token positions.
    - Embeds breadcrumb context (Chương / Mục) at the start of each Điều block
      to support accurate citation after retrieval.
    - Collapses internal double-newlines *within* a Điều (between Khoản lines)
      to single newlines so LightRAG does not split in the middle of an article.
    - Normalizes CHƯƠNG / MỤC header lines.
    - Falls back to the original text unchanged if no Điều markers are found.

    Args:
        text:        Raw extracted text (plain text or Markdown).
        source_file: Original filename; embedded in the preamble for citation.

    Returns:
        Normalized text string.
    """
    matches = list(_RE_DIEU.finditer(text))
    if not matches:
        # No legal structure detected — return as-is; LightRAG handles it.
        return text.strip()

    parts: list[str] = []

    # --- Preamble (text before the first Điều) ---
    preamble = text[: matches[0].start()].strip()
    if preamble:
        preamble_normalized = _normalize_headers(preamble)
        parts.append(preamble_normalized)

    # --- Each Điều block ---
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        dieu_raw = text[start:end]

        dieu_num = match.group(1)
        chuong, muc = _context_before(text, start)

        # Build breadcrumb prefix for citation context
        breadcrumb = _build_breadcrumb(chuong, muc, source_file)

        # Collapse internal double-newlines within this Điều block
        # so LightRAG does not split in the middle of an article.
        dieu_body = _collapse_internal_breaks(dieu_raw.strip())

        block = f"{breadcrumb}{dieu_body}" if breadcrumb else dieu_body
        parts.append(block)

    # Join with double-newline — LightRAG will split exactly here.
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Legacy / internal-use public helpers
# (preserved; no longer called from routes.py)
# ---------------------------------------------------------------------------

@dataclass
class LegalChunk:
    """A single chunk from a legal document with metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.text = self.text.strip()

    @property
    def is_valid(self) -> bool:
        return len(self.text) >= _MIN_CHUNK_LEN


def chunk_markdown(
    markdown_text: str,
    source_file: str = "",
) -> list[LegalChunk]:
    """Split Markdown text into LegalChunk objects (for standalone use).

    Not used in the main upload flow — LightRAG handles internal chunking.
    Kept for unit-testing and potential offline analysis.
    """
    chunks = list(_split_by_dieu(markdown_text, source_file))
    if chunks:
        return [c for c in chunks if c.is_valid]
    return _split_by_paragraph(markdown_text, source_file)


def chunk_text(plain_text: str, source_file: str = "") -> list[LegalChunk]:
    """Chunk plain text using the same strategy as chunk_markdown."""
    return chunk_markdown(plain_text, source_file)


def chunks_to_strings(chunks: list[LegalChunk]) -> list[str]:
    """Convert LegalChunk list to plain strings."""
    return [c.text for c in chunks]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_breadcrumb(chuong: str, muc: str, source_file: str) -> str:
    """Build a short context prefix for a Điều block.

    Example output: ``[Chương I > Mục 2] `` or ``[Chương III] ``
    The breadcrumb is placed on the same line as the Điều header so it does
    not create an extra ``\\n\\n`` split point.
    """
    parts: list[str] = []
    if chuong:
        parts.append(chuong)
    if muc:
        parts.append(muc)
    if not parts:
        return ""
    return "[" + " > ".join(parts) + "] "


def _collapse_internal_breaks(text: str) -> str:
    """Replace double (or more) newlines *inside* a block with a single newline.

    This prevents LightRAG from treating Khoản / Điểm separators as chunk
    boundaries within a single Điều.
    """
    # Collapse 2+ consecutive newlines into one (preserving single newlines)
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_headers(text: str) -> str:
    """Normalize CHƯƠNG / MỤC header lines to plain uppercase text.

    Strips Markdown heading prefixes (##, ###, **…**) so the headers are
    consistent regardless of PDF-to-Markdown conversion style.
    """
    # Strip leading Markdown heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Strip bold markers around header text
    text = re.sub(r"\*{1,2}(CHƯƠNG[^\*]*)\*{1,2}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\*{1,2}(MỤC[^\*]*)\*{1,2}", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _context_before(text: str, pos: int) -> tuple[str, str]:
    """Return the nearest CHƯƠNG and MỤC labels before *pos*."""
    preceding = text[:pos]

    chuong = ""
    for m in _RE_CHUONG.finditer(preceding):
        chuong = m.group(1).strip()

    muc = ""
    for m in _RE_MUC.finditer(preceding):
        muc = m.group(1).strip()

    return chuong, muc


def _split_by_dieu(text: str, source_file: str) -> Iterator[LegalChunk]:
    """Yield one LegalChunk per Điều (internal helper)."""
    matches = list(_RE_DIEU.finditer(text))
    if not matches:
        return

    preamble = text[: matches[0].start()].strip()
    if len(preamble) >= _MIN_CHUNK_LEN:
        chuong, muc = _context_before(text, matches[0].start())
        yield LegalChunk(
            text=preamble,
            metadata={"type": "preamble", "source_file": source_file,
                      "chuong": chuong, "muc": muc},
        )

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        dieu_text = text[start:end].strip()
        dieu_num = match.group(1)
        chuong, muc = _context_before(text, start)

        if len(dieu_text) <= _MAX_CHUNK_CHARS:
            yield LegalChunk(
                text=dieu_text,
                metadata={"type": "dieu", "dieu": dieu_num,
                          "chuong": chuong, "muc": muc,
                          "source_file": source_file},
            )
        else:
            yield from _split_dieu_by_khoan(dieu_text, dieu_num, chuong, muc, source_file)


def _split_dieu_by_khoan(
    dieu_text: str,
    dieu_num: str,
    chuong: str,
    muc: str,
    source_file: str,
) -> Iterator[LegalChunk]:
    """Further split an oversized Điều at Khoản boundaries (internal helper)."""
    re_khoan = re.compile(r"^(\d+)\.\s", re.MULTILINE)
    parts = list(re_khoan.finditer(dieu_text))

    if not parts:
        yield LegalChunk(
            text=dieu_text[:_MAX_CHUNK_CHARS],
            metadata={"type": "dieu", "dieu": dieu_num,
                      "chuong": chuong, "muc": muc,
                      "source_file": source_file},
        )
        return

    for i, part in enumerate(parts):
        start = part.start()
        end = parts[i + 1].start() if i + 1 < len(parts) else len(dieu_text)
        khoan_text = dieu_text[start:end].strip()
        khoan_num = part.group(1)
        yield LegalChunk(
            text=f"Điều {dieu_num}. Khoản {khoan_num}:\n{khoan_text}",
            metadata={"type": "khoan", "dieu": dieu_num, "khoan": khoan_num,
                      "chuong": chuong, "muc": muc,
                      "source_file": source_file},
        )


def _split_by_paragraph(text: str, source_file: str) -> list[LegalChunk]:
    """Fallback paragraph splitter when no Điều structure is found."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[LegalChunk] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < _MAX_CHUNK_CHARS:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                chunks.append(LegalChunk(
                    text=buffer,
                    metadata={"type": "paragraph", "source_file": source_file},
                ))
            buffer = para
    if buffer:
        chunks.append(LegalChunk(
            text=buffer,
            metadata={"type": "paragraph", "source_file": source_file},
        ))
    return [c for c in chunks if c.is_valid]
