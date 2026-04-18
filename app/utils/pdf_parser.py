"""Structure-preserving PDF extraction using pymupdf4llm."""

from __future__ import annotations

import io

from app.utils.logger import logger


class PDFParser:
    """Extract text from PDFs with structure preservation.

    Primary: pymupdf4llm — produces Markdown with headings, preserving
    document hierarchy for downstream section parsing.
    Fallback: pypdf — plain text extraction.
    """

    @staticmethod
    def extract(file_data: io.BytesIO) -> str:
        """Extract text from a PDF, returning Markdown-formatted content.

        pymupdf4llm outputs Markdown with `#` headings that
        DocumentExtractor.parse_text_structure() already handles,
        giving us section hierarchy for free.
        """
        file_data.seek(0)

        try:
            return PDFParser._extract_pymupdf4llm(file_data)
        except Exception as exc:
            logger.warning("pymupdf4llm extraction failed: %s — falling back to pypdf.", exc)

        file_data.seek(0)
        return PDFParser._extract_pypdf(file_data)

    @staticmethod
    def _extract_pymupdf4llm(file_data: io.BytesIO) -> str:
        import pymupdf4llm

        md_text: str = pymupdf4llm.to_markdown(file_data)
        cleaned = PDFParser._clean_markdown(md_text)
        if not cleaned.strip():
            raise ValueError("pymupdf4llm returned empty content.")
        return cleaned

    @staticmethod
    def _extract_pypdf(file_data: io.BytesIO) -> str:
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError("pypdf not installed. Run: pip install pypdf") from exc

        file_data.seek(0)
        reader = pypdf.PdfReader(file_data)

        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        content = "\n\n".join(pages)
        if not content.strip():
            raise ValueError(
                "No readable text found in PDF. The file may be image-only (scanned). "
                "Use OCR or upload a text-selectable PDF."
            )
        return content

    @staticmethod
    def _clean_markdown(md: str) -> str:
        """Post-process pymupdf4llm Markdown for cleaner section parsing."""
        import re

        # Remove image references — they don't help text evaluation
        md = re.sub(r"!\[.*?\]\(.*?\)", "", md)

        # Collapse excessive blank lines
        md = re.sub(r"\n{4,}", "\n\n\n", md)

        # Remove page-break markers pymupdf4llm sometimes inserts
        md = re.sub(r"---+\s*\n", "\n", md)

        # Fix surrogate characters from bad encoding
        md = md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")

        return md.strip()
