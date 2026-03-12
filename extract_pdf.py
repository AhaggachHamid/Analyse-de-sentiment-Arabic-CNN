"""
PDF Hierarchical Section Extractor
===================================
Extracts text and tables from technical PDF documents (Avis Technique style)
and restructures them as hierarchical sections saved to a .txt file.

Usage:
    python3 extract_pdf.py <input.pdf> [output.txt]

If output.txt is omitted, the output file is placed next to the PDF with .txt extension.
Multiple PDFs can be processed at once:
    python3 extract_pdf.py file1.pdf file2.pdf [--outdir /path/to/dir]
"""

import sys
import os
import re
from typing import Optional, List, Dict, Tuple, Set
import fitz          # PyMuPDF  – font/size-aware text extraction
import pdfplumber    # table extraction + bounding boxes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECTION_SEPARATOR = "\n" + "=" * 80 + "\n"

# Heading detection thresholds (font size, bold flag)
HEADING_RULES = [
    # (min_size, max_size, must_be_bold, markdown_prefix)
    (18.0, 99.0, True,  "# "),      # H1  – e.g. "1. Avis du Groupe Spécialisé"
    (10.0, 17.9, True,  "## "),     # H2  – e.g. "1.1. Domaine d'emploi accepté"
    (8.5,   9.9, True,  "### "),    # H3  – e.g. "1.1.1. Zone géographique"
    (8.5,   9.9, False, "#### "),   # H4  – e.g. "1.2.1.1. Réaction au feu"
]

# Regex that matches numbered section identifiers like "1.", "1.1.", "2.3.4.", etc.
SECTION_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?\s")

# Page header/footer patterns to skip (common in technical docs)
SKIP_LINE_RE = re.compile(
    r"^(Avis Technique\s+\d+/\d+-\d+.*|Page\s+\d+\s+sur\s+\d+)$",
    re.IGNORECASE,
)

# Table caption pattern
TABLEAU_CAPTION_RE = re.compile(r"Tableau\s+\d+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_span(size: float, bold: bool) -> Optional[str]:
    """Return the markdown heading prefix for a span, or None if body text."""
    for min_s, max_s, must_bold, prefix in HEADING_RULES:
        if min_s <= size <= max_s:
            if must_bold and not bold:
                continue
            return prefix
    return None


def is_bold(flags: int) -> bool:
    """PyMuPDF stores bold in bit 4 (value 16) of the flags integer."""
    return bool(flags & 16)


def table_to_text(table: list) -> str:
    """Render a pdfplumber table as a plain-text grid with borders."""
    if not table:
        return ""

    # Normalise cells: replace None with ""
    rows = []
    for row in table:
        rows.append([str(cell).strip() if cell is not None else "" for cell in row])

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    col_widths = [0] * col_count
    for row in rows:
        for ci, cell in enumerate(row):
            max_line = max((len(ln) for ln in cell.split("\n")), default=0)
            col_widths[ci] = max(col_widths[ci], max_line)

    def make_border(widths):
        parts = ["-" * (w + 2) for w in widths]
        return "+" + "+".join(parts) + "+"

    def render_row(row, widths):
        cell_lines = [cell.split("\n") for cell in row]
        max_lines = max(len(cl) for cl in cell_lines)
        lines = []
        for li in range(max_lines):
            parts = []
            for ci, cl in enumerate(cell_lines):
                text = cl[li] if li < len(cl) else ""
                parts.append(f" {text:<{widths[ci]}} ")
            lines.append("|" + "|".join(parts) + "|")
        return "\n".join(lines)

    border = make_border(col_widths)
    rendered = [border]
    for row in rows:
        rendered.append(render_row(row, col_widths))
        rendered.append(border)
    return "\n".join(rendered)


def rects_overlap(r1: Tuple, r2: Tuple, tolerance: float = 2.0) -> bool:
    """Check if two (x0, top, x1, bottom) rectangles overlap."""
    x0_1, top_1, x1_1, bot_1 = r1
    x0_2, top_2, x1_2, bot_2 = r2
    return not (
        x1_1 + tolerance < x0_2 or
        x1_2 + tolerance < x0_1 or
        bot_1 + tolerance < top_2 or
        bot_2 + tolerance < top_1
    )


# ---------------------------------------------------------------------------
# Span extraction (PyMuPDF) with table-region exclusion
# ---------------------------------------------------------------------------

def extract_page_spans(page_fitz, excluded_rects: List[Tuple] = None) -> List[dict]:
    """
    Return a flat list of span dicts from a PyMuPDF page.
    Spans whose bounding box overlaps any excluded_rect are skipped.
    Each span dict: {text, size, bold, x0, y0, x1, y1, block_no, line_no}
    """
    if excluded_rects is None:
        excluded_rects = []

    spans = []
    blocks = page_fitz.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for b_idx, block in enumerate(blocks):
        if block["type"] != 0:
            continue
        for l_idx, line in enumerate(block["lines"]):
            merged: List[dict] = []
            for span in line["spans"]:
                txt = span["text"]
                sz = round(span["size"], 1)
                bd = is_bold(span["flags"])
                # Span bounding box in (x0, top, x1, bottom) format
                bbox = span["bbox"]  # (x0, y0, x1, y1)
                span_rect = (bbox[0], bbox[1], bbox[2], bbox[3])

                # Skip spans inside table regions
                in_table = any(rects_overlap(span_rect, er) for er in excluded_rects)
                if in_table:
                    continue

                if merged and merged[-1]["size"] == sz and merged[-1]["bold"] == bd:
                    merged[-1]["text"] += txt
                    merged[-1]["x1"] = bbox[2]
                    merged[-1]["y1"] = bbox[3]
                else:
                    merged.append({
                        "text": txt,
                        "size": sz,
                        "bold": bd,
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "block_no": b_idx,
                        "line_no": l_idx,
                    })
            spans.extend(merged)
    return spans


def group_spans_into_lines(spans: List[dict]) -> List[List[dict]]:
    """Group spans that share the same (block_no, line_no) into lines."""
    from collections import defaultdict
    lines_dict: Dict[tuple, List[dict]] = defaultdict(list)
    for sp in spans:
        key = (sp["block_no"], sp["line_no"])
        lines_dict[key].append(sp)
    sorted_keys = sorted(lines_dict.keys(), key=lambda k: (
        lines_dict[k][0]["y0"], lines_dict[k][0]["x0"]
    ))
    return [lines_dict[k] for k in sorted_keys]


def line_to_heading(line_spans: List[dict]) -> Tuple[Optional[str], str]:
    """
    Analyse a line's spans and decide if it is a heading.
    Returns (prefix_or_None, full_text).
    """
    full_text = "".join(sp["text"] for sp in line_spans).strip()
    if not full_text:
        return None, full_text

    classifications = set()
    for sp in line_spans:
        t = sp["text"].strip()
        if not t:
            continue
        cls = classify_span(sp["size"], sp["bold"])
        classifications.add(cls)

    # All non-empty spans agree on a heading level
    if len(classifications) == 1 and None not in classifications:
        return classifications.pop(), full_text

    # Mixed: dominant span is a heading and text starts with a section number
    dominant = max(line_spans, key=lambda s: len(s["text"].strip()))
    cls = classify_span(dominant["size"], dominant["bold"])
    if cls is not None and SECTION_NUMBER_RE.match(full_text):
        return cls, full_text

    return None, full_text


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_sections(pdf_path: str) -> List[dict]:
    """
    Parse the PDF and return a list of section dicts:
      {
        "heading": str,
        "level": int,
        "content_blocks": [
            {"type": "text", "text": "..."},
            {"type": "table", "caption": "...", "page": int, "index": int, "data": [[...]]},
        ]
      }
    """
    doc_fitz = fitz.open(pdf_path)

    # Pre-extract tables AND their bounding boxes with pdfplumber
    # tables_by_page[page_num] = list of {"bbox": (x0,top,x1,bot), "data": [[...]]}
    tables_by_page: Dict[int, List[dict]] = {}
    with pdfplumber.open(pdf_path) as pdf_pl:
        for page_num, page in enumerate(pdf_pl.pages):
            found = page.find_tables()
            if found:
                tables_by_page[page_num] = []
                for tbl_obj in found:
                    bbox = tbl_obj.bbox  # (x0, top, x1, bottom) in PDF coords
                    data = tbl_obj.extract()
                    tables_by_page[page_num].append({
                        "bbox": bbox,
                        "data": data,
                    })

    sections: List[dict] = []
    current_section: Optional[dict] = None
    current_text_lines: List[str] = []

    def flush_text():
        nonlocal current_text_lines
        if current_section is not None and current_text_lines:
            paragraph = "\n".join(current_text_lines).strip()
            if paragraph:
                current_section["content_blocks"].append(
                    {"type": "text", "text": paragraph}
                )
        current_text_lines = []

    def new_section(prefix: str, heading_text: str):
        nonlocal current_section
        flush_text()
        level = prefix.count("#")
        current_section = {
            "heading": prefix + heading_text,
            "level": level,
            "content_blocks": [],
        }
        sections.append(current_section)

    for page_num in range(len(doc_fitz)):
        page_fitz = doc_fitz[page_num]
        page_tables = tables_by_page.get(page_num, [])

        # Build excluded rects from table bounding boxes
        # pdfplumber uses (x0, top, x1, bottom) with top-left origin
        # PyMuPDF also uses top-left origin for text bbox, so coords are compatible
        excluded_rects = [t["bbox"] for t in page_tables]

        # Extract text spans, skipping table regions
        spans = extract_page_spans(page_fitz, excluded_rects)
        lines = group_spans_into_lines(spans)

        # We'll track which tables have been placed (by index on this page)
        # and their approximate y-position so we can interleave them correctly.
        # We use a pointer: after each caption line, insert the next table.
        table_idx = 0

        # Collect (y_top, table_dict) for ordering tables by vertical position
        ordered_tables = sorted(
            enumerate(page_tables),
            key=lambda x: x[1]["bbox"][1]  # sort by top y
        )
        # Rebuild as a simple list in vertical order
        page_tables_ordered = [t for _, t in ordered_tables]
        table_idx = 0

        for line_spans in lines:
            full_text = "".join(sp["text"] for sp in line_spans).strip()

            if SKIP_LINE_RE.match(full_text):
                continue
            if not full_text:
                continue

            prefix, heading_text = line_to_heading(line_spans)

            if prefix:
                new_section(prefix, heading_text)
            else:
                if current_section is None:
                    current_section = {
                        "heading": "# Document",
                        "level": 1,
                        "content_blocks": [],
                    }
                    sections.append(current_section)

                # If this line is a table caption, flush text and attach next table
                if TABLEAU_CAPTION_RE.search(full_text) and table_idx < len(page_tables_ordered):
                    flush_text()
                    current_text_lines.append(full_text)
                    flush_text()
                    tbl = page_tables_ordered[table_idx]
                    current_section["content_blocks"].append({
                        "type": "table",
                        "caption": full_text,
                        "page": page_num + 1,
                        "index": table_idx + 1,
                        "data": tbl["data"],
                    })
                    table_idx += 1
                else:
                    current_text_lines.append(full_text)

        # Flush remaining text for this page
        flush_text()

        # Attach any remaining tables that had no caption
        for ti in range(table_idx, len(page_tables_ordered)):
            if current_section is None:
                current_section = {
                    "heading": "# Document",
                    "level": 1,
                    "content_blocks": [],
                }
                sections.append(current_section)
            tbl = page_tables_ordered[ti]
            current_section["content_blocks"].append({
                "type": "table",
                "caption": "",
                "page": page_num + 1,
                "index": ti + 1,
                "data": tbl["data"],
            })

    doc_fitz.close()
    return sections


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_sections(sections: List[dict]) -> str:
    """Convert the section list to a formatted text string."""
    output_parts: List[str] = []

    for sec in sections:
        output_parts.append(SECTION_SEPARATOR)
        output_parts.append(sec["heading"])
        output_parts.append("")

        for block in sec["content_blocks"]:
            if block["type"] == "text":
                output_parts.append(block["text"])
                output_parts.append("")
            elif block["type"] == "table":
                # Caption was already written as the preceding text block
                page = block["page"]
                idx = block["index"]
                data = block["data"]

                output_parts.append(f"[Table {idx} — Page {page}]")
                output_parts.append(table_to_text(data))
                output_parts.append("")

    return "\n".join(output_parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract and structure a PDF into hierarchical sections.
    Returns the output file path.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        base = os.path.splitext(pdf_path)[0]
        output_path = base + "_extracted.txt"

    print(f"[*] Parsing: {pdf_path}")
    sections = build_sections(pdf_path)
    print(f"[*] Found {len(sections)} sections")

    text = render_sections(sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[*] Output written to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Support multiple input files with optional --outdir
    args = sys.argv[1:]
    outdir = None
    if "--outdir" in args:
        idx = args.index("--outdir")
        outdir = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
        os.makedirs(outdir, exist_ok=True)

    pdf_files = [a for a in args if a.endswith(".pdf")]
    out_files = [a for a in args if not a.endswith(".pdf")]

    if not pdf_files:
        print(__doc__)
        sys.exit(1)

    for i, pdf_file in enumerate(pdf_files):
        if outdir:
            base = os.path.splitext(os.path.basename(pdf_file))[0]
            out_file = os.path.join(outdir, base + "_extracted.txt")
        elif i < len(out_files):
            out_file = out_files[i]
        else:
            out_file = None
        extract_pdf(pdf_file, out_file)
