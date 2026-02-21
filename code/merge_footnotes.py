#!/usr/bin/env python3
"""
merge_footnotes.py - Merge footnotes inline into document body text.

Authority split (anti-hallucination design):
  - LLM classification is authoritative for footnote LOCATION: which
    footnotes exist, which page they belong to, merge_status, merge_location.
  - PaddleOCR source .md is authoritative for footnote TEXT: the actual
    content of each footnote is extracted from the $ ^{N} $ text blocks
    in the source Markdown, never from the LLM's fn_text field.

Two-phase approach:
  Phase 1: Load footnote inventory from classification output (manifest.json
           or raw JSON). This gives us fn_number, page, merge_status.
  Phase 2: For each footnote, extract its verbatim text from the source .md
           OCR blocks, find its body reference marker, and replace the marker
           with [FN{N}: ocr_text]. Remove the footnote text block.

Usage:
    python merge_footnotes.py <input_combined.md> [output_path]

    # If classification data exists, uses LLM for inventory + OCR for text.
    # If not, falls back to pure OCR marker detection.

    # Dry run - show what would be merged without writing
    python merge_footnotes.py <input_combined.md> --dry-run

    # Force use of OCR markers only (skip classification data)
    python merge_footnotes.py <input_combined.md> --ocr-only
"""

import re
import sys
import json
import argparse
from pathlib import Path


# ── Patterns ──────────────────────────────────────────────────────────
# OCR body reference: $ ^{N} $ (may have varying whitespace)
FN_MARKER_RE = re.compile(r"\s*\$\s*\^\{(\d+)\}\s*\$")

# Footnote text block at bottom of page: line starting with $ ^{N} $ then text
FN_TEXT_BLOCK_RE = re.compile(r"^\s*\$\s*\^\{(\d+)\}\s*\$\s+(.+)$")

# Page delimiters
PAGE_START_RE = re.compile(r"^---\[Start PDF page (\d+)\]---\s*$")
PAGE_END_RE = re.compile(r"^---\[End PDF page (\d+)\]---\s*$")


# ── Load classification footnote data ─────────────────────────────────

def load_classification_footnotes(md_path: Path) -> dict | None:
    """
    Try to load footnote INVENTORY from the classification output.
    Returns {fn_number: {page, merge_status, merge_location}} or None.

    NOTE: We intentionally ignore the LLM's fn_text field. The LLM is
    authoritative only for location data. Footnote text content comes
    from the OCR source .md file.
    """
    doc_name = md_path.stem
    output_dir = md_path.parent / f"{doc_name}_classification"

    # Try manifest first
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _extract_fn_inventory(manifest.get("pages", []))

    # Try raw JSON
    raw_jsons = list(output_dir.glob("*_text_classification_raw.json"))
    if raw_jsons:
        raw = json.loads(raw_jsons[0].read_text(encoding="utf-8"))
        return _extract_fn_inventory(raw.get("pages", []))

    return None


def _extract_fn_inventory(pages: list) -> dict:
    """
    Extract footnote inventory (location data only) from classification results.
    The LLM's fn_text is stored as _llm_text for diagnostics but NOT used
    for the merge — OCR text takes priority.
    """
    footnotes = {}
    for page in pages:
        pg_num = page.get("page_number", 0)
        for fn in page.get("footnotes", []):
            fn_num = fn.get("fn_number", 0)
            if fn_num > 0:
                footnotes[fn_num] = {
                    "page": pg_num,
                    "merge_status": fn.get("merge_status", ""),
                    "merge_location": fn.get("merge_location", ""),
                    "_llm_text": fn.get("fn_text", ""),  # diagnostic only
                }
    return footnotes


# ── OCR-based footnote extraction ─────────────────────────────────────

def extract_ocr_footnotes(lines: list) -> dict:
    """
    Extract footnote TEXT from OCR markers on a single page.
    Returns {fn_number: fn_text} for footnote text blocks found.
    This is the authoritative source for footnote content.
    """
    footnotes = {}
    for line in lines:
        m = FN_TEXT_BLOCK_RE.match(line)
        if m:
            fn_num = int(m.group(1))
            fn_text = m.group(2).strip()
            footnotes[fn_num] = fn_text
    return footnotes


# ── Page processing ───────────────────────────────────────────────────

def split_into_pages(content: str) -> tuple:
    """
    Split .md content into page blocks.
    Returns (preamble_lines, [page_blocks]).
    Each page_block: {page_number, start_delim, lines, end_delim}
    """
    all_lines = content.split("\n")
    blocks = []
    preamble = []
    current_page = None
    current_lines = []
    current_start = ""

    for line in all_lines:
        sm = PAGE_START_RE.match(line)
        em = PAGE_END_RE.match(line)

        if sm:
            current_page = int(sm.group(1))
            current_start = line
            current_lines = []
        elif em:
            if current_page is not None:
                blocks.append({
                    "page_number": current_page,
                    "start_delim": current_start,
                    "lines": current_lines,
                    "end_delim": line,
                })
            current_page = None
            current_lines = []
        elif current_page is not None:
            current_lines.append(line)
        else:
            preamble.append(line)

    return preamble, blocks


def remove_fn_text_blocks(lines: list, fn_numbers_to_remove: set) -> list:
    """
    Remove footnote text block lines and their surrounding blank lines.
    Only removes blocks whose fn_number is in fn_numbers_to_remove.
    """
    fn_line_indices = set()
    for i, line in enumerate(lines):
        m = FN_TEXT_BLOCK_RE.match(line)
        if m and int(m.group(1)) in fn_numbers_to_remove:
            fn_line_indices.add(i)

    if not fn_line_indices:
        return lines

    cleaned = []
    for i, line in enumerate(lines):
        if i in fn_line_indices:
            continue
        # Skip blank lines adjacent to removed footnote blocks
        if line.strip() == "":
            next_content = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    next_content = j
                    break
            if next_content in fn_line_indices:
                continue
            prev_content = None
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    prev_content = j
                    break
            if prev_content in fn_line_indices:
                continue
        cleaned.append(line)

    return cleaned


def merge_refs_inline(lines: list, fn_data: dict) -> list:
    """
    Replace body reference markers $ ^{N} $ with [FN{N}: text] inline.
    fn_data: {fn_number: {fn_text, ...}} or {fn_number: fn_text_string}
    """
    def replace_ref(m):
        fn_num = int(m.group(1))
        entry = fn_data.get(fn_num)
        if entry is None:
            return m.group(0)  # keep marker if no data
        fn_text = entry["fn_text"] if isinstance(entry, dict) else entry
        if fn_text:
            return f" [FN{fn_num}: {fn_text}]"
        return m.group(0)

    merged = []
    for line in lines:
        merged.append(FN_MARKER_RE.sub(replace_ref, line))
    return merged


# ── Main processing ───────────────────────────────────────────────────

def process_document(content: str, classification_fns: dict | None) -> tuple:
    """
    Process the full document.

    Authority split:
    - classification_fns (if available): LLM provides the footnote inventory
      (which footnotes exist, corrected page numbers). Used to determine
      WHICH footnotes to process.
    - OCR source text: always provides the footnote TEXT content. The text
      from $ ^{N} $ blocks in the source .md is what gets merged inline.

    Returns (merged_content, stats_dict).
    """
    preamble, blocks = split_into_pages(content)

    source = "classification+ocr" if classification_fns else "ocr_only"
    total_found = 0
    total_merged = 0
    total_unmatched = 0
    total_no_ocr_text = 0
    page_stats = []

    output_parts = []
    if preamble:
        output_parts.append("\n".join(preamble))

    # Pre-scan: find which pages have OCR markers for each fn_number
    # (both body references and text blocks)
    marker_pages = {}  # fn_number -> set of page_numbers with markers
    for block in blocks:
        for line in block["lines"]:
            for m in FN_MARKER_RE.finditer(line):
                fn_num = int(m.group(1))
                marker_pages.setdefault(fn_num, set()).add(block["page_number"])

    # Pre-scan: extract ALL OCR footnote text blocks across the document
    # This is the authoritative source for footnote content
    ocr_texts = {}  # fn_number -> fn_text (from OCR)
    for block in blocks:
        page_ocr = extract_ocr_footnotes(block["lines"])
        for fn_num, fn_text in page_ocr.items():
            ocr_texts[fn_num] = fn_text

    # If using classification data, correct LLM page attribution errors
    if classification_fns:
        for fn_num, data in classification_fns.items():
            if fn_num in marker_pages:
                actual_pages = marker_pages[fn_num]
                llm_page = data["page"]
                if llm_page not in actual_pages:
                    corrected = min(actual_pages)
                    data["_original_page"] = llm_page
                    data["page"] = corrected

    for block in blocks:
        pg_num = block["page_number"]
        lines = block["lines"]

        # Determine which footnotes are on this page
        if classification_fns:
            # Use LLM inventory for which footnotes exist on this page,
            # but get the actual text from OCR
            page_fns = {}
            for num, data in classification_fns.items():
                if data["page"] == pg_num:
                    ocr_text = ocr_texts.get(num, "")
                    if not ocr_text:
                        total_no_ocr_text += 1
                    page_fns[num] = {
                        "fn_text": ocr_text,  # OCR text (authoritative)
                        "page": pg_num,
                        "_llm_text": data.get("_llm_text", ""),
                        "_text_source": "ocr" if ocr_text else "none",
                    }
        else:
            # Pure OCR mode — extract both inventory and text from markers
            page_ocr = extract_ocr_footnotes(lines)
            page_fns = {
                num: {"fn_text": text, "page": pg_num, "_text_source": "ocr"}
                for num, text in page_ocr.items()
            }

        if page_fns:
            total_found += len(page_fns)

            # Count body references on this page that match our footnotes
            refs_on_page = set()
            for line in lines:
                for m in FN_MARKER_RE.finditer(line):
                    fn_num = int(m.group(1))
                    if fn_num in page_fns:
                        refs_on_page.add(fn_num)

            # Remove footnote text blocks from bottom of page
            cleaned = remove_fn_text_blocks(lines, set(page_fns.keys()))

            # Merge references inline using OCR text
            merged = merge_refs_inline(cleaned, page_fns)

            total_merged += len(refs_on_page)

            # Check for footnotes without a body reference on this page
            unmatched_fns = set(page_fns.keys()) - refs_on_page
            if unmatched_fns:
                total_unmatched += len(unmatched_fns)

            # Track which had no OCR text
            no_ocr = [n for n in page_fns if not page_fns[n]["fn_text"]]

            page_stats.append({
                "page": pg_num,
                "fn_numbers": sorted(page_fns.keys()),
                "merged_refs": sorted(refs_on_page),
                "unmatched": sorted(unmatched_fns),
                "no_ocr_text": sorted(no_ocr),
            })
        else:
            merged = lines

        # Reassemble page
        output_parts.append(block["start_delim"])
        output_parts.append("\n".join(merged))
        output_parts.append(block["end_delim"])

    merged_content = "\n".join(output_parts)

    stats = {
        "source": source,
        "total_found": total_found,
        "total_merged": total_merged,
        "total_unmatched": total_unmatched,
        "total_no_ocr_text": total_no_ocr_text,
        "pages_with_footnotes": len(page_stats),
        "page_details": page_stats,
    }

    return merged_content, stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge footnotes inline into document body text"
    )
    parser.add_argument(
        "input", type=Path,
        help="Path to combined .md file"
    )
    parser.add_argument(
        "output", type=Path, nargs="?", default=None,
        help="Output path (default: {input_stem}_fn_merged.md)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stats without writing output"
    )
    parser.add_argument(
        "--ocr-only", action="store_true",
        help="Use OCR markers only, skip classification data"
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read input
    content = input_path.read_text(encoding="utf-8", errors="replace")
    print(f"Input: {input_path.name}")

    # Load classification footnote data (inventory only — text comes from OCR)
    classification_fns = None
    if not args.ocr_only:
        classification_fns = load_classification_footnotes(input_path)
        if classification_fns:
            print(f"Footnote inventory: LLM classification ({len(classification_fns)} footnotes)")
            print(f"Footnote text: OCR source .md (authoritative)")
            for num in sorted(classification_fns.keys()):
                fn = classification_fns[num]
                loc = fn.get("merge_location", "")
                loc_display = f' | loc: "{loc}"' if loc else ""
                print(f"  FN{num} (pg {fn['page']}, {fn['merge_status']}){loc_display}")
        else:
            print("Footnote source: OCR markers only (no classification data found)")
    else:
        print("Footnote source: OCR markers only (--ocr-only)")

    # Process
    merged_content, stats = process_document(content, classification_fns)

    # Print results
    print(f"\nMerge results:")
    print(f"  Source: {stats['source']}")
    print(f"  Pages with footnotes: {stats['pages_with_footnotes']}")
    print(f"  Footnotes found: {stats['total_found']}")
    print(f"  Merged inline: {stats['total_merged']}")
    if stats["total_no_ocr_text"] > 0:
        print(f"  WARNING - Footnotes with no OCR text block: {stats['total_no_ocr_text']}")
    if stats["total_unmatched"] > 0:
        print(f"  WARNING - Footnotes without body reference marker: {stats['total_unmatched']}")

    for ps in stats["page_details"]:
        fn_list = ", ".join(str(n) for n in ps["fn_numbers"])
        line = f"    Page {ps['page']:>3}: FN [{fn_list}]"
        if ps["no_ocr_text"]:
            no_ocr_list = ", ".join(str(n) for n in ps["no_ocr_text"])
            line += f"  (no OCR text: [{no_ocr_list}])"
        if ps["unmatched"]:
            um_list = ", ".join(str(n) for n in ps["unmatched"])
            line += f"  (no body ref: [{um_list}])"
        print(line)

    if args.dry_run:
        print("\n(Dry run - no file written)")
        return

    # Write output
    if args.output:
        output_path = args.output.resolve()
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_fn_merged.md"

    output_path.write_text(merged_content, encoding="utf-8")
    print(f"\nOutput: {output_path.name}")
    print(f"  {len(content):,} chars -> {len(merged_content):,} chars")


if __name__ == "__main__":
    main()
