#!/usr/bin/env python3
"""
review_footnotes.py — Review footnote detection results from 13_text_classifier.py

Reads the classification output (raw JSON or manifest) and displays footnotes
alongside the source text so you can evaluate detection quality.

Usage:
    # Review a single classified document
    python review_footnotes.py ../PDFs/SomeDocument_combined_classification

    # Review from a raw JSON file directly
    python review_footnotes.py ../PDFs/SomeDoc_classification/SomeDoc_text_classification_raw.json

    # Review from a manifest.json
    python review_footnotes.py ../PDFs/SomeDoc_classification/manifest.json

    # Show full footnote text (default truncates to 120 chars)
    python review_footnotes.py ../PDFs/SomeDoc_classification --full

    # Also show the source page text around footnotes
    python review_footnotes.py ../PDFs/SomeDoc_classification --context
"""

import json
import sys
import re
import argparse
from pathlib import Path


def load_footnote_data(target: Path) -> tuple:
    """
    Load footnote data from a classification output directory, raw JSON, or manifest.
    Returns (doc_name, pages_with_fn_data, source_md_path_or_none).
    """
    if target.is_file():
        if target.name == "manifest.json":
            manifest = json.loads(target.read_text(encoding="utf-8"))
            doc_name = manifest.get("source_md", manifest.get("document_id", target.parent.name))
            pages = manifest.get("pages", [])
            # Manifest pages have 'footnotes' directly
            fn_pages = []
            for p in pages:
                fn_pages.append({
                    "page_number": p["page_number"],
                    "has_footnote": p.get("has_footnote", False),
                    "footnotes": p.get("footnotes", []),
                })
            return doc_name, fn_pages, target.parent

        elif target.suffix == ".json":
            raw = json.loads(target.read_text(encoding="utf-8"))
            doc_name = target.stem.replace("_text_classification_raw", "")
            pages = raw.get("pages", [])
            fn_pages = []
            for p in pages:
                fn_pages.append({
                    "page_number": p["page_number"],
                    "has_footnote": p.get("has_footnote", False),
                    "footnotes": p.get("footnotes", []),
                })
            return doc_name, fn_pages, target.parent

    elif target.is_dir():
        # Look for manifest first, then raw JSON
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            return load_footnote_data(manifest_path)

        raw_jsons = list(target.glob("*_text_classification_raw.json"))
        if raw_jsons:
            return load_footnote_data(raw_jsons[0])

    print(f"ERROR: Could not find classification data in {target}", file=sys.stderr)
    sys.exit(1)


def find_source_md(output_dir: Path) -> Path | None:
    """Try to find the source .md file from a classification output directory."""
    # Convention: {doc_name}_classification/ sits next to {doc_name}.md
    dir_name = output_dir.name
    if dir_name.endswith("_classification"):
        md_name = dir_name[: -len("_classification")] + ".md"
        md_path = output_dir.parent / md_name
        if md_path.exists():
            return md_path
    return None


def parse_source_pages(md_path: Path) -> dict:
    """Parse the combined .md and return {page_number: text}."""
    content = md_path.read_text(encoding="utf-8", errors="replace")
    page_start_re = re.compile(r"^---\[Start PDF page (\d+)\]---\s*$")
    page_end_re = re.compile(r"^---\[End PDF page (\d+)\]---\s*$")

    pages = {}
    current_page = None
    current_lines = []

    for line in content.splitlines():
        sm = page_start_re.match(line)
        em = page_end_re.match(line)
        if sm:
            current_page = int(sm.group(1))
            current_lines = []
        elif em:
            if current_page is not None:
                pages[current_page] = "\n".join(current_lines).strip()
            current_page = None
            current_lines = []
        elif current_page is not None:
            current_lines.append(line)

    return pages


def find_footnote_context(page_text: str, fn_number: int, context_chars: int = 200) -> str:
    """
    Try to find where a footnote superscript reference appears in the body text.
    Returns surrounding text or empty string.
    """
    # Look for superscript-like patterns: "text.3 " or "text³" or "text 3"
    patterns = [
        # Footnote number at end of sentence/word
        rf'(?:[\.\,\;\:\"\'\)\])])\s*{fn_number}(?:\s|$|\.|,)',
        # Standalone footnote number reference
        rf'\b{fn_number}\b',
    ]

    for pat in patterns:
        m = re.search(pat, page_text)
        if m:
            start = max(0, m.start() - context_chars // 2)
            end = min(len(page_text), m.end() + context_chars // 2)
            snippet = page_text[start:end].replace("\n", " ")
            return f"...{snippet}..."

    return ""


def print_separator(char="-", width=80):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(
        description="Review footnote detection results from text classifier"
    )
    parser.add_argument(
        "path", type=Path,
        help="Path to classification output dir, raw JSON, or manifest.json"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show full footnote text (default truncates to 120 chars)"
    )
    parser.add_argument(
        "--context", action="store_true",
        help="Show source page text around footnote references"
    )
    args = parser.parse_args()

    target = args.path.resolve()
    doc_name, fn_pages, output_dir = load_footnote_data(target)

    # Try to load source text for --context mode
    source_pages = {}
    if args.context:
        md_path = find_source_md(output_dir)
        if md_path:
            source_pages = parse_source_pages(md_path)
            print(f"Source text loaded from: {md_path.name}")
        else:
            print("WARNING: Could not find source .md file for context display")

    # Collect all footnotes
    all_footnotes = []
    pages_with_fn = 0
    pages_without_fn = 0

    for p in fn_pages:
        if p["has_footnote"]:
            pages_with_fn += 1
            for fn in p.get("footnotes", []):
                all_footnotes.append({
                    **fn,
                    "page": p["page_number"],
                })
        else:
            pages_without_fn += 1

    # Sort by footnote number
    all_footnotes.sort(key=lambda x: (x.get("fn_number", 0), x.get("page", 0)))

    # Print header
    print()
    print_separator("=")
    print(f"FOOTNOTE REVIEW: {doc_name}")
    print_separator("=")
    print(f"Total pages: {len(fn_pages)}")
    print(f"Pages with footnotes: {pages_with_fn}")
    print(f"Total footnotes found: {len(all_footnotes)}")
    print()

    if not all_footnotes:
        print("No footnotes detected in this document.")
        print_separator("=")
        return

    # Print sequential index
    print_separator()
    print(f"{'FN':<5} {'Page':<6} {'Status':<12} {'Text'}")
    print_separator()

    for fn in all_footnotes:
        fn_num = fn.get("fn_number", "?")
        page = fn.get("page", "?")
        status = fn.get("merge_status", "?")
        text = fn.get("fn_text", "")

        if not args.full and len(text) > 120:
            text = text[:117] + "..."

        # Highlight issues
        status_display = status
        if status == "missing":
            status_display = "** MISSING **"
        elif status == "partial":
            status_display = "~ partial ~"

        print(f"{fn_num:<5} {page:<6} {status_display:<12} {text}")

    print_separator()

    # Validation
    print()
    print("VALIDATION:")
    print_separator("-", 40)

    issues = []
    seen = set()
    expected = 1

    for fn in all_footnotes:
        num = fn.get("fn_number", 0)

        # Gaps
        if num > expected:
            for missing in range(expected, num):
                issues.append(f"  GAP: Footnote {missing} is missing (jump from {expected-1} to {num})")

        # Duplicates
        if num in seen:
            issues.append(f"  DUPLICATE: Footnote {num} appears multiple times")

        # Missing text
        if fn.get("merge_status") == "missing":
            issues.append(f"  MISSING TEXT: Footnote {num} on page {fn['page']} — superscript found but no footnote text")

        seen.add(num)
        expected = num + 1

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  All footnotes sequential, no gaps, no duplicates.")

    print()

    # Context display
    if args.context and source_pages:
        print_separator("=")
        print("FOOTNOTE CONTEXT (source text around references)")
        print_separator("=")

        for fn in all_footnotes:
            fn_num = fn.get("fn_number", 0)
            page = fn.get("page", 0)
            page_text = source_pages.get(page, "")

            print(f"\n--- Footnote {fn_num} (page {page}) ---")
            fn_text = fn.get("fn_text", "")
            if not args.full and len(fn_text) > 200:
                fn_text = fn_text[:197] + "..."
            print(f"  Text: {fn_text}")

            merge_loc = fn.get("merge_location", "")
            if merge_loc:
                print(f"  Location: {merge_loc}")

            if page_text:
                # Show where in the page the footnote text actually appears
                if fn.get("fn_text", "") and fn["fn_text"][:40] in page_text:
                    idx = page_text.index(fn["fn_text"][:40])
                    start = max(0, idx - 100)
                    end = min(len(page_text), idx + len(fn["fn_text"]) + 50)
                    context_snippet = page_text[start:end].replace("\n", " | ")
                    print(f"  Source match: ...{context_snippet}...")
                else:
                    # Try to find superscript reference
                    ctx = find_footnote_context(page_text, fn_num)
                    if ctx:
                        print(f"  Body ref: {ctx}")

        print()

    print_separator("=")
    print(f"Summary: {len(all_footnotes)} footnotes across {pages_with_fn} pages, {len(issues)} issue(s)")
    print_separator("=")


if __name__ == "__main__":
    main()
