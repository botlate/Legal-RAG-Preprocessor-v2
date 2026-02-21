#!/usr/bin/env python3
"""
13_text_classifier.py — Text-based whole-document classifier (GPT-5.2)

Sends the ENTIRE document text in a single API call and receives
page-by-page classification + caption metadata as structured JSON.

Enhanced with:
  - Modular prompt design (8 assembled sections)
  - First-page image review (caption cross-check)
  - Two-pass image review architecture
  - Hallucination controls (verify extracted text against source)
  - Footnote tracking + sequential index
  - Cause-of-action extraction for complaints
  - Manifest output (central data structure)
  - Token budget awareness

Usage:
    # Classify a single document folder
    python 13_text_classifier.py ../doc_files/SomeDocument

    # Classify a combined .md file
    python 13_text_classifier.py ../PDFs/SomeDocument_combined.md

    # Classify all documents in doc_files
    python 13_text_classifier.py ../doc_files

    # Compare results against existing image-based classification
    python 13_text_classifier.py ../doc_files/SomeDocument --compare

    # Force re-classification even if results exist
    python 13_text_classifier.py ../doc_files --force
"""

import os
import sys
import json
import csv
import logging
import time
import re
import base64
import io
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
MODEL = "gpt-5.2"
OUTPUT_CSV_SUFFIX = "_classification_text.csv"
OUTPUT_CAPTION_SUFFIX = "_caption_text.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("text_classifier")

# ── Categories (must match 11_document_classifier.py) ───────────────────
CATEGORIES = [
    "Form",
    "Pleading first page",
    "Pleading table of contents",
    "Pleading table of authorities",
    "Exhibit cover page",
    "Exhibit content",
    "Proof of service",
    "Pleading body",
]

DOCUMENT_TYPES = [
    "complaint", "motion", "opposition", "reply", "brief",
    "declaration", "notice", "order", "other",
]

IMAGE_REVIEW_REASONS = [
    "garbled_ocr", "handwritten", "visual_element",
    "stamped_date", "footnote_unclear", "other",
]

FOOTNOTE_MERGE_STATUSES = ["merged", "missing", "partial", "not_applicable"]

# ── Token budget ────────────────────────────────────────────────────────
MODEL_CONTEXT_WINDOWS = {
    "gpt-5.2": 1_048_576,
    "gpt-4.1": 1_048_576,
    "gpt-4.1-mini": 1_048_576,
    "gpt-4.1-nano": 1_048_576,
    "o3": 200_000,
    "o4-mini": 200_000,
}

# ── Search function (embedded in prompt) ────────────────────────────────

def search_source_text(
    search_text: str, page_number: int, pages: List[Dict]
) -> Optional[str]:
    """
    Search the POVL Markdown source for an exact match (after whitespace
    normalization), scoped to the specified page.

    Returns the verbatim source text at the match location, or None if
    no match is found. No fuzzy matching — exact only.
    """
    if not search_text or not search_text.strip():
        return None

    # Find the target page
    target_page = None
    for pg in pages:
        if pg["number"] == page_number:
            target_page = pg
            break
    if target_page is None:
        return None

    # Normalize: collapse all whitespace runs to single space
    def normalize(s):
        return re.sub(r"\s+", " ", s).strip()

    needle = normalize(search_text)
    if not needle:
        return None

    source = target_page["text"]
    source_norm = normalize(source)

    idx = source_norm.find(needle)
    if idx == -1:
        return None

    # Map normalized index back to original source to extract verbatim text.
    # Walk both strings in parallel.
    src_i = 0  # position in original source
    norm_i = 0  # position in normalized source

    # Skip leading whitespace in original
    while src_i < len(source) and source[src_i] in " \t\n\r":
        src_i += 1

    # Advance to the match start
    while norm_i < idx:
        if source[src_i] in " \t\n\r":
            # Skip whitespace run in original
            while src_i < len(source) and source[src_i] in " \t\n\r":
                src_i += 1
            norm_i += 1  # one space in normalized
        else:
            src_i += 1
            norm_i += 1

    match_start = src_i

    # Advance through the match length
    match_len = len(needle)
    consumed = 0
    while consumed < match_len and src_i < len(source):
        if source[src_i] in " \t\n\r":
            while src_i < len(source) and source[src_i] in " \t\n\r":
                src_i += 1
            consumed += 1
        else:
            src_i += 1
            consumed += 1

    return source[match_start:src_i]


# Clean version of the search function for embedding in the LLM prompt.
# The LLM sees this so it understands the contract.
SEARCH_FUNCTION_FOR_PROMPT = '''\
def search_source_text(search_text, page_number, source_pages):
    """Exact match after whitespace normalization, scoped to one page."""
    import re
    def normalize(s):
        return re.sub(r"\\s+", " ", s).strip()
    needle = normalize(search_text)
    page_text = source_pages[page_number]
    if normalize(page_text).find(needle) != -1:
        return verbatim_text_at_match   # exact source chars
    return None  # field flagged as unverified
'''


# ── Modular Prompt Sections ─────────────────────────────────────────────

PROMPT_BASE = """\
You are an expert legal document analyst specializing in California state \
court litigation filings. You will receive the full OCR text of a court \
filing, with each page separated by "=== PAGE N ===" markers.

YOUR TASKS:
1. Classify every page into exactly one category.
2. Extract caption information from the first page of the pleading.
3. For exhibits, identify labels and titles.
4. Identify the overall document type (complaint, motion, opposition, reply, \
brief, declaration, notice, order, or other).
5. Track footnotes on each page.
6. For complaints: extract causes of action with paragraph ranges.
7. Request image review for pages where OCR text is insufficient.

CRITICAL PRINCIPLE: The OCR text you receive is ground truth. You classify \
and extract metadata from it. You NEVER modify, rewrite, or paraphrase the \
source text.

SOURCE-TEXT VERIFICATION:
Certain fields in your output are SEARCH-CONSTRAINED. For these fields, \
your answer is not the value itself — it is an input to a search function \
that will attempt to find your string as an exact match (after whitespace \
normalization) in the source text on the page you specify. The search \
function is:

""" + SEARCH_FUNCTION_FOR_PROMPT + """

If the search function finds a match, the verbatim text from the source \
document will be written to the manifest. If it does not find a match, \
the field will be left empty and flagged for manual review. Your string \
must appear exactly as written in the source document. Do not paraphrase, \
summarize, correct spelling, fix formatting, or editorialize. Any \
deviation from the source text will cause the search to fail.

FIDELITY REQUIREMENT:
When extracting text from the document, reproduce it exactly as written. \
Do not paraphrase, summarize, correct spelling, fix formatting, or \
editorialize. Your job is to report what the document says, not to \
improve it.

For the following fields, your answers will be passed directly into the \
Python search function shown above, which matches against the source \
text with whitespace normalization only. Any deviation from the text as \
it actually appears in the document — ignoring whitespace — means the \
answer cannot be used:
  - document_title (in caption_info)
  - cause of action titles (search_text in causes_of_action)
  - section headings used in section_path

All other fields are answered directly in your JSON response:
  - court, case_number, filing_attorneys, judge, department
  - filing_party, named_plaintiffs, named_defendants
  - exhibit_label (the letter or number designation — infer sequentially \
    if no explicit cover page is present)
  - exhibit_title (your editorial description of what the exhibit contains)
  - filing_date (from image review if available)
  - Page-level classification (from fixed category list)
  - notes, exhibit_notes (free text)\
"""

PROMPT_CATEGORIES = """\
CATEGORIES (choose exactly one per page):
- "Form" — A pre-printed judicial council, FRCP, or local court form. \
  These have form numbers (SUM-100, POS-010, MC-025, FL-100, AO 88, etc.) \
  and pre-printed field labels/checkboxes. If the document is on pleading \
  paper (numbered lines), it is NOT a form.
- "Pleading first page" — The first page showing the case caption: \
  attorney block at top, court name, party names (vs.), case number, and \
  document title. Usually only ONE page in the document gets this label.
- "Pleading table of contents" — Lists section headings with page numbers.
- "Pleading table of authorities" — Lists legal citations (cases, statutes, \
  treatises) with page numbers.
- "Exhibit cover page" — A separator page for an exhibit. Text is typically \
  very short, often just "EXHIBIT A" or "EXHIBIT 1" with little else.
- "Exhibit content" — Pages that are part of an exhibit (everything between \
  one exhibit cover and the next, or between the last exhibit cover and the \
  end of the document).
- "Proof of service" — Declares service of the document on other parties. \
  Contains language like "I served," "proof of service," "declare under \
  penalty of perjury," and lists of persons/addresses served.
- "Pleading body" — Main argument/content pages of the pleading. This is \
  the default for pages that don't fit other categories.

KEY CLASSIFICATION GUIDANCE:
- A document typically starts with ONE "Pleading first page" followed by \
  body pages, then possibly exhibits at the end.
- Exhibit sections follow a pattern: cover page -> content pages -> next \
  cover page -> content pages, etc.
- Once you see an exhibit cover page, ALL subsequent pages are exhibit \
  content UNTIL you see another exhibit cover page or the document ends.
- Table of contents and table of authorities usually appear near the \
  beginning, after the first page and before the body.
- If a page's text is extremely short (just an exhibit label), it is \
  likely an exhibit cover page.

CRITICAL RULES FOR EXHIBITS:
- Once a page is classified as "Exhibit cover page" or "Exhibit content," \
  ALL subsequent pages are "Exhibit content" until you see another \
  explicit "Exhibit cover page." No exceptions.
- Do NOT sub-classify pages within exhibits. Everything inside an exhibit \
  is "Exhibit content" regardless of what it contains.
- A new exhibit ONLY begins at an explicit "Exhibit cover page."
- If exhibit documents are attached WITHOUT explicit exhibit cover pages, \
  still recognize them as exhibits. Look for a clear structural break \
  after the pleading body (after the signature block): if pages after the \
  signature contain entirely different documents, those are exhibit \
  attachments. Classify the first page of each attached document as \
  "Exhibit cover page" with a sequentially inferred label (A, B, C...) \
  and all following pages as "Exhibit content" until the next structural break.

CRITICAL RULES FOR "Proof of service":
- "Proof of service" applies ONLY to the POS for THIS filing — the \
  document being filed with the court. It is almost always the very last \
  1-2 pages of the ENTIRE document (after all exhibits).
- NEVER classify a page as "Proof of service" if it falls inside an \
  exhibit section. POS pages within exhibits are "Exhibit content."
- If the document has exhibits, the filing's own POS (if any) appears \
  AFTER the last exhibit page.\
"""

PROMPT_CAPTION = """\
CAPTION EXTRACTION:
From the page classified as "Pleading first page," extract all available \
caption details. Use empty string for any field not found.

The document_title field is SEARCH-CONSTRAINED: provide the exact title \
text as it appears in the source OCR, plus the page number where it \
appears. Do not paraphrase or reformat. Include subtitles if present \
(e.g., "MOTION FOR SUMMARY JUDGMENT; MEMORANDUM OF POINTS AND AUTHORITIES \
IN SUPPORT THEREOF"). The search function will verify your string against \
the source text.

All other caption fields (court, case_number, filing_attorneys, judge, \
etc.) are NOT search-constrained — provide them directly.

An image of PDF page 1 is always provided alongside the text. This may or \
may not be the "Pleading first page" — sometimes page 1 is a cover sheet, \
a form, or a table of contents. If the image IS the pleading first page, \
use it to cross-check caption fields. Prefer the image-derived value for \
visually stamped elements (filing dates, clerk stamps) and the text-derived \
value for typed content (party names, document title, attorney information).

If page 1 is NOT the pleading first page (e.g., it is a cover sheet or \
form), you may add an image_review_request for the actual pleading first \
page so the second pass can extract caption data from that page's image.\
"""

PROMPT_SECTION_STRUCTURE = """\
SECTION PATH (for briefs, motions, and similar structured documents):
For each "Pleading body" page, identify the section hierarchy that the page \
falls under. Use the table of contents (if present) as the definitive \
outline. Format as a path string with " / " separators:
  - "INTRODUCTION"
  - "ARGUMENT / I. Standard of Review"
  - "ARGUMENT / II. The Trial Court Erred / A. Legal Framework"

If there is no TOC and the document is not clearly sectioned, use empty string.
For exhibit pages, form pages, POS pages, and TOC/TOA pages: use empty string.\
"""

PROMPT_CAUSE_OF_ACTION = """\
CAUSE OF ACTION EXTRACTION (for complaints and cross-complaints ONLY):
If the document is a complaint, cross-complaint, amended complaint, \
petition, or any initial pleading with causes of action, extract them.

For each COA, provide:
  - number: the COA number (1st, 2nd, etc.)
  - search_text: the EXACT title of the cause of action as it appears in \
    the source text. This is SEARCH-CONSTRAINED — the search function will \
    verify it. Do not split, combine, abbreviate, or rephrase titles. \
    Include statutory citations and parentheticals if present in the heading \
    (e.g., "NEGLIGENCE PER SE (Health & Saf. Code §§ 17920.3, 17920.10)").
  - page: the page number where the COA title appears
  - paragraph_range: the start and end paragraph numbers
  - page_range: the start and end page numbers where this COA appears
  - incorporates_by_reference: the paragraph range incorporated by \
    reference (start=0, end=0 if none)

For non-complaint documents (motions, briefs, declarations, etc.), return \
an empty causes_of_action array. Do NOT try to force-fit this structure \
onto documents that don't have numbered causes of action.\
"""

PROMPT_EXHIBIT = """\
EXHIBIT DETAILS:
- For each "Exhibit cover page," set exhibit_label to the letter or number \
  (e.g., "A", "1", "B-1") WITHOUT the word "Exhibit."
- For each "Exhibit content" page, set exhibit_label to the same label as \
  its preceding exhibit cover.
- For the FIRST "Exhibit content" page after each exhibit cover, set \
  exhibit_title to a brief (10-30 word) description of what the exhibit \
  document is. Leave exhibit_title empty for all other pages.

NESTED EXHIBITS (sub-documents within exhibits):
- Exhibits often contain sub-documents: attachments, enclosures, appendices, \
  sub-exhibits, or multiple separate documents bundled together.
- When you identify the START of a nested sub-document within an exhibit, \
  set nested_exhibit_label to its label (e.g., "Attachment 1", \
  "Enclosure A", "Appendix B").
- Set nested_exhibit_title to a brief description of the nested document \
  on its FIRST page only.
- Only set nested fields on the first page of each nested sub-document. \
  Leave them empty on continuation pages.

EXHIBIT NOTES (exhibit_notes column):
- Use exhibit_notes for observations about exhibit pages: what type of \
  document the page contains, structural markers (e.g., "proof of service \
  for underlying document," "signature page," "cover letter," "table of \
  contents for this exhibit"). This helps downstream processing understand \
  exhibit structure without misclassifying pages.
- Leave exhibit_notes empty for non-exhibit pages.\
"""

PROMPT_FOOTNOTE = """\
FOOTNOTE DETECTION:
For each page, check if it contains footnotes. Set has_footnote to true if \
footnotes appear on the page. In the footnotes array, list each footnote with:
  - fn_number: the footnote number (integer)
  - fn_text: the full text of the footnote as it appears on the page
  - merge_status: "merged" if the footnote text is clearly complete on this \
    page, "partial" if it continues to the next page, "missing" if only a \
    superscript reference exists but no footnote text was found, \
    "not_applicable" if no footnote detection was performed
  - merge_location: a brief description of where in the body text this \
    footnote's superscript reference appears (e.g., "after 'consistent with \
    the contract terms.'"). Empty string if not determinable.

If a page has no footnotes, set has_footnote to false and return an empty \
footnotes array.\
"""

PROMPT_IMAGE_REVIEW = """\
IMAGE REVIEW REQUESTS:
As you process the document text, identify pages where you NEED to see the \
actual page image to resolve an ambiguity that the OCR text alone cannot \
answer. Be conservative — only request images when genuinely needed.

Valid reasons to request an image:
  - "garbled_ocr": The OCR text on this page is garbled, truncated, or \
    clearly corrupted, making classification or extraction unreliable.
  - "handwritten": You suspect handwritten content (labels, annotations, \
    signatures with substantive text) that OCR may have missed.
  - "visual_element": The page appears to contain a map, diagram, chart, \
    photograph, or other visual element not captured by OCR.
  - "stamped_date": A filing date or other date appears to be from a \
    court clerk's stamp rather than typed text.
  - "footnote_unclear": Footnote numbering or text is ambiguous from OCR.
  - "other": Any other reason the image would resolve an uncertainty.

For each request, specify the page_number, reason (from the list above), \
and a specific question you want answered by looking at the image.\
"""

PROMPT_NOTES = """\
NOTES FIELD (pressure release valve):
The "notes" field (for non-exhibit pages) and "exhibit_notes" field (for \
exhibit pages) are for any observations, uncertainties, or contextual \
commentary you want to record. These fields are for your use — they will \
NOT be used in downstream processing. Use them freely for anything that \
doesn't fit the structured fields. Examples:
  - "This page appears to be a duplicate of page 3"
  - "OCR quality is poor; classification is low-confidence"
  - "Unusual formatting — may be a form but lacks form number"
  - "Signature block suggests end of main pleading"
Do not leave observations unrecorded just because there's no structured \
field for them. Put them in notes.\
"""


def assemble_system_prompt() -> str:
    """Join all prompt sections into the final system prompt."""
    sections = [
        PROMPT_BASE,
        PROMPT_CATEGORIES,
        PROMPT_CAPTION,
        PROMPT_SECTION_STRUCTURE,
        PROMPT_CAUSE_OF_ACTION,
        PROMPT_EXHIBIT,
        PROMPT_FOOTNOTE,
        PROMPT_IMAGE_REVIEW,
        PROMPT_NOTES,
    ]
    return "\n\n".join(sections)


# Backward compatibility: module-level SYSTEM_PROMPT
SYSTEM_PROMPT = assemble_system_prompt()


# ── Structured output JSON schema (Step 2) ────────────────────────────

RESPONSE_SCHEMA = {
    "name": "document_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "document_type": {
                "type": "string",
                "description": "Overall document type",
                "enum": DOCUMENT_TYPES,
            },
            "pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "1-based page number matching the === PAGE N === delimiter",
                        },
                        "category": {
                            "type": "string",
                            "description": "One of the classification categories",
                            "enum": CATEGORIES,
                        },
                        "subtype": {
                            "type": "string",
                            "description": "For Forms: the form number (e.g. 'SUM-100'). Empty string otherwise.",
                        },
                        "section_path": {
                            "type": "string",
                            "description": "For Pleading body pages: the section hierarchy path (e.g. 'ARGUMENT / I. Standard of Review'). Empty string for non-body pages or unstructured documents.",
                        },
                        "exhibit_label": {
                            "type": "string",
                            "description": "For Exhibit pages: the exhibit letter or number. Empty string if not an exhibit.",
                        },
                        "exhibit_title": {
                            "type": "string",
                            "description": "For the FIRST content page of each exhibit: a 10-30 word description. Empty string otherwise.",
                        },
                        "nested_exhibit_label": {
                            "type": "string",
                            "description": "If this page starts a sub-document within an exhibit, the nested label. Empty string otherwise.",
                        },
                        "nested_exhibit_title": {
                            "type": "string",
                            "description": "For the FIRST page of a nested sub-document: a brief title. Empty string otherwise.",
                        },
                        "exhibit_notes": {
                            "type": "string",
                            "description": "For exhibit pages ONLY: observations about the page content. Empty string for non-exhibit pages.",
                        },
                        "notes": {
                            "type": "string",
                            "description": "For NON-exhibit pages ONLY: general observations. Empty string for exhibit pages.",
                        },
                        "has_footnote": {
                            "type": "boolean",
                            "description": "True if this page contains footnotes.",
                        },
                        "footnotes": {
                            "type": "array",
                            "description": "Footnotes found on this page. Empty array if none.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "fn_number": {
                                        "type": "integer",
                                        "description": "The footnote number.",
                                    },
                                    "fn_text": {
                                        "type": "string",
                                        "description": "The full text of the footnote.",
                                    },
                                    "merge_status": {
                                        "type": "string",
                                        "description": "Status of footnote merge.",
                                        "enum": FOOTNOTE_MERGE_STATUSES,
                                    },
                                    "merge_location": {
                                        "type": "string",
                                        "description": "Where in the body text the superscript reference appears. Empty string if unknown.",
                                    },
                                },
                                "required": ["fn_number", "fn_text", "merge_status", "merge_location"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "page_number", "category", "subtype", "section_path",
                        "exhibit_label", "exhibit_title",
                        "nested_exhibit_label", "nested_exhibit_title",
                        "exhibit_notes", "notes",
                        "has_footnote", "footnotes",
                    ],
                    "additionalProperties": False,
                },
            },
            "caption_info": {
                "type": "object",
                "description": "Extracted from the Pleading first page. All fields empty string if not found.",
                "properties": {
                    "document_title": {
                        "type": "object",
                        "description": "SEARCH-CONSTRAINED: exact title from source text + page number.",
                        "properties": {
                            "search_text": {
                                "type": "string",
                                "description": "The exact document title as it appears in the source OCR text. Empty string if not found.",
                            },
                            "page": {
                                "type": "integer",
                                "description": "The page number where this title appears.",
                            },
                        },
                        "required": ["search_text", "page"],
                        "additionalProperties": False,
                    },
                    "filing_date": {
                        "type": "string",
                        "description": "Filing date as stamped or printed",
                    },
                    "filing_party": {
                        "type": "string",
                        "description": "Who filed: e.g. 'Plaintiff', 'Defendants'",
                    },
                    "named_plaintiffs": {
                        "type": "string",
                        "description": "Comma-separated list of named plaintiffs",
                    },
                    "named_defendants": {
                        "type": "string",
                        "description": "Comma-separated list of named defendants",
                    },
                    "filing_attorneys": {
                        "type": "string",
                        "description": "Attorney name(s) and firm",
                    },
                    "court": {
                        "type": "string",
                        "description": "Court name",
                    },
                    "case_number": {
                        "type": "string",
                        "description": "Case number",
                    },
                    "judge": {
                        "type": "string",
                        "description": "Judge name if shown",
                    },
                    "department": {
                        "type": "string",
                        "description": "Department number if shown",
                    },
                    "hearing_date": {
                        "type": "string",
                        "description": "Hearing date if shown",
                    },
                    "hearing_time": {
                        "type": "string",
                        "description": "Hearing time if shown",
                    },
                },
                "required": [
                    "document_title", "filing_date", "filing_party",
                    "named_plaintiffs", "named_defendants", "filing_attorneys",
                    "court", "case_number", "judge", "department",
                    "hearing_date", "hearing_time",
                ],
                "additionalProperties": False,
            },
            "causes_of_action": {
                "type": "array",
                "description": "For complaints only. Empty array for non-complaints.",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "COA number (1, 2, 3...)",
                        },
                        "search_text": {
                            "type": "string",
                            "description": "SEARCH-CONSTRAINED: the exact cause of action title as it appears in the source text. Do not paraphrase.",
                        },
                        "page": {
                            "type": "integer",
                            "description": "The page number where this COA title appears.",
                        },
                        "paragraph_range": {
                            "type": "object",
                            "description": "Start and end paragraph numbers",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                            "required": ["start", "end"],
                            "additionalProperties": False,
                        },
                        "page_range": {
                            "type": "object",
                            "description": "Start and end page numbers",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                            "required": ["start", "end"],
                            "additionalProperties": False,
                        },
                        "incorporates_by_reference": {
                            "type": "object",
                            "description": "Paragraph range incorporated by reference. Use start=0, end=0 if none.",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                            "required": ["start", "end"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "number", "search_text", "page",
                        "paragraph_range", "page_range",
                        "incorporates_by_reference",
                    ],
                    "additionalProperties": False,
                },
            },
            "image_review_requests": {
                "type": "array",
                "description": "Pages where the LLM needs to see the actual image. Empty array if none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "The page to review",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why the image is needed",
                            "enum": IMAGE_REVIEW_REASONS,
                        },
                        "question": {
                            "type": "string",
                            "description": "Specific question to answer from the image",
                        },
                    },
                    "required": ["page_number", "reason", "question"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "document_type", "pages", "caption_info",
            "causes_of_action", "image_review_requests",
        ],
        "additionalProperties": False,
    },
}

# Pass 2 schema for image review responses
IMAGE_REVIEW_RESPONSE_SCHEMA = {
    "name": "image_review_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "The page this answer is for",
                        },
                        "answer": {
                            "type": "string",
                            "description": "Answer to the question about this page",
                        },
                        "updated_category": {
                            "type": "string",
                            "description": "Updated category if the image changes classification. Empty string if no change.",
                            "enum": CATEGORIES + [""],
                        },
                        "updated_exhibit_label": {
                            "type": "string",
                            "description": "Updated exhibit label from image. Empty string if no change.",
                        },
                        "updated_footnotes": {
                            "type": "array",
                            "description": "Updated footnotes from image review. Empty array if no changes.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "fn_number": {"type": "integer"},
                                    "fn_text": {"type": "string"},
                                    "merge_status": {
                                        "type": "string",
                                        "enum": FOOTNOTE_MERGE_STATUSES,
                                    },
                                    "merge_location": {"type": "string"},
                                },
                                "required": ["fn_number", "fn_text", "merge_status", "merge_location"],
                                "additionalProperties": False,
                            },
                        },
                        "caption_updates": {
                            "type": "object",
                            "description": "Updated caption fields from image. All empty string if no changes.",
                            "properties": {
                                "filing_date": {"type": "string"},
                                "document_title": {"type": "string"},
                                "case_number": {"type": "string"},
                            },
                            "required": ["filing_date", "document_title", "case_number"],
                            "additionalProperties": False,
                        },
                        "source": {
                            "type": "string",
                            "description": "Always 'image_review'",
                        },
                    },
                    "required": [
                        "page_number", "answer", "updated_category",
                        "updated_exhibit_label", "updated_footnotes",
                        "caption_updates", "source",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answers"],
        "additionalProperties": False,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────
PAGE_TXT_RE = re.compile(r"^page_(\d{4})\.txt$", re.IGNORECASE)
MD_PAGE_START_RE = re.compile(r"^---\[Start PDF page (\d+)\]---\s*$")
MD_PAGE_END_RE = re.compile(r"^---\[End PDF page (\d+)\]---\s*$")
PARAGRAPH_NUM_RE = re.compile(r"^\s*(?:¶\s*)?(\d+)\.\s", re.MULTILINE)


def read_document_pages(text_dir: Path) -> List[Dict]:
    """Read all page_XXXX.txt files, return sorted list of {number, filename, text}."""
    pages = []
    for p in sorted(text_dir.iterdir()):
        m = PAGE_TXT_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        text = p.read_text(encoding="utf-8", errors="replace")
        pages.append({"number": idx, "filename": p.name, "text": text})
    pages.sort(key=lambda x: x["number"])
    return pages


def read_combined_md(md_path: Path) -> List[Dict]:
    """Parse a combined .md file with ---[Start PDF page N]--- delimiters."""
    content = md_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    pages = []
    current_page = None
    current_lines = []

    for line in lines:
        start_m = MD_PAGE_START_RE.match(line)
        end_m = MD_PAGE_END_RE.match(line)

        if start_m:
            current_page = int(start_m.group(1))
            current_lines = []
        elif end_m:
            if current_page is not None:
                pages.append({
                    "number": current_page,
                    "filename": f"page_{current_page:04d}.txt",
                    "text": "\n".join(current_lines).strip(),
                })
            current_page = None
            current_lines = []
        elif current_page is not None:
            current_lines.append(line)

    pages.sort(key=lambda x: x["number"])
    return pages


def build_document_text(pages: List[Dict]) -> str:
    """Concatenate pages with clear delimiters."""
    parts = []
    for pg in pages:
        parts.append(f"=== PAGE {pg['number']} ===")
        parts.append(pg["text"])
        parts.append("")  # blank line separator
    return "\n".join(parts)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token estimate."""
    return int(len(text) / chars_per_token)


# ── Token budget awareness (Step 3) ────────────────────────────────────

def check_token_budget(
    document_text: str,
    model: str,
    system_prompt: str,
    num_images: int = 0,
    tokens_per_image: int = 1100,
    expected_output_tokens: int = 10000,
) -> Dict:
    """
    Estimate token usage and warn if approaching context limits.
    Returns dict with estimates and warning flag.
    """
    context_window = MODEL_CONTEXT_WINDOWS.get(model, 1_000_000)

    input_text_tokens = estimate_tokens(document_text + system_prompt)
    image_tokens = num_images * tokens_per_image
    total_input = input_text_tokens + image_tokens
    total_usage = total_input + expected_output_tokens
    usage_pct = (total_usage / context_window) * 100

    budget = {
        "context_window": context_window,
        "input_text_tokens": input_text_tokens,
        "image_tokens": image_tokens,
        "total_input": total_input,
        "expected_output_tokens": expected_output_tokens,
        "total_estimated": total_usage,
        "usage_percent": round(usage_pct, 1),
        "warning": usage_pct > 80,
    }

    log.info(
        "Token budget: ~%d input + ~%d images + ~%d output = ~%d total (%.1f%% of %dk window)",
        input_text_tokens, image_tokens, expected_output_tokens,
        total_usage, usage_pct, context_window // 1000,
    )
    if budget["warning"]:
        log.warning(
            "Token usage at %.1f%% of context window — approaching limit!",
            usage_pct,
        )

    return budget


# ── Image utilities (Step 4) ───────────────────────────────────────────

def find_source_pdf(md_path: Path) -> Optional[Path]:
    """
    Find the source PDF for a combined .md file.
    Convention: {name}_combined.md -> {name}.pdf in same directory.
    """
    stem = md_path.stem
    # Strip _combined suffix
    if stem.endswith("_combined"):
        pdf_stem = stem[: -len("_combined")]
    else:
        pdf_stem = stem

    # Check same directory
    pdf_path = md_path.parent / f"{pdf_stem}.pdf"
    if pdf_path.exists():
        return pdf_path

    # Check parent directory
    pdf_path = md_path.parent.parent / f"{pdf_stem}.pdf"
    if pdf_path.exists():
        return pdf_path

    log.debug("No source PDF found for %s", md_path.name)
    return None


def extract_page_image_base64(pdf_path: Path, page_number: int, dpi: int = 200) -> Optional[str]:
    """
    Extract a single page from a PDF as a base64-encoded JPEG.
    page_number is 1-based.
    """
    try:
        import fitz
    except ImportError:
        log.warning("PyMuPDF (fitz) not installed — cannot extract page images from PDF")
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if page_number < 1 or page_number > len(doc):
            log.warning("Page %d out of range for %s (%d pages)", page_number, pdf_path.name, len(doc))
            doc.close()
            return None

        page = doc.load_page(page_number - 1)  # 0-based index
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("jpeg")
        doc.close()

        return base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        log.warning("Failed to extract page %d from %s: %s", page_number, pdf_path.name, e)
        return None


def find_fallback_png(md_path: Path, page_number: int) -> Optional[Path]:
    """
    Look for a fallback PNG in doc_files/{name}/PNG/page_XXXX.png.
    """
    stem = md_path.stem
    if stem.endswith("_combined"):
        doc_name = stem[: -len("_combined")]
    else:
        doc_name = stem

    # Check sibling doc_files directory structure
    candidates = [
        md_path.parent / doc_name / "PNG" / f"page_{page_number:04d}.png",
        md_path.parent.parent / "doc_files" / doc_name / "PNG" / f"page_{page_number:04d}.png",
        md_path.parent / f"{doc_name}_classification" / f"page_{page_number:04d}.png",
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def get_page_image_base64(
    md_path: Optional[Path], pdf_path: Optional[Path], page_number: int
) -> Optional[str]:
    """
    Get a page image as base64 JPEG, trying PDF extraction first, then fallback PNG.
    """
    # Try PDF extraction first
    if pdf_path and pdf_path.exists():
        b64 = extract_page_image_base64(pdf_path, page_number)
        if b64:
            return b64

    # Fallback to PNG
    if md_path:
        png_path = find_fallback_png(md_path, page_number)
        if png_path:
            try:
                png_data = png_path.read_bytes()
                # Convert PNG to JPEG for consistency
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(png_data))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                log.warning("Failed to read fallback PNG %s: %s", png_path, e)

    return None


def save_sent_image(b64_data: str, output_dir: Path, label: str) -> Path:
    """
    Save a base64-encoded JPEG to the output directory for audit/debugging.
    label is e.g. 'page_0001_pass1' or 'page_0026_pass2'.
    Returns the path written.
    """
    images_dir = output_dir / "sent_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    img_path = images_dir / f"{label}.jpg"
    img_path.write_bytes(base64.b64decode(b64_data))
    log.info("Saved sent image: %s", img_path)
    return img_path


# ── API call (Step 5) ──────────────────────────────────────────────────

def classify_document(
    client: OpenAI,
    document_text: str,
    model: str = MODEL,
    page1_image_b64: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict:
    """
    Send the full document to the model and get structured classification.
    Returns parsed JSON matching RESPONSE_SCHEMA.

    If page1_image_b64 is provided, sends the first-page image alongside
    the text for caption cross-checking.
    """
    prompt = system_prompt or SYSTEM_PROMPT
    est_input = estimate_tokens(document_text + prompt)
    num_images = 1 if page1_image_b64 else 0
    log.info("Sending ~%d estimated tokens + %d image(s) to %s", est_input, num_images, model)

    # Build user message
    user_text = (
        "Below is the complete OCR text of a court filing. "
        "Classify every page and extract caption information.\n\n"
        + document_text
    )

    if page1_image_b64:
        # Multipart message: text + image (same pattern as 11_document_classifier.py)
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "text",
                "text": (
                    "\n\nThe image below is the FIRST PAGE of this filing. "
                    "Use it to cross-check caption information, especially "
                    "filing date stamps and case numbers that OCR may have garbled."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{page1_image_b64}",
                    "detail": "high",
                },
            },
        ]
    else:
        user_content = user_text

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": RESPONSE_SCHEMA,
        },
        temperature=0.1,
    )
    elapsed = time.time() - t0

    usage = response.usage
    log.info(
        "Pass 1 completed in %.1fs — input: %s, output: %s tokens",
        elapsed,
        usage.prompt_tokens if usage else "?",
        usage.completion_tokens if usage else "?",
    )

    content = response.choices[0].message.content
    return json.loads(content)


# ── Two-pass image review (Step 6) ─────────────────────────────────────

def execute_image_review(
    client: OpenAI,
    pass1_result: Dict,
    md_path: Optional[Path],
    pdf_path: Optional[Path],
    pages: List[Dict],
    model: str = MODEL,
    output_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Read image_review_requests from Pass 1, extract page images,
    send them to the model for review.
    Returns Pass 2 result or None if no review needed.
    """
    requests = pass1_result.get("image_review_requests", [])
    if not requests:
        log.info("No image review requests from Pass 1 — skipping Pass 2")
        return None

    log.info("Pass 2: %d image review request(s)", len(requests))

    # Build page text lookup
    page_text_map = {p["number"]: p["text"] for p in pages}

    # Collect images and build context
    image_parts = []
    questions_text_parts = []

    for req in requests:
        pg_num = req["page_number"]
        reason = req["reason"]
        question = req["question"]

        b64 = get_page_image_base64(md_path, pdf_path, pg_num)
        if not b64:
            log.warning("Could not obtain image for page %d — skipping", pg_num)
            continue

        # Save a copy of the image being sent
        if output_dir:
            save_sent_image(b64, output_dir, f"page_{pg_num:04d}_pass2")

        questions_text_parts.append(
            f"PAGE {pg_num} (reason: {reason}): {question}"
        )
        image_parts.append({
            "type": "text",
            "text": f"\n--- Image for PAGE {pg_num} ---",
        })
        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            },
        })

    if not image_parts:
        log.warning("No images could be obtained for any review requests")
        return None

    # Log token budget for Pass 2
    num_images = len([p for p in image_parts if p["type"] == "image_url"])
    check_token_budget(
        "\n".join(questions_text_parts),
        model,
        "Image review system prompt",
        num_images=num_images,
        expected_output_tokens=2000,
    )

    # Build Pass 2 messages
    system_msg = (
        "You are reviewing page images from a court filing to answer specific "
        "questions that could not be resolved from OCR text alone. For each page, "
        "answer the question and update any classification or metadata fields if "
        "the image reveals different information than the text suggested."
    )

    user_content = [
        {
            "type": "text",
            "text": (
                "Below are images of specific pages from a court filing, along with "
                "questions about each. Please answer each question and provide any "
                "updated classification data.\n\nQuestions:\n"
                + "\n".join(questions_text_parts)
                + "\n\nImages follow:"
            ),
        },
        *image_parts,
    ]

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": IMAGE_REVIEW_RESPONSE_SCHEMA,
        },
        temperature=0.1,
    )
    elapsed = time.time() - t0

    usage = response.usage
    log.info(
        "Pass 2 completed in %.1fs — input: %s, output: %s tokens",
        elapsed,
        usage.prompt_tokens if usage else "?",
        usage.completion_tokens if usage else "?",
    )

    content = response.choices[0].message.content
    return json.loads(content)


def merge_image_review(pass1_result: Dict, pass2_result: Dict) -> Dict:
    """
    Apply Pass 2 answers back into Pass 1 result.
    Creates caption_review field for discrepancies.
    Prefers image for filing_date stamps, text for typed content.
    """
    result = json.loads(json.dumps(pass1_result))  # deep copy

    # Build page lookup
    page_map = {p["page_number"]: p for p in result["pages"]}

    caption_review = {}

    for answer in pass2_result.get("answers", []):
        pg_num = answer["page_number"]

        # Update category if changed
        if answer.get("updated_category") and answer["updated_category"] in CATEGORIES:
            if pg_num in page_map:
                old_cat = page_map[pg_num]["category"]
                if old_cat != answer["updated_category"]:
                    log.info(
                        "Pass 2 updated page %d: %s -> %s",
                        pg_num, old_cat, answer["updated_category"],
                    )
                    page_map[pg_num]["category"] = answer["updated_category"]

        # Update exhibit label if changed
        if answer.get("updated_exhibit_label") and pg_num in page_map:
            page_map[pg_num]["exhibit_label"] = answer["updated_exhibit_label"]

        # Update footnotes if provided
        if answer.get("updated_footnotes") and pg_num in page_map:
            page_map[pg_num]["footnotes"] = answer["updated_footnotes"]
            page_map[pg_num]["has_footnote"] = len(answer["updated_footnotes"]) > 0

        # Caption updates — track discrepancies
        cap_updates = answer.get("caption_updates", {})
        for field in ["filing_date", "document_title", "case_number"]:
            new_val = cap_updates.get(field, "")
            if new_val:
                old_val = result["caption_info"].get(field, "")
                if old_val != new_val:
                    caption_review[field] = {
                        "text_value": old_val,
                        "image_value": new_val,
                        "source": "image_review",
                    }
                    # Prefer image for stamped dates, text for typed content
                    if field == "filing_date":
                        result["caption_info"][field] = new_val
                        caption_review[field]["chosen"] = "image"
                    else:
                        # Keep text-derived value for typed content
                        caption_review[field]["chosen"] = "text"

    if caption_review:
        result["caption_review"] = caption_review

    result["image_review_completed"] = True
    return result


# ── Search-constrained field processing ────────────────────────────────

def normalize_search_fields(result: Dict) -> Dict:
    """
    Flatten search-constrained fields from {search_text, page} objects
    to simple string values. This is the DEFAULT path — no verification.
    The prompt framing already constrains the LLM to provide verbatim text.
    """
    result = json.loads(json.dumps(result))  # deep copy

    # caption_info.document_title: {search_text, page} -> string
    dt = result.get("caption_info", {}).get("document_title", {})
    if isinstance(dt, dict):
        result["caption_info"]["document_title"] = dt.get("search_text", "")

    # causes_of_action[].search_text -> title (for downstream compatibility)
    for coa in result.get("causes_of_action", []):
        if "search_text" in coa:
            coa["title"] = coa.pop("search_text")
        # 'page' stays in the COA for reference

    return result


def verify_search_fields(result: Dict, pages: List[Dict]) -> Dict:
    """
    Run the search function on all search-constrained fields.
    Replaces LLM text with verbatim source text where found.
    Flags fields that fail verification.

    This is the OPTIONAL verification path (--verify flag).
    """
    result = json.loads(json.dumps(result))  # deep copy

    # caption_info.document_title
    dt = result.get("caption_info", {}).get("document_title", {})
    if isinstance(dt, dict) and dt.get("search_text"):
        verbatim = search_source_text(dt["search_text"], dt.get("page", 1), pages)
        if verbatim:
            result["caption_info"]["document_title"] = verbatim
            result["caption_info"]["document_title_verified"] = True
        else:
            result["caption_info"]["document_title"] = dt["search_text"]
            result["caption_info"]["document_title_proposed_value"] = dt["search_text"]
            result["caption_info"]["document_title_verified"] = False
            log.warning("document_title search failed: %r", dt["search_text"][:80])
    elif isinstance(dt, dict):
        result["caption_info"]["document_title"] = dt.get("search_text", "")

    # causes_of_action[].search_text
    for coa in result.get("causes_of_action", []):
        search_text = coa.get("search_text", "")
        coa_page = coa.get("page", 0)
        if search_text and coa_page > 0:
            verbatim = search_source_text(search_text, coa_page, pages)
            if verbatim:
                coa["title"] = verbatim
                coa["title_verified"] = True
            else:
                coa["title"] = search_text
                coa["title_proposed_value"] = search_text
                coa["title_verified"] = False
                log.warning(
                    "COA %d title search failed on page %d: %r",
                    coa.get("number", 0), coa_page, search_text[:80],
                )
        else:
            coa["title"] = search_text

        # Remove search_text from final output (replaced by title)
        coa.pop("search_text", None)

    return result


def verify_caption_fields(caption_info: Dict, pages: List[Dict]) -> Dict:
    """
    Lightweight verification for non-search-constrained caption fields.
    Just passes through — the search-constrained framing in the prompt
    is the primary hallucination control. Kept for backward compatibility.
    """
    return dict(caption_info)


# ── Footnote tracking (Step 8) ─────────────────────────────────────────

def generate_footnote_index(
    result: Dict, output_dir: Path, doc_name: str
) -> List[str]:
    """
    Collect all footnotes from result, write footnote_index.csv,
    validate sequential numbering, and return list of issues.
    """
    all_footnotes = []
    for page in result.get("pages", []):
        pg_num = page.get("page_number", 0)
        for fn in page.get("footnotes", []):
            all_footnotes.append({
                "fn_number": fn.get("fn_number", 0),
                "page": pg_num,
                "merge_status": fn.get("merge_status", "not_applicable"),
                "fn_text": fn.get("fn_text", ""),
            })

    if not all_footnotes:
        log.info("No footnotes found in document")
        return []

    # Sort by footnote number
    all_footnotes.sort(key=lambda x: x["fn_number"])

    # Write CSV
    csv_path = output_dir / f"{doc_name}_footnote_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fn_number", "page", "merge_status", "fn_text"]
        )
        writer.writeheader()
        writer.writerows(all_footnotes)
    log.info("Footnote index written: %s (%d footnotes)", csv_path, len(all_footnotes))

    # Validate
    issues = []
    seen_numbers = set()
    expected_num = 1

    for fn in all_footnotes:
        num = fn["fn_number"]

        # Check for gaps
        if num != expected_num:
            if num > expected_num:
                for missing in range(expected_num, num):
                    issues.append(f"MISSING footnote {missing} (gap before fn {num} on page {fn['page']})")
            else:
                issues.append(f"OUT OF ORDER: footnote {num} on page {fn['page']} (expected {expected_num})")

        # Check for duplicates
        if num in seen_numbers:
            issues.append(f"DUPLICATE: footnote {num} on page {fn['page']}")

        seen_numbers.add(num)
        expected_num = num + 1

    for issue in issues:
        log.warning("Footnote validation: %s", issue)

    return issues


# ── Cause of action verification (Step 9) ──────────────────────────────

def verify_causes_of_action(coa_list: List[Dict], pages: List[Dict]) -> List[Dict]:
    """
    Verify COA paragraph ranges against source text.
    Title verification is handled by search-constrained field processing.
    Returns COA list with paragraph verification flags.
    """
    if not coa_list:
        return []

    # Build set of paragraph numbers found in source text
    page_paragraphs = {}  # page_number -> set of paragraph numbers
    for pg in pages:
        nums = set()
        for m in PARAGRAPH_NUM_RE.finditer(pg["text"]):
            nums.add(int(m.group(1)))
        if nums:
            page_paragraphs[pg["number"]] = nums

    all_paragraphs = set()
    for nums in page_paragraphs.values():
        all_paragraphs.update(nums)

    verified_coas = []
    for coa in coa_list:
        v_coa = dict(coa)
        para_range = coa.get("paragraph_range", {})
        start = para_range.get("start", 0)
        end = para_range.get("end", 0)

        # Check paragraph range
        if start > 0 and end > 0:
            expected = set(range(start, end + 1))
            missing = expected - all_paragraphs
            v_coa["paragraph_range_verified"] = len(missing) == 0
            v_coa["missing_paragraphs"] = sorted(missing) if missing else []
        else:
            v_coa["paragraph_range_verified"] = False
            v_coa["missing_paragraphs"] = []

        verified_coas.append(v_coa)

    return verified_coas


# ── Manifest output (Step 10) ──────────────────────────────────────────

def write_manifest(
    result: Dict,
    pages: List[Dict],
    verified_caption: Dict,
    verified_coa: List[Dict],
    md_path: Optional[Path],
    pdf_path: Optional[Path],
    output_dir: Path,
    doc_name: str,
) -> Path:
    """
    Build and write manifest.json — the central data structure.
    """
    # Generate document_id from source file
    source_id = doc_name
    if md_path:
        source_id = hashlib.md5(str(md_path).encode()).hexdigest()[:12]
    elif pdf_path:
        source_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]

    # Build page lookup from LLM result
    classified = {p["page_number"]: p for p in result.get("pages", [])}

    # Build manifest pages
    manifest_pages = []
    for pg in pages:
        num = pg["number"]
        cl = classified.get(num, {})

        manifest_page = {
            "page_number": num,
            "text_file": f"text_pages/page_{num:04d}.txt",
            "png_file": f"PNG/page_{num:04d}.png",
            "layout_labels": [],  # populated by POVL if available
            "classification": cl.get("category", "Pleading body"),
            "section_path": cl.get("section_path", ""),
            "exhibit_label": cl.get("exhibit_label", ""),
            "exhibit_title": cl.get("exhibit_title", ""),
            "nested_exhibit_label": cl.get("nested_exhibit_label", ""),
            "nested_exhibit_title": cl.get("nested_exhibit_title", ""),
            "exhibit_notes": cl.get("exhibit_notes", ""),
            "notes": cl.get("notes", ""),
            "has_footnote": cl.get("has_footnote", False),
            "footnotes": cl.get("footnotes", []),
            "chunk_ids": [],  # populated by chunker later
        }
        manifest_pages.append(manifest_page)

    # Build caption for manifest (subset of verified_caption, without verification flags)
    caption_fields = [
        "document_title", "filing_date", "filing_party",
        "named_plaintiffs", "named_defendants", "filing_attorneys",
        "court", "case_number", "judge", "department",
        "hearing_date", "hearing_time",
    ]
    manifest_caption = {}
    for f in caption_fields:
        manifest_caption[f] = verified_caption.get(f, "")
        # Include verification flags
        if f"{f}_verified" in verified_caption:
            manifest_caption[f"{f}_verified"] = verified_caption[f"{f}_verified"]
            manifest_caption[f"{f}_match_type"] = verified_caption.get(f"{f}_match_type", "")

    manifest = {
        "document_id": source_id,
        "source_pdf": str(pdf_path.name) if pdf_path else "",
        "source_md": str(md_path.name) if md_path else "",
        "total_pages": len(pages),
        "document_type": result.get("document_type", "other"),
        "caption": manifest_caption,
        "pages": manifest_pages,
        "causes_of_action": verified_coa,
        "chunks": [],  # populated by chunker later
        "image_review_requests": result.get("image_review_requests", []),
        "image_review_completed": result.get("image_review_completed", False),
    }

    # Add caption_review if present (from Pass 2 discrepancies)
    if "caption_review" in result:
        manifest["caption_review"] = result["caption_review"]

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Manifest written: %s", manifest_path)
    return manifest_path


# ── Output writers (unchanged for backward compatibility) ──────────────

def write_classification_csv(
    result: Dict, pages: List[Dict], metadata_dir: Path, doc_name: str
) -> Path:
    """Write classification CSV in the same format as 11_document_classifier.py."""
    csv_path = metadata_dir / f"{doc_name}{OUTPUT_CSV_SUFFIX}"

    # Build a lookup from page_number -> classification
    classified = {p["page_number"]: p for p in result["pages"]}

    rows = []
    for pg in pages:
        num = pg["number"]
        png_filename = f"page_{num:04d}.png"

        cl = classified.get(num, {})
        rows.append({
            "filename": png_filename,
            "category": cl.get("category", "Pleading body"),
            "subtype": cl.get("subtype", ""),
            "exhibit_label": cl.get("exhibit_label", ""),
            "exhibit_title": cl.get("exhibit_title", ""),
            "nested_exhibit_label": cl.get("nested_exhibit_label", ""),
            "nested_exhibit_title": cl.get("nested_exhibit_title", ""),
            "exhibit_notes": cl.get("exhibit_notes", ""),
            "notes": cl.get("notes", ""),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "category", "subtype",
                        "exhibit_label", "exhibit_title",
                        "nested_exhibit_label", "nested_exhibit_title",
                        "exhibit_notes", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info("Classification CSV written: %s", csv_path)
    return csv_path


def write_caption_file(result: Dict, metadata_dir: Path, doc_name: str) -> Path:
    """Write caption info as JSON (same style as the vision-based pipeline)."""
    caption_path = metadata_dir / f"{doc_name}{OUTPUT_CAPTION_SUFFIX}"

    caption = result.get("caption_info", {})
    content = "```json\n" + json.dumps(caption, indent=2, ensure_ascii=False) + "\n```"
    caption_path.write_text(content, encoding="utf-8")

    log.info("Caption file written: %s", caption_path)
    return caption_path


# ── Comparison ─────────────────────────────────────────────────────────
def compare_classifications(metadata_dir: Path, doc_name: str, pages: List[Dict]):
    """Compare text-based vs image-based classification results."""
    text_csv = metadata_dir / f"{doc_name}{OUTPUT_CSV_SUFFIX}"

    image_csvs = [
        f for f in metadata_dir.glob("*_classification.csv")
        if OUTPUT_CSV_SUFFIX not in f.name
    ]
    if not image_csvs:
        log.warning("No image-based classification CSV found for comparison.")
        return
    image_csv = image_csvs[0]

    def load_csv(path):
        with path.open("r", encoding="utf-8", newline="") as f:
            return {row["filename"]: row for row in csv.DictReader(f)}

    text_data = load_csv(text_csv)
    image_data = load_csv(image_csv)

    all_files = sorted(set(text_data.keys()) | set(image_data.keys()))
    matches = 0
    mismatches = 0
    only_text = 0
    only_image = 0

    print(f"\n{'='*80}")
    print(f"COMPARISON: text-based vs image-based classification")
    print(f"Document: {doc_name}")
    print(f"{'='*80}")
    print(f"{'Page':<20} {'Image-based':<30} {'Text-based':<30}")
    print(f"{'-'*80}")

    for fn in all_files:
        t = text_data.get(fn, {})
        i = image_data.get(fn, {})
        t_cat = t.get("category", "—")
        i_cat = i.get("category", "—")

        if fn not in text_data:
            only_image += 1
            marker = "[IMAGE ONLY]"
        elif fn not in image_data:
            only_text += 1
            marker = "[TEXT ONLY]"
        elif t_cat == i_cat:
            matches += 1
            marker = ""
        else:
            mismatches += 1
            marker = " <-- MISMATCH"

        print(f"{fn:<20} {i_cat:<30} {t_cat:<30}{marker}")

        if t.get("exhibit_label") or i.get("exhibit_label"):
            t_ex = t.get("exhibit_label", "")
            i_ex = i.get("exhibit_label", "")
            if t_ex != i_ex:
                print(f"{'':20} {'  exhibit: ' + i_ex:<30} {'  exhibit: ' + t_ex:<30} <-- LABEL DIFF")

    total = matches + mismatches
    accuracy = (matches / total * 100) if total > 0 else 0
    print(f"{'-'*80}")
    print(f"Pages compared: {total}")
    print(f"  Matches:    {matches} ({accuracy:.1f}%)")
    print(f"  Mismatches: {mismatches} ({100-accuracy:.1f}%)")
    if only_text:
        print(f"  Text-only:  {only_text}")
    if only_image:
        print(f"  Image-only: {only_image}")
    print(f"{'='*80}\n")


# ── Refactored pipeline orchestration (Step 11) ────────────────────────

def _classify_and_write(
    pages: List[Dict],
    output_dir: Path,
    doc_name: str,
    client: OpenAI,
    force: bool,
    compare: bool,
    model: str = MODEL,
    md_path: Optional[Path] = None,
    verify: bool = False,
) -> bool:
    """
    Core pipeline: Pass 1 -> validate -> Pass 2 -> normalize/verify -> write.

    verify: if True, run search_source_text on constrained fields.
            Default False — the prompt framing is the primary control.
    """
    log.info("Processing %s — %d pages", doc_name, len(pages))

    # 1. Build document text
    document_text = build_document_text(pages)
    est_tokens = estimate_tokens(document_text)
    log.info("Document text: %d chars, ~%d tokens", len(document_text), est_tokens)

    # 2. Locate source PDF and extract page 1 image
    pdf_path = None
    page1_b64 = None

    if md_path:
        pdf_path = find_source_pdf(md_path)
        if pdf_path:
            log.info("Found source PDF: %s", pdf_path.name)

    page1_b64 = get_page_image_base64(md_path, pdf_path, 1)
    if page1_b64:
        log.info("Page 1 image obtained for caption cross-check")
        save_sent_image(page1_b64, output_dir, "page_0001_pass1")
    else:
        log.info("No page 1 image available — text-only classification")

    # Token budget check
    check_token_budget(
        document_text, model, SYSTEM_PROMPT,
        num_images=1 if page1_b64 else 0,
    )

    # 3. Pass 1: classify with text + page 1 image
    output_dir.mkdir(parents=True, exist_ok=True)
    result = classify_document(
        client, document_text, model=model,
        page1_image_b64=page1_b64,
    )

    # 4. Validate page count
    returned_pages = len(result.get("pages", []))
    if returned_pages != len(pages):
        log.warning(
            "Page count mismatch: document has %d pages, API returned %d classifications",
            len(pages), returned_pages,
        )

    # 5. Pass 2: image review if requests exist
    pass2_result = execute_image_review(
        client, result, md_path, pdf_path, pages, model=model,
        output_dir=output_dir,
    )
    if pass2_result:
        result = merge_image_review(result, pass2_result)

    # 6. Normalize search-constrained fields (or verify if --verify)
    if verify:
        log.info("Running search-constrained field verification")
        result = verify_search_fields(result, pages)
    else:
        result = normalize_search_fields(result)

    # Verify COA paragraph ranges (always — this is just regex checking)
    verified_coa = verify_causes_of_action(
        result.get("causes_of_action", []), pages
    )
    result["causes_of_action"] = verified_coa

    # Caption passthrough (search-constrained framing is the control)
    verified_caption = verify_caption_fields(
        result.get("caption_info", {}), pages
    )

    # 7. Write all outputs
    # a) CSV (backward-compatible)
    write_classification_csv(result, pages, output_dir, doc_name)

    # b) Caption file (backward-compatible)
    write_caption_file(result, output_dir, doc_name)

    # c) Raw JSON
    raw_path = output_dir / f"{doc_name}_text_classification_raw.json"
    raw_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Raw JSON saved: %s", raw_path)

    # d) Footnote index
    fn_issues = generate_footnote_index(result, output_dir, doc_name)
    if fn_issues:
        log.info("Footnote issues found: %d", len(fn_issues))

    # e) Manifest (central data structure)
    write_manifest(
        result, pages, verified_caption, verified_coa,
        md_path, pdf_path, output_dir, doc_name,
    )

    # Compare if requested
    if compare:
        compare_classifications(output_dir, doc_name, pages)

    return True


# ── Document processing ───────────────────────────────────────────────
def process_document(
    doc_dir: Path, client: OpenAI, force: bool, compare: bool,
    model: str = MODEL, verify: bool = False,
):
    """Process a single document folder."""
    text_dir = doc_dir / "text_pages"
    metadata_dir = doc_dir / "metadata"
    doc_name = doc_dir.name

    if not text_dir.is_dir():
        log.warning("[skip] %s — no text_pages/ directory", doc_name)
        return False

    # Check if already processed
    existing = list(metadata_dir.glob(f"*{OUTPUT_CSV_SUFFIX}"))
    if existing and not force:
        log.info("[skip] %s — text classification already exists. Use --force to re-run.", doc_name)
        if compare:
            pages = read_document_pages(text_dir)
            compare_classifications(metadata_dir, doc_name, pages)
        return True

    # Read pages
    pages = read_document_pages(text_dir)
    if not pages:
        log.warning("[skip] %s — no page_*.txt files found", doc_name)
        return False

    return _classify_and_write(
        pages, metadata_dir, doc_name, client, force, compare,
        model=model, verify=verify,
    )


def process_md_file(
    md_path: Path, client: OpenAI, force: bool,
    model: str = MODEL, verify: bool = False,
):
    """Process a combined .md file directly."""
    doc_name = md_path.stem
    output_dir = md_path.parent / f"{doc_name}_classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    existing = list(output_dir.glob(f"*{OUTPUT_CSV_SUFFIX}"))
    if existing and not force:
        log.info("[skip] %s — text classification already exists. Use --force to re-run.", doc_name)
        return True

    pages = read_combined_md(md_path)
    if not pages:
        log.error("No pages parsed from %s", md_path)
        return False

    return _classify_and_write(
        pages, output_dir, doc_name, client, force, compare=False,
        model=model, md_path=md_path, verify=verify,
    )


def process_all(base_dir: Path, client: OpenAI, force: bool, compare: bool, model: str = MODEL, verify: bool = False):
    """Process all document folders under base_dir."""
    doc_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "text_pages").is_dir()
    )

    if not doc_dirs:
        log.error("No document folders with text_pages/ found in %s", base_dir)
        return

    log.info("Found %d document(s) to process", len(doc_dirs))
    processed = 0
    skipped = 0

    for doc_dir in doc_dirs:
        try:
            ok = process_document(doc_dir, client, force, compare, model=model, verify=verify)
            if ok:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            log.error("Error processing %s: %s", doc_dir.name, e, exc_info=True)
            skipped += 1

    log.info("Done. Processed: %d, Skipped: %d", processed, skipped)


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Text-based document classifier using GPT-5.2 (whole-document, single API call)"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single document folder OR to doc_files/ for batch processing",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare text-based results against existing image-based classification",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify even if text classification already exists",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"OpenAI model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run search-constrained field verification against source text (off by default)",
    )
    args = parser.parse_args()

    # Initialize client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    model_name = args.model

    client = OpenAI(api_key=api_key)

    target = args.path.resolve()
    if not target.exists():
        log.error("Path does not exist: %s", target)
        sys.exit(1)

    # Determine input type: .md file, single doc folder, or batch directory
    if target.is_file() and target.suffix.lower() == ".md":
        process_md_file(target, client, args.force, model=model_name, verify=args.verify)
    elif (target / "text_pages").is_dir():
        process_document(target, client, args.force, args.compare, model=model_name, verify=args.verify)
    else:
        process_all(target, client, args.force, args.compare, model=model_name, verify=args.verify)


if __name__ == "__main__":
    main()
