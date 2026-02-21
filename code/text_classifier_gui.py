"""
Text Classifier GUI

Tkinter GUI for running the whole-document text classifier (13_text_classifier.py).
Lets users browse for .md files or document folders, run classification,
view results, merge footnotes, and compare against image-based classification.
"""
import csv
import json
import os
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_PDF_DIR = PROJECT_DIR / "PDFs"

# Import the classifier module
sys.path.insert(0, str(BASE_DIR))
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "text_classifier", BASE_DIR / "13_text_classifier.py"
    )
    tc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tc)
except Exception as e:
    tc = None
    _IMPORT_ERROR = str(e)

# Import the merge_footnotes module
try:
    _mf_spec = importlib.util.spec_from_file_location(
        "merge_footnotes", BASE_DIR / "merge_footnotes.py"
    )
    mf = importlib.util.module_from_spec(_mf_spec)
    _mf_spec.loader.exec_module(mf)
except Exception as e:
    mf = None
    _MF_IMPORT_ERROR = str(e)


# ── Page delimiters (same as merge_footnotes.py) ─────────────────────
PAGE_START_RE = re.compile(r"^---\[Start PDF page (\d+)\]---\s*$")
PAGE_END_RE = re.compile(r"^---\[End PDF page (\d+)\]---\s*$")


def _split_merged_into_pages(content: str) -> list:
    """
    Split merged .md content into per-page text blocks.
    Returns [{page_number, text}, ...].
    """
    pages = []
    current_page = None
    current_lines = []

    for line in content.split("\n"):
        sm = PAGE_START_RE.match(line)
        em = PAGE_END_RE.match(line)
        if sm:
            current_page = int(sm.group(1))
            current_lines = []
        elif em:
            if current_page is not None:
                pages.append({
                    "page_number": current_page,
                    "text": "\n".join(current_lines),
                })
            current_page = None
            current_lines = []
        elif current_page is not None:
            current_lines.append(line)
    return pages


def _build_merge_json(
    merged_content: str,
    stats: dict,
    classification_data: dict | None,
    doc_name: str,
) -> dict:
    """
    Build the per-page + section-split JSON output from a footnote merge.

    classification_data: manifest.json or raw JSON (for section_path, classification).
    """
    merged_pages = _split_merged_into_pages(merged_content)

    # Build lookup from classification data
    class_lookup = {}  # page_number -> {classification, section_path, ...}
    if classification_data:
        for pg in classification_data.get("pages", []):
            pn = pg.get("page_number", 0)
            class_lookup[pn] = {
                "classification": pg.get("classification", pg.get("category", "")),
                "section_path": pg.get("section_path", ""),
                "has_footnote": pg.get("has_footnote", False),
                "footnotes": pg.get("footnotes", []),
            }

    # Build merged footnote numbers per page from stats
    merged_fn_by_page = {}
    for pd in stats.get("page_details", []):
        merged_fn_by_page[pd["page"]] = pd.get("merged_refs", [])

    # Build per-page output
    pages_out = []
    for mp in merged_pages:
        pn = mp["page_number"]
        cl = class_lookup.get(pn, {})
        pages_out.append({
            "page_number": pn,
            "classification": cl.get("classification", ""),
            "section_path": cl.get("section_path", ""),
            "section_heading": _lowest_heading(cl.get("section_path", "")),
            "text": mp["text"],
            "footnotes_merged": merged_fn_by_page.get(pn, []),
        })

    # Build sections by grouping consecutive pages with same lowest heading
    sections = _build_sections(pages_out)

    return {
        "document": doc_name,
        "total_pages": len(pages_out),
        "merge_stats": {
            "source": stats.get("source", ""),
            "footnotes_found": stats.get("total_found", 0),
            "footnotes_merged": stats.get("total_merged", 0),
            "footnotes_no_ocr_text": stats.get("total_no_ocr_text", 0),
            "footnotes_no_body_ref": stats.get("total_unmatched", 0),
        },
        "sections": sections,
        "pages": pages_out,
    }


def _lowest_heading(section_path: str) -> str:
    """Extract the lowest (most specific) heading from a section_path."""
    if not section_path:
        return ""
    parts = [p.strip() for p in section_path.split("/")]
    return parts[-1] if parts else ""


def _build_sections(pages_out: list) -> list:
    """
    Group consecutive pages by their lowest-level section heading.
    Pages with no section_path go into an unnamed section.
    """
    if not pages_out:
        return []

    sections = []
    current_heading = None
    current_path = None
    current_pages = []
    current_texts = []

    for pg in pages_out:
        heading = pg["section_heading"]
        path = pg["section_path"]

        if heading != current_heading:
            # Flush previous section
            if current_pages:
                sections.append({
                    "section_heading": current_heading or "(no heading)",
                    "section_path": current_path or "",
                    "page_numbers": current_pages,
                    "text": "\n\n".join(current_texts),
                })
            current_heading = heading
            current_path = path
            current_pages = [pg["page_number"]]
            current_texts = [pg["text"]]
        else:
            current_pages.append(pg["page_number"])
            current_texts.append(pg["text"])
            # Keep the most complete section_path seen for this heading
            if len(path) > len(current_path or ""):
                current_path = path

    # Flush last section
    if current_pages:
        sections.append({
            "section_heading": current_heading or "(no heading)",
            "section_path": current_path or "",
            "page_numbers": current_pages,
            "text": "\n\n".join(current_texts),
        })

    return sections


class TextClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Classifier (GPT-5.2)")
        self.root.geometry("960x780")
        self.root.minsize(750, 600)

        self.processing = False
        self._stop_requested = False
        self._client = None

        self._create_widgets()
        self._scan_documents()

    # ── UI Construction ──────────────────────────────────────────────────

    def _create_widgets(self):
        # ---- Top: folder path + scan ----
        path_frame = ttk.LabelFrame(self.root, text="Document Source", padding=10)
        path_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.source_dir_var = tk.StringVar(value=str(DEFAULT_PDF_DIR))
        ttk.Label(path_frame, text="Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.source_dir_var, width=70).grid(
            row=0, column=1, sticky="ew", padx=5
        )
        ttk.Button(path_frame, text="Browse...", command=self._browse_folder).grid(
            row=0, column=2
        )
        ttk.Button(path_frame, text="Scan", command=self._scan_documents).grid(
            row=0, column=3, padx=(5, 0)
        )
        path_frame.columnconfigure(1, weight=1)

        # ---- Middle: file list ----
        list_frame = ttk.LabelFrame(self.root, text="Documents", padding=5)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Treeview with columns
        cols = ("filename", "pages", "status", "manifest", "merged")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                 selectmode="extended", height=10)
        self.tree.heading("filename", text="Document", anchor="w")
        self.tree.heading("pages", text="Pages", anchor="center")
        self.tree.heading("status", text="Status", anchor="w")
        self.tree.heading("manifest", text="Manifest", anchor="center")
        self.tree.heading("merged", text="FN Merged", anchor="center")
        self.tree.column("filename", width=420, stretch=True)
        self.tree.column("pages", width=55, stretch=False, anchor="center")
        self.tree.column("status", width=170, stretch=False)
        self.tree.column("manifest", width=70, stretch=False, anchor="center")
        self.tree.column("merged", width=75, stretch=False, anchor="center")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ---- Options row ----
        opts_frame = ttk.Frame(self.root)
        opts_frame.pack(fill="x", padx=10, pady=5)

        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts_frame, text="Force re-classify",
                         variable=self.force_var).pack(side="left")

        self.model_var = tk.StringVar(value="gpt-5.2")
        ttk.Label(opts_frame, text="  Model:").pack(side="left", padx=(15, 2))
        model_combo = ttk.Combobox(opts_frame, textvariable=self.model_var, width=18,
                                    values=["gpt-5.2", "gpt-4.1", "gpt-4.1-mini",
                                            "gpt-4.1-nano", "o3", "o4-mini"])
        model_combo.pack(side="left")

        # ---- Buttons ----
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.run_btn = ttk.Button(btn_frame, text="Classify Selected",
                                   command=self._start_classification)
        self.run_btn.pack(side="left", padx=(0, 5))

        self.run_all_btn = ttk.Button(btn_frame, text="Classify All",
                                       command=lambda: self._start_classification(all_docs=True))
        self.run_all_btn.pack(side="left", padx=(0, 5))

        self.stop_btn = ttk.Button(btn_frame, text="Stop", state="disabled",
                                    command=self._request_stop)
        self.stop_btn.pack(side="left", padx=(0, 15))

        # Separator
        ttk.Separator(btn_frame, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2
        )

        self.merge_btn = ttk.Button(btn_frame, text="Merge Footnotes",
                                     command=self._start_merge_footnotes)
        self.merge_btn.pack(side="left", padx=(0, 5))

        self.merge_all_btn = ttk.Button(
            btn_frame, text="Merge All FN",
            command=lambda: self._start_merge_footnotes(all_docs=True),
        )
        self.merge_all_btn.pack(side="left", padx=(0, 15))

        # Separator
        ttk.Separator(btn_frame, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2
        )

        self.view_btn = ttk.Button(btn_frame, text="View Results",
                                    command=self._view_results)
        self.view_btn.pack(side="left", padx=(0, 5))

        ttk.Button(btn_frame, text="Open Folder",
                   command=self._open_selected_folder).pack(side="left")

        ttk.Button(btn_frame, text="Select All",
                   command=self._select_all).pack(side="right")

        # ---- Log ----
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10,
                                                   state="disabled", wrap="word",
                                                   font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        # ---- Status bar ----
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief="sunken", anchor="w")
        status_bar.pack(fill="x", padx=10, pady=(0, 10))

    # ── Helpers ──────────────────────────────────────────────────────────

    def log(self, msg):
        self.root.after(0, self._log_append, msg)

    def _log_append(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_status(self, msg):
        self.root.after(0, self.status_var.set, msg)

    def _browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.source_dir_var.get() or str(PROJECT_DIR))
        if d:
            self.source_dir_var.set(d)
            self._scan_documents()

    def _select_all(self):
        for item in self.tree.get_children():
            self.tree.selection_add(item)

    # ── Scanning ─────────────────────────────────────────────────────────

    def _scan_documents(self):
        """Scan the source folder for .md files and document folders."""
        self.tree.delete(*self.tree.get_children())
        source = Path(self.source_dir_var.get())
        if not source.is_dir():
            return

        found = []

        # Look for combined .md files
        for md_file in sorted(source.glob("*_combined.md")):
            page_count = self._count_md_pages(md_file)
            status, has_manifest = self._check_status(md_file)
            has_merged = self._check_merged(md_file)
            found.append((md_file.name, page_count, status,
                          "Yes" if has_manifest else "No",
                          "Yes" if has_merged else "No",
                          str(md_file)))

        # Look for document folders with text_pages/
        for d in sorted(source.iterdir()):
            if d.is_dir() and (d / "text_pages").is_dir():
                page_count = len(list((d / "text_pages").glob("page_*.txt")))
                status, has_manifest = self._check_folder_status(d)
                has_merged = self._check_folder_merged(d)
                found.append((d.name, page_count, status,
                              "Yes" if has_manifest else "No",
                              "Yes" if has_merged else "No",
                              str(d)))

        for name, pages, status, manifest, merged, path in found:
            self.tree.insert("", "end",
                             values=(name, pages, status, manifest, merged),
                             tags=(path,))

        self.set_status(f"Found {len(found)} document(s)")

    def _count_md_pages(self, md_path: Path) -> int:
        """Count pages in a combined .md file."""
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
            return len(re.findall(r"---\[Start PDF page \d+\]---", text))
        except Exception:
            return 0

    def _check_status(self, md_path: Path) -> tuple:
        """Check if a .md file has already been classified. Returns (status, has_manifest)."""
        doc_name = md_path.stem
        output_dir = md_path.parent / f"{doc_name}_classification"
        csv_file = output_dir / f"{doc_name}_classification_text.csv"
        manifest_file = output_dir / "manifest.json"
        has_manifest = manifest_file.exists()

        if csv_file.exists():
            return ("Classified", has_manifest)
        return ("Not classified", False)

    def _check_folder_status(self, doc_dir: Path) -> tuple:
        """Check if a document folder has been text-classified. Returns (status, has_manifest)."""
        metadata_dir = doc_dir / "metadata"
        if not metadata_dir.is_dir():
            return ("Not classified", False)
        text_csvs = list(metadata_dir.glob("*_classification_text.csv"))
        manifest_file = metadata_dir / "manifest.json"
        has_manifest = manifest_file.exists()
        if text_csvs:
            return ("Classified", has_manifest)
        return ("Not classified", False)

    def _check_merged(self, md_path: Path) -> bool:
        """Check if a .md file has a fn_merged JSON output."""
        doc_name = md_path.stem
        output_dir = md_path.parent / f"{doc_name}_classification"
        return (output_dir / f"{doc_name}_fn_merged.json").exists()

    def _check_folder_merged(self, doc_dir: Path) -> bool:
        """Check if a document folder has fn_merged JSON output."""
        metadata_dir = doc_dir / "metadata"
        return any(metadata_dir.glob("*_fn_merged.json")) if metadata_dir.is_dir() else False

    # ── Classification ───────────────────────────────────────────────────

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                messagebox.showerror("Error",
                    "OPENAI_API_KEY environment variable is not set.\n\n"
                    "Set it before launching this GUI.")
                return None
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_selected_paths(self):
        """Get file paths from selected treeview items."""
        paths = []
        for item_id in self.tree.selection():
            tags = self.tree.item(item_id, "tags")
            if tags:
                paths.append(Path(tags[0]))
        return paths

    def _get_all_paths(self):
        """Get file paths from all treeview items."""
        paths = []
        for item_id in self.tree.get_children():
            tags = self.tree.item(item_id, "tags")
            if tags:
                paths.append(Path(tags[0]))
        return paths

    def _set_processing(self, active):
        """Enable/disable buttons during processing."""
        state = "disabled" if active else "normal"
        self.run_btn.config(state=state)
        self.run_all_btn.config(state=state)
        self.merge_btn.config(state=state)
        self.merge_all_btn.config(state=state)
        self.stop_btn.config(state="normal" if active else "disabled")
        self.processing = active

    def _start_classification(self, all_docs=False):
        if self.processing:
            return

        if tc is None:
            messagebox.showerror("Import Error",
                f"Could not import 13_text_classifier.py:\n{_IMPORT_ERROR}")
            return

        client = self._get_client()
        if client is None:
            return

        paths = self._get_all_paths() if all_docs else self._get_selected_paths()
        if not paths:
            messagebox.showinfo("Info", "No documents selected.")
            return

        self._set_processing(True)
        self._stop_requested = False

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        thread = threading.Thread(
            target=self._run_classification,
            args=(paths, client),
            daemon=True,
        )
        thread.start()

    def _request_stop(self):
        self._stop_requested = True
        self.log("\n--- Stop requested. Will halt after current document. ---\n")

    def _run_classification(self, paths, client):
        """Background thread: classify each document via tc.process_md_file / tc.process_document."""
        model = self.model_var.get()
        force = self.force_var.get()
        total = len(paths)
        success = 0
        failed = 0

        self.log(f"Starting classification of {total} document(s) with {model}\n")
        self.log(f"Force re-classify: {force}\n\n")

        for i, path in enumerate(paths, 1):
            if self._stop_requested:
                self.log("Stopped by user.\n")
                break

            name = path.name
            self.set_status(f"[{i}/{total}] Classifying: {name}")
            self.log(f"{'='*60}\n[{i}/{total}] {name}\n{'='*60}\n")

            try:
                t0 = time.time()

                if path.is_file() and path.suffix.lower() == ".md":
                    # Delegate fully to tc.process_md_file
                    ok = tc.process_md_file(path, client, force, model=model)
                    elapsed = time.time() - t0

                    if ok:
                        self.log(f"  Classified in {elapsed:.1f}s\n")
                        self._update_tree_status(name, f"Classified ({elapsed:.1f}s)", True)
                        success += 1
                    else:
                        self.log(f"  Skipped (already classified or no pages)\n")
                        self._update_tree_status(name, "Skipped", False)
                        success += 1

                elif path.is_dir() and (path / "text_pages").is_dir():
                    ok = tc.process_document(path, client, force, compare=False, model=model)
                    elapsed = time.time() - t0
                    if ok:
                        self._update_tree_status(name, f"Classified ({elapsed:.1f}s)", True)
                        success += 1
                    else:
                        self._update_tree_status(name, "Skipped", False)
                        success += 1

                else:
                    self.log(f"  Not a valid document. Skipping.\n")
                    failed += 1

                self.log("\n")

            except Exception as e:
                elapsed = time.time() - t0
                self.log(f"  ERROR: {e}\n\n")
                self._update_tree_status(name, "Error", False)
                failed += 1

        self.log(f"{'='*60}\n")
        self.log(f"Done. Success: {success}, Failed: {failed}\n")
        self.set_status(f"Done -- {success} classified, {failed} failed")

        self.root.after(0, self._processing_finished)

    def _processing_finished(self):
        self._set_processing(False)

    def _update_tree_status(self, doc_name, status, check_manifest=False):
        """Update the status and manifest columns for a document in the treeview."""
        def _update():
            for item_id in self.tree.get_children():
                vals = self.tree.item(item_id, "values")
                if vals and vals[0] == doc_name:
                    self.tree.set(item_id, "status", status)
                    if check_manifest:
                        # Re-check manifest existence
                        path = Path(self.tree.item(item_id, "tags")[0])
                        has_manifest = self._path_has_manifest(path)
                        self.tree.set(item_id, "manifest", "Yes" if has_manifest else "No")
                    break
        self.root.after(0, _update)

    def _update_tree_merged(self, doc_name, merged: bool):
        """Update the FN Merged column for a document."""
        def _update():
            for item_id in self.tree.get_children():
                vals = self.tree.item(item_id, "values")
                if vals and vals[0] == doc_name:
                    self.tree.set(item_id, "merged", "Yes" if merged else "No")
                    break
        self.root.after(0, _update)

    def _path_has_manifest(self, path: Path) -> bool:
        """Check if a path has a manifest.json in its output directory."""
        if path.is_file() and path.suffix.lower() == ".md":
            doc_name = path.stem
            output_dir = path.parent / f"{doc_name}_classification"
            return (output_dir / "manifest.json").exists()
        elif path.is_dir():
            return (path / "metadata" / "manifest.json").exists()
        return False

    # ── Merge Footnotes ──────────────────────────────────────────────────

    def _start_merge_footnotes(self, all_docs=False):
        """Start footnote merge for selected (or all) documents."""
        if self.processing:
            return

        if mf is None:
            messagebox.showerror("Import Error",
                f"Could not import merge_footnotes.py:\n{_MF_IMPORT_ERROR}")
            return

        paths = self._get_all_paths() if all_docs else self._get_selected_paths()
        if not paths:
            messagebox.showinfo("Info", "No documents selected.")
            return

        # Filter to only .md files (merge only works on combined .md)
        md_paths = [p for p in paths if p.is_file() and p.suffix.lower() == ".md"]
        if not md_paths:
            messagebox.showinfo("Info",
                "Footnote merge requires combined .md files.\n"
                "No .md files found in selection.")
            return

        self._set_processing(True)
        self._stop_requested = False

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        thread = threading.Thread(
            target=self._run_merge_footnotes,
            args=(md_paths,),
            daemon=True,
        )
        thread.start()

    def _run_merge_footnotes(self, md_paths):
        """Background thread: merge footnotes for each .md file."""
        total = len(md_paths)
        success = 0
        failed = 0

        self.log(f"Starting footnote merge for {total} document(s)\n")
        self.log(f"Authority: LLM for location, OCR for text\n\n")

        for i, md_path in enumerate(md_paths, 1):
            if self._stop_requested:
                self.log("Stopped by user.\n")
                break

            name = md_path.name
            doc_name = md_path.stem
            self.set_status(f"[{i}/{total}] Merging FN: {name}")
            self.log(f"{'='*60}\n[{i}/{total}] {name}\n{'='*60}\n")

            try:
                t0 = time.time()

                # Read source
                content = md_path.read_text(encoding="utf-8", errors="replace")

                # Load classification footnote inventory
                classification_fns = mf.load_classification_footnotes(md_path)
                if classification_fns:
                    self.log(f"  LLM inventory: {len(classification_fns)} footnotes\n")
                    for num in sorted(classification_fns.keys()):
                        fn = classification_fns[num]
                        self.log(f"    FN{num} (pg {fn['page']}, {fn['merge_status']})\n")
                else:
                    self.log(f"  No classification data -- using OCR markers only\n")

                # Run merge
                merged_content, stats = mf.process_document(content, classification_fns)

                self.log(f"  Merged: {stats['total_merged']}/{stats['total_found']} footnotes\n")
                if stats.get("total_no_ocr_text", 0) > 0:
                    self.log(f"  WARNING: {stats['total_no_ocr_text']} footnotes had no OCR text\n")
                if stats["total_unmatched"] > 0:
                    self.log(f"  WARNING: {stats['total_unmatched']} footnotes had no body ref\n")

                # Write merged .md
                merged_md_path = md_path.parent / f"{doc_name}_fn_merged.md"
                merged_md_path.write_text(merged_content, encoding="utf-8")
                self.log(f"  Wrote: {merged_md_path.name}\n")

                # Load classification data for section_path info
                classification_data = self._load_classification_data(md_path)

                # Build and write per-page + section JSON
                output_dir = md_path.parent / f"{doc_name}_classification"
                output_dir.mkdir(parents=True, exist_ok=True)
                merge_json = _build_merge_json(
                    merged_content, stats, classification_data, doc_name,
                )
                json_path = output_dir / f"{doc_name}_fn_merged.json"
                json_path.write_text(
                    json.dumps(merge_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                elapsed = time.time() - t0
                n_sections = len(merge_json.get("sections", []))
                self.log(f"  Wrote: {json_path.name} ({n_sections} sections)\n")
                self.log(f"  Done in {elapsed:.1f}s\n")

                self._update_tree_merged(name, True)
                success += 1

            except Exception as e:
                self.log(f"  ERROR: {e}\n")
                self._update_tree_merged(name, False)
                failed += 1

            self.log("\n")

        self.log(f"{'='*60}\n")
        self.log(f"Done. Success: {success}, Failed: {failed}\n")
        self.set_status(f"FN merge done -- {success} merged, {failed} failed")

        self.root.after(0, self._processing_finished)

    def _load_classification_data(self, md_path: Path) -> dict | None:
        """Load manifest.json or raw classification JSON for a .md file."""
        doc_name = md_path.stem
        output_dir = md_path.parent / f"{doc_name}_classification"

        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            try:
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        raw_jsons = list(output_dir.glob("*_text_classification_raw.json"))
        if raw_jsons:
            try:
                return json.loads(raw_jsons[0].read_text(encoding="utf-8"))
            except Exception:
                pass

        return None

    # ── View Results ─────────────────────────────────────────────────────

    def _view_results(self):
        """Open a results viewer window for the selected document."""
        paths = self._get_selected_paths()
        if not paths:
            messagebox.showinfo("Info", "Select a document first.")
            return

        path = paths[0]  # View first selected

        # Find output files
        if path.is_file() and path.suffix.lower() == ".md":
            doc_name = path.stem
            output_dir = path.parent / f"{doc_name}_classification"
        elif path.is_dir():
            doc_name = path.name
            output_dir = path / "metadata"
        else:
            return

        csv_file = None
        caption_file = None
        manifest_file = None

        if output_dir.is_dir():
            csv_candidates = list(output_dir.glob("*_classification_text.csv"))
            caption_candidates = list(output_dir.glob("*_caption_text.txt"))
            manifest_candidate = output_dir / "manifest.json"
            if csv_candidates:
                csv_file = csv_candidates[0]
            if caption_candidates:
                caption_file = caption_candidates[0]
            if manifest_candidate.exists():
                manifest_file = manifest_candidate

        if not csv_file:
            messagebox.showinfo("Info", f"No classification results found for:\n{path.name}")
            return

        self._show_results_window(doc_name, csv_file, caption_file, manifest_file)

    def _show_results_window(self, doc_name, csv_file, caption_file, manifest_file=None):
        """Display classification results in a new window."""
        win = tk.Toplevel(self.root)
        win.title(f"Results: {doc_name}")
        win.geometry("1200x750")
        win.minsize(900, 500)

        # Load manifest data if available
        manifest = None
        if manifest_file and manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # ---- Document type badge + summary ----
        header_frame = ttk.Frame(win)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        if manifest:
            doc_type = manifest.get("document_type", "unknown")
            total_pages = manifest.get("total_pages", 0)
            has_review = manifest.get("image_review_completed", False)
            num_coa = len(manifest.get("causes_of_action", []))
            num_fn = sum(
                len(p.get("footnotes", []))
                for p in manifest.get("pages", [])
            )

            type_text = f"Document Type: {doc_type.upper()}"
            summary_parts = [f"{total_pages} pages"]
            if num_fn > 0:
                summary_parts.append(f"{num_fn} footnotes")
            if num_coa > 0:
                summary_parts.append(f"{num_coa} causes of action")
            if has_review:
                summary_parts.append("Image review: completed")
            summary_text = " | ".join(summary_parts)

            ttk.Label(header_frame, text=type_text,
                      font=("", 11, "bold")).pack(side="left")
            ttk.Label(header_frame, text=f"    {summary_text}",
                      font=("", 9)).pack(side="left", padx=(15, 0))

        # ---- Caption info ----
        if caption_file and caption_file.exists():
            cap_frame = ttk.LabelFrame(win, text="Caption Information", padding=8)
            cap_frame.pack(fill="x", padx=10, pady=(5, 5))

            try:
                raw = caption_file.read_text(encoding="utf-8")
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw.rsplit("```", 1)[0]
                caption = json.loads(raw.strip())

                # Get verification flags from manifest if available
                verified_caption = {}
                if manifest:
                    verified_caption = manifest.get("caption", {})

                row = 0
                col = 0
                for key, val in caption.items():
                    if not val:
                        continue
                    label = key.replace("_", " ").title()

                    # Check verification status
                    is_verified = verified_caption.get(f"{key}_verified", None)
                    if is_verified is True:
                        verify_marker = " [v]"
                    elif is_verified is False:
                        verify_marker = " [!]"
                    else:
                        verify_marker = ""

                    ttk.Label(cap_frame, text=f"{label}:", font=("", 9, "bold")).grid(
                        row=row, column=col, sticky="ne", padx=(10, 3), pady=1
                    )
                    ttk.Label(cap_frame, text=f"{val}{verify_marker}",
                              wraplength=350).grid(
                        row=row, column=col + 1, sticky="nw", pady=1
                    )
                    row += 1
                    if row > 5 and col == 0:
                        row = 0
                        col = 2
            except Exception as e:
                ttk.Label(cap_frame, text=f"Error reading caption: {e}").pack()

        # ---- Causes of action (if complaint) ----
        if manifest and manifest.get("causes_of_action"):
            coa_frame = ttk.LabelFrame(win, text="Causes of Action", padding=5)
            coa_frame.pack(fill="x", padx=10, pady=(0, 5))

            for coa in manifest["causes_of_action"]:
                num = coa.get("number", "?")
                title = coa.get("title", "Unknown")
                para = coa.get("paragraph_range", {})
                para_str = f"pp. {para.get('start', '?')}-{para.get('end', '?')}"
                verified = coa.get("title_verified", None)
                v_mark = " [v]" if verified else (" [!]" if verified is False else "")
                ttk.Label(coa_frame,
                          text=f"  {num}. {title} ({para_str}){v_mark}",
                          font=("Consolas", 9)).pack(anchor="w")

        # ---- Classification table ----
        table_frame = ttk.LabelFrame(win, text="Page Classifications", padding=5)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        cols = ("page", "category", "section", "exhibit", "exhibit_title",
                "nested", "nested_title", "footnotes", "exhibit_notes", "notes")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)

        tree.heading("page", text="Page", anchor="w")
        tree.heading("category", text="Category", anchor="w")
        tree.heading("section", text="Section Path", anchor="w")
        tree.heading("exhibit", text="Exhibit", anchor="center")
        tree.heading("exhibit_title", text="Exhibit Title", anchor="w")
        tree.heading("nested", text="Nested", anchor="center")
        tree.heading("nested_title", text="Nested Title", anchor="w")
        tree.heading("footnotes", text="FN", anchor="center")
        tree.heading("exhibit_notes", text="Exhibit Notes", anchor="w")
        tree.heading("notes", text="Notes", anchor="w")

        tree.column("page", width=80, stretch=False)
        tree.column("category", width=140, stretch=False)
        tree.column("section", width=180, stretch=True)
        tree.column("exhibit", width=55, stretch=False, anchor="center")
        tree.column("exhibit_title", width=180, stretch=True)
        tree.column("nested", width=80, stretch=False, anchor="center")
        tree.column("nested_title", width=140, stretch=True)
        tree.column("footnotes", width=35, stretch=False, anchor="center")
        tree.column("exhibit_notes", width=130, stretch=True)
        tree.column("notes", width=130, stretch=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # Alternating row colors
        tree.tag_configure("exhibit", background="#e8f0fe")
        tree.tag_configure("cover", background="#d2e3fc")
        tree.tag_configure("pos", background="#fce8e6")
        tree.tag_configure("first", background="#e6f4ea")

        # Build footnote count from manifest if available
        fn_counts = {}
        if manifest:
            for mp in manifest.get("pages", []):
                fn_counts[mp.get("page_number", 0)] = len(mp.get("footnotes", []))

        # Load CSV data
        try:
            with csv_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    cat = row.get("category", "")
                    tag = ""
                    if cat == "Exhibit cover page":
                        tag = "cover"
                    elif cat == "Exhibit content":
                        tag = "exhibit"
                    elif cat == "Proof of service":
                        tag = "pos"
                    elif cat == "Pleading first page":
                        tag = "first"

                    # Get section_path and footnote count from manifest
                    page_num = row_idx + 1
                    section_path = ""
                    fn_count = fn_counts.get(page_num, 0)
                    if manifest:
                        for mp in manifest.get("pages", []):
                            if mp.get("page_number") == page_num:
                                section_path = mp.get("section_path", "")
                                break

                    tree.insert("", "end", values=(
                        row.get("filename", ""),
                        cat,
                        section_path,
                        row.get("exhibit_label", ""),
                        row.get("exhibit_title", ""),
                        row.get("nested_exhibit_label", ""),
                        row.get("nested_exhibit_title", ""),
                        str(fn_count) if fn_count > 0 else "",
                        row.get("exhibit_notes", ""),
                        row.get("notes", ""),
                    ), tags=(tag,))
        except Exception as e:
            ttk.Label(table_frame, text=f"Error reading CSV: {e}").grid(
                row=0, column=0, sticky="nsew"
            )

        # ---- Summary stats ----
        stat_frame = ttk.Frame(win)
        stat_frame.pack(fill="x", padx=10, pady=(0, 10))

        try:
            with csv_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            cats = {}
            exhibits = set()
            for r in rows:
                c = r.get("category", "unknown")
                cats[c] = cats.get(c, 0) + 1
                if r.get("exhibit_label"):
                    exhibits.add(r["exhibit_label"])

            summary = f"Total pages: {len(rows)}  |  "
            summary += "  |  ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
            if exhibits:
                summary += f"  |  Exhibits: {', '.join(sorted(exhibits))}"
            ttk.Label(stat_frame, text=summary, font=("", 8)).pack(anchor="w")
        except Exception:
            pass

    # ── Open folder ──────────────────────────────────────────────────────

    def _open_selected_folder(self):
        paths = self._get_selected_paths()
        if not paths:
            messagebox.showinfo("Info", "Select a document first.")
            return

        path = paths[0]
        if path.is_file():
            doc_name = path.stem
            output_dir = path.parent / f"{doc_name}_classification"
            if output_dir.is_dir():
                os.startfile(str(output_dir))
            else:
                os.startfile(str(path.parent))
        elif path.is_dir():
            os.startfile(str(path))


def main():
    root = tk.Tk()
    app = TextClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
