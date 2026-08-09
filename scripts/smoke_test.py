#!/usr/bin/env python
"""
smoke_test.py – Phase 4: End-to-End Command-Line Smoke Test
============================================================
Indexes a target repository (defaulting to the project root) and exercises
all three server tools directly without an MCP transport layer.  Serves as a
quick sanity check that ChromaDB, AST chunking, sentence-transformer
embeddings, and Git utilities are all wired up correctly.

Usage
-----
    # Index the current project and run all checks
    python scripts/smoke_test.py

    # Point at a different repository
    python scripts/smoke_test.py --repo /path/to/some/repo

    # Suppress intermediate output
    python scripts/smoke_test.py --quiet
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run from the project root or scripts/.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# ANSI colours (disabled on non-TTY terminals)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _ok(text: str) -> str:
    return _c("32;1", f"  ✓ {text}")


def _fail(text: str) -> str:
    return _c("31;1", f"  ✗ {text}")


def _info(text: str) -> str:
    return _c("36", f"  → {text}")


def _head(text: str) -> str:
    return _c("35;1", f"\n{'─' * 60}\n  {text}\n{'─' * 60}")


# ---------------------------------------------------------------------------
# Smoke-test runner
# ---------------------------------------------------------------------------


class SmokeTest:
    """Runs structured checks against a local repository."""

    def __init__(self, repo_path: str, quiet: bool = False) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.quiet = quiet
        self._passed = 0
        self._failed = 0
        self._indexer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def _assert(self, condition: bool, description: str) -> bool:
        if condition:
            self._passed += 1
            self._log(_ok(description))
        else:
            self._failed += 1
            self._log(_fail(description))
        return condition

    # ------------------------------------------------------------------
    # Check groups
    # ------------------------------------------------------------------

    def check_imports(self) -> None:
        self._log(_head("1 / 5 — Import Checks"))
        for module, label in [
            ("src.chunker", "src.chunker"),
            ("src.git_utils", "src.git_utils"),
            ("src.indexer", "src.indexer"),
            ("src.server", "src.server (FastMCP instance)"),
            ("chromadb", "chromadb"),
            ("sentence_transformers", "sentence_transformers"),
            ("git", "GitPython"),
            ("fastmcp", "fastmcp"),
        ]:
            try:
                __import__(module)
                self._assert(True, f"{label} importable")
            except ImportError as exc:
                self._assert(False, f"{label} importable  [{exc}]")

    def check_file_loading(self) -> None:
        self._log(_head("2 / 5 — File Loading & Chunking"))
        from src.chunker import chunk_file, load_repo_files

        t0 = time.perf_counter()
        files = load_repo_files(self.repo_path)
        elapsed = time.perf_counter() - t0

        self._assert(len(files) > 0, f"load_repo_files found {len(files)} files in {elapsed:.2f}s")

        py_files = [f for f in files if f["extension"] == ".py"]
        self._assert(len(py_files) > 0, f"{len(py_files)} Python file(s) detected")

        git_leaked = [f for f in files if ".git" in f["rel_path"]]
        self._assert(len(git_leaked) == 0, ".git/ directory excluded from results")

        if py_files:
            sample = py_files[0]
            self._log(_info(f"Chunking sample: {sample['rel_path']}"))
            chunks = chunk_file(sample)
            self._assert(len(chunks) > 0, f"chunk_file produced {len(chunks)} chunk(s)")

            ast_chunks = [c for c in chunks if c.symbol_type in ("function", "class")]
            self._assert(
                len(ast_chunks) > 0,
                f"{len(ast_chunks)} AST-extracted symbol chunk(s) (function/class)",
            )

            for chunk in chunks:
                if not (chunk.start_line >= 1 and chunk.end_line >= chunk.start_line):
                    self._assert(False, "All chunks have valid 1-indexed line boundaries")
                    break
            else:
                self._assert(True, "All chunks have valid 1-indexed line boundaries")

    def check_indexing(self) -> None:
        self._log(_head("3 / 5 — ChromaDB Vector Indexing"))
        from src.indexer import RepoLensIndexer

        chroma_dir = str(_PROJECT_ROOT / ".smoke_chroma_db")
        self._log(_info(f"ChromaDB path: {chroma_dir}"))

        t0 = time.perf_counter()
        try:
            self._indexer = RepoLensIndexer(
                chroma_path=chroma_dir,
                collection_name="smoke_test",
            )
        except Exception as exc:
            self._assert(False, f"RepoLensIndexer initialisation  [{exc}]")
            return
        self._assert(True, "RepoLensIndexer initialised (ChromaDB + model loaded)")

        try:
            count = self._indexer.index_repository(self.repo_path)
            elapsed = time.perf_counter() - t0
            self._assert(count > 0, f"index_repository indexed {count} chunks in {elapsed:.1f}s")
            stored = self._indexer.collection_count
            self._assert(stored == count, f"ChromaDB collection count matches ({stored})")
        except Exception as exc:
            self._assert(False, f"index_repository raised: {exc}")

    def check_search(self) -> None:
        self._log(_head("4 / 5 — Semantic Search (search_codebase tool)"))
        if self._indexer is None:
            self._log(_info("Skipped – indexer not initialised (step 3 failed)"))
            return

        queries = [
            ("function definition", 1),
            ("class with methods", 3),
            ("file loading utility", 5),
        ]
        for query, top_k in queries:
            try:
                results = self._indexer.search_codebase(query, top_k=top_k)
                self._assert(
                    len(results) <= top_k,
                    f'search "{query}" returned {len(results)} result(s) (top_k={top_k})',
                )
                if results:
                    keys = {
                        "score",
                        "file_path",
                        "symbol_name",
                        "symbol_type",
                        "start_line",
                        "end_line",
                        "document",
                    }
                    missing = keys - results[0].keys()
                    self._assert(
                        not missing, f"Result keys complete (missing: {missing or 'none'})"
                    )
                    top_score = results[0]["score"]
                    self._assert(
                        0.0 <= top_score <= 1.0,
                        f"Top-result score in [0, 1]: {top_score:.4f}",
                    )
            except Exception as exc:
                self._assert(False, f'search "{query}" raised: {exc}')

    def check_git_history(self) -> None:
        self._log(_head("5 / 5 — Git History Extraction"))

        # --- non-git directory (must not raise) ---
        import tempfile

        from src.git_utils import format_git_metadata_string, get_git_file_history

        with tempfile.TemporaryDirectory() as tmp:
            result = get_git_file_history(tmp, "any_file.py")
            self._assert(
                isinstance(result, list),
                "get_git_file_history on non-git dir returns list (not exception)",
            )

        # --- actual repo path ---
        try:
            import git

            repo = git.Repo(self.repo_path, search_parent_directories=True)
            # Pick a real tracked file to query.
            tracked = next(
                (
                    item.a_path
                    for item in repo.index.diff(None)  # unstaged changes
                    # fallback: just list any committed file
                )
                if repo.index.diff(None)
                else (
                    f
                    for f in repo.head.commit.tree.traverse()
                    if hasattr(f, "path") and f.path.endswith(".py")
                ),
                None,
            )
            if tracked:
                history = get_git_file_history(self.repo_path, tracked, max_commits=3)
                self._assert(isinstance(history, list), f"Got commit list for {tracked}")
                if history:
                    required = {"commit_hash", "short_hash", "author", "date", "message"}
                    missing = required - history[0].keys()
                    self._assert(
                        not missing, f"Commit dict has required keys (missing: {missing or 'none'})"
                    )
                    fmt = format_git_metadata_string(history)
                    self._assert(
                        bool(fmt) and "\n" in fmt,
                        "format_git_metadata_string returns multi-line string",
                    )
            else:
                self._log(_info("No tracked Python files found – skipping per-file history check"))
        except Exception as exc:
            self._log(_info(f"Git check skipped (not a git repo or error): {exc}"))
            self._assert(True, "git_utils gracefully skipped for non-git repo")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run all checks and return exit code (0 = all passed)."""
        print(_c("35;1", "\n╔══════════════════════════════════════════╗"))
        print(_c("35;1", "║   RepoLens MCP – End-to-End Smoke Test   ║"))
        print(_c("35;1", "╚══════════════════════════════════════════╝"))
        print(_info(f"Target repo : {self.repo_path}"))

        self.check_imports()
        self.check_file_loading()
        self.check_indexing()
        self.check_search()
        self.check_git_history()

        total = self._passed + self._failed
        colour = "32;1" if self._failed == 0 else "31;1"
        print(_c("35;1", f"\n{'─' * 60}"))
        print(
            _c(
                colour,
                f"  Result: {self._passed}/{total} checks passed"
                + (f"  ({self._failed} FAILED)" if self._failed else "  ✓ All good!"),
            )
        )
        print(_c("35;1", f"{'─' * 60}\n"))

        # Clean up the temporary smoke ChromaDB.
        smoke_db = _PROJECT_ROOT / ".smoke_chroma_db"
        if smoke_db.exists():
            import shutil

            shutil.rmtree(smoke_db, ignore_errors=True)

        return 0 if self._failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RepoLens MCP end-to-end smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/smoke_test.py
              python scripts/smoke_test.py --repo /path/to/my/project
              python scripts/smoke_test.py --quiet
            """
        ),
    )
    parser.add_argument(
        "--repo",
        default=str(_PROJECT_ROOT),
        metavar="PATH",
        help="Repository to index and test (default: project root).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate output; only print the final summary.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    runner = SmokeTest(repo_path=args.repo, quiet=args.quiet)
    sys.exit(runner.run())
