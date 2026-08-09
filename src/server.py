"""
server.py - Phase 3: FastMCP Server Entry Point
================================================
Exposes three MCP tools over stdio transport for use with Claude Desktop,
Cursor, or any other MCP-compatible client:

* ``search_codebase``    - Semantic vector search over indexed repository code.
* ``read_file_content``  - Safe line-range reader for any file in the repo.
* ``get_file_history``   - Git commit provenance log for a single file.

Environment variables
---------------------
REPO_PATH
    Absolute or relative path to the local repository to index and serve.
    Defaults to the current working directory (``"."``) when unset.
CHROMA_PATH
    Directory where ChromaDB persists data.
    Defaults to ``./chroma_db``.

Usage
-----
Run directly with Python for stdio transport::

    REPO_PATH=/path/to/repo python src/server.py

Or register with Claude Desktop / Cursor in their MCP config JSON.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
import threading
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Logging - configured early so all sub-modules inherit the level.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("repolens.server")

# ---------------------------------------------------------------------------
# FastMCP import
# ---------------------------------------------------------------------------
from fastmcp import FastMCP  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Runtime configuration from environment
# ---------------------------------------------------------------------------
import hashlib

REPO_PATH: str = os.environ.get("REPO_PATH", ".")
_repo_abs = os.path.abspath(REPO_PATH)
_repo_hash = hashlib.sha256(_repo_abs.encode()).hexdigest()[:12]
_default_db_path = os.path.expanduser(f"~/.repolens/db_{_repo_hash}")

CHROMA_PATH: str = os.environ.get("CHROMA_PATH", _default_db_path)

# ---------------------------------------------------------------------------
# Server + indexer initialisation
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP("RepoLens MCP")

# Deferred at module level so tests can patch before import triggers indexing.
_indexer = None
_indexer_lock = threading.Lock()

def _get_indexer():
    """Return the module-level indexer, initialising it on first access."""
    global _indexer
    if _indexer is not None:
        return _indexer
        
    with _indexer_lock:
        if _indexer is None:
            from src.indexer import RepoLensIndexer

            logger.info("Initialising RepoLensIndexer for repo: %s", REPO_PATH)
            _indexer = RepoLensIndexer(chroma_path=CHROMA_PATH)
            count = _indexer.index_repository(REPO_PATH)
            logger.info("Indexing complete - %d chunks stored.", count)
    return _indexer


# ---------------------------------------------------------------------------
# Helper: result formatter
# ---------------------------------------------------------------------------

_SNIPPET_MAX_LINES: int = 20  # Maximum lines of source shown per search result


def _format_search_results(results: list[dict]) -> str:
    """
    Convert a list of search-result dicts (from ``RepoLensIndexer.search_codebase``)
    into a clean, human-readable string for MCP tool return values.
    """
    if not results:
        return "No results found."

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        score = r.get("score", 0.0)
        file_path = r.get("file_path", "unknown")
        symbol = r.get("symbol_name", "")
        sym_type = r.get("symbol_type", "")
        start_l = r.get("start_line", 0)
        end_l = r.get("end_line", 0)

        # Trim the source snippet to avoid overwhelming the context window.
        doc: str = r.get("document", "")
        # Extract everything after the "---" separator (the raw code).
        sep = "---\n"
        code_part = doc[doc.find(sep) + len(sep) :] if sep in doc else doc
        snippet_lines = code_part.splitlines()[:_SNIPPET_MAX_LINES]
        snippet = "\n".join(snippet_lines)
        if len(code_part.splitlines()) > _SNIPPET_MAX_LINES:
            snippet += f"\n… ({len(code_part.splitlines()) - _SNIPPET_MAX_LINES} more lines)"

        block = (
            f"── Result {i} ────────────────────────────────────────────\n"
            f"Score      : {score:.4f}\n"
            f"File       : {file_path}\n"
            f"Symbol     : {symbol} ({sym_type})  Lines {start_l}-{end_l}\n"
            f"─────────────────────────────────────────────────────────\n"
            f"{snippet}\n"
        )
        parts.append(block)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_codebase(query: str, top_k: int = 5) -> str:
    """
    Search the indexed repository codebase using semantic vector similarity.

    Finds code chunks (functions, classes, or module-level code) that are
    semantically relevant to the query and returns them ranked by similarity.

    Parameters
    ----------
    query:
        A natural-language description or code snippet to search for.
        Examples: "function that authenticates users", "database connection pool",
        "retry logic with exponential back-off".
    top_k:
        Maximum number of results to return (default 5, max recommended 20).

    Returns
    -------
    str
        Formatted list of matching code chunks with similarity scores, file
        paths, line boundaries, symbol names, and source code snippets.
        Returns "No results found." when the index is empty or the query
        has no matching chunks.
    """
    if not query or not query.strip():
        return "Error: query must be a non-empty string."

    top_k = max(1, min(top_k, 50))  # guard against extreme values

    try:
        indexer = _get_indexer()
        results = indexer.search_codebase(query=query, top_k=top_k)
        return _format_search_results(results)
    except Exception as exc:
        logger.error("search_codebase error: %s", exc, exc_info=True)
        return f"Error during search: {exc}"


@mcp.tool()
def read_file_content(file_path: str, start_line: int = 1, end_line: int = 200) -> str:
    """
    Read a slice of a source file from the indexed repository.

    Parameters
    ----------
    file_path:
        Path to the file **relative to the repository root** (e.g.
        ``src/utils.py`` or ``tests/test_auth.py``).
    start_line:
        1-indexed line number to begin reading from (inclusive). Defaults to 1.
    end_line:
        1-indexed line number to stop reading at (inclusive). Defaults to 200.
        Set to a large number (e.g. 99999) to read the entire file.

    Returns
    -------
    str
        The requested line range with 1-indexed line numbers prepended, e.g.::

             1 | def foo():
             2 |     return 42

        Returns a descriptive error string if the file does not exist, cannot
        be read, or the requested range is out of bounds.
    """
    if not file_path or not file_path.strip():
        return "Error: file_path must be a non-empty string."

    # Sanitise: prevent path traversal outside the repo.
    repo_root = Path(REPO_PATH).resolve()
    target = (repo_root / file_path.lstrip("/\\")).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        return f"Error: '{file_path}' is outside the repository root."

    if not target.exists():
        return f"Error: file not found – '{file_path}'."
    if not target.is_file():
        return f"Error: '{file_path}' is not a regular file."

    try:
        all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Error reading '{file_path}': {exc}"

    total = len(all_lines)

    # Clamp to valid range.
    s = max(1, start_line)
    e = min(end_line, total)

    if s > total:
        return (
            f"Error: start_line {start_line} exceeds file length ({total} lines) for '{file_path}'."
        )
    if s > e:
        return f"Error: start_line ({start_line}) is greater than end_line ({end_line})."

    selected = all_lines[s - 1 : e]  # convert to 0-indexed slice
    width = len(str(e))  # column width for line numbers

    header = f"── {file_path}  (lines {s}–{e} of {total}) ──"
    numbered = "\n".join(f"{s + idx:>{width}} | {line}" for idx, line in enumerate(selected))
    return f"{header}\n{numbered}"


@mcp.tool()
def get_file_history(file_path: str) -> str:
    """
    Retrieve the recent Git commit history for a file in the repository.

    Shows the 10 most-recent commits that touched the specified file,
    including the short commit hash, date, author, and commit message.

    Parameters
    ----------
    file_path:
        Path to the file **relative to the repository root** (e.g.
        ``src/chunker.py``).

    Returns
    -------
    str
        Formatted commit log for the file, e.g.::

            Git history for src/chunker.py (last 10 commits):
            ──────────────────────────────────────────────────
            [a1b2c3d] 2024-06-01 | Alice Smith <alice@example.com>: Add retry logic
            [d4e5f6g] 2024-05-20 | Bob Jones <bob@example.com>: Fix edge case

        Returns an informational message if the repository has no Git history
        or the file has never been committed.
    """
    if not file_path or not file_path.strip():
        return "Error: file_path must be a non-empty string."

    from src.git_utils import format_git_metadata_string, get_git_file_history

    try:
        history = get_git_file_history(
            repo_path=REPO_PATH,
            rel_file_path=file_path,
            max_commits=10,
        )
    except Exception as exc:
        logger.error("get_file_history error: %s", exc, exc_info=True)
        return f"Error retrieving git history for '{file_path}': {exc}"

    if not history:
        return (
            f"No git history found for '{file_path}'.\n"
            "The file may be untracked, or this is not a Git repository."
        )

    summary = format_git_metadata_string(history)
    header = (
        f"Git history for {file_path} "
        f"(last {len(history)} commit{'s' if len(history) != 1 else ''}):"
    )
    divider = "─" * len(header)
    return f"{header}\n{divider}\n{summary}"

# ---------------------------------------------------------------------------
# Background Pre-loading
# ---------------------------------------------------------------------------

def _preload_indexer() -> None:
    """Pre-load the indexer (ChromaDB + SentenceTransformer) in the background.
    This prevents the first tool call from timing out (e.g. 15s+ model load)."""
    try:
        _get_indexer()
    except Exception as exc:
        logger.error("Background pre-load failed: %s", exc)

threading.Thread(target=_preload_indexer, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # The indexer is eagerly loaded in a background thread above.

    # Expose a server-card.json for Smithery external registry scanning
    from starlette.responses import JSONResponse
    
    @mcp.custom_route("/.well-known/mcp/server-card.json", methods=["GET"])
    async def server_card(request):
        return JSONResponse({
            "name": "sanjeev1412-official/repolens-mcp",
            "version": "1.0.0",
            "description": "RepoLens MCP: Context Layer for Local Codebases",
            "tools": [
                {
                    "name": "search_codebase",
                    "description": "Search the indexed repository codebase using semantic vector similarity. Finds code chunks (functions, classes, or module-level code) that are semantically relevant to the query and returns them ranked by similarity."
                },
                {
                    "name": "read_file_content",
                    "description": "Read exact contents of a file within the workspace. Prepends line numbers to output for LLM readability. Protects against path traversal outside the repository."
                },
                {
                    "name": "get_file_history",
                    "description": "Retrieves the Git commit history (authors, dates, and messages) for a specific file to provide context on code provenance and rationale."
                }
            ]
        })

    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse":
        port = int(os.environ.get("PORT", 8000))
        logger.info("Starting FastMCP server with SSE transport on port %d", port)
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        logger.info("Starting FastMCP server with stdio transport")
        mcp.run(transport="stdio")
