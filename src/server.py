"""
server.py - Phase 3: FastMCP Server Entry Point
================================================
Exposes four MCP tools over stdio transport for use with Claude Desktop,
Cursor, or any other MCP-compatible client:

* ``search_codebase``    - Semantic vector search over indexed repository code.
* ``read_file_content``  - Safe line-range reader for any file in the repo.
* ``get_file_history``   - Git commit provenance log for a single file.
* ``index_github_repo``  - Clone and index a remote GitHub repository on the fly.

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

import asyncio
import logging
import os
import shutil
import sys
import tempfile
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

# When set to "1", _get_indexer() skips the initial index_repository(REPO_PATH)
# call and only initialises the model + ChromaDB client. This is essential on
# memory-constrained cloud hosts (e.g. Render free tier, 512 MB) where:
#   a) REPO_PATH=/app points at the server itself, not a user repo, AND
#   b) the background GitHub indexing thread also runs, so double-indexing
#      would cause an OOM crash before any tool call can complete.
# Set to "0" (default) on local or higher-memory hosts to retain full behaviour.
_SKIP_INITIAL_INDEX: bool = os.environ.get("REPOLENS_SKIP_INITIAL_INDEX", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Server + indexer initialisation
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP("RepoLens MCP")

# Deferred at module level so tests can patch before import triggers indexing.
_indexer = None
_indexer_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Background indexing job tracker
# ---------------------------------------------------------------------------
# Maps repo_url → {"status": "running"|"done"|"error", "message": str, "count": int}
# Allows index_github_repo to return immediately while work continues in a thread.
_index_jobs: dict[str, dict] = {}
_index_jobs_lock = threading.Lock()

def _get_indexer():
    """Return the module-level indexer, initialising it on first access.

    When ``REPOLENS_SKIP_INITIAL_INDEX=1`` the indexer is initialised (model
    loaded, ChromaDB client opened) but ``index_repository(REPO_PATH)`` is
    intentionally skipped.  This is the correct mode for cloud deployments
    where ``REPO_PATH`` points at the server itself and GitHub repos are
    indexed on demand via ``index_github_repo``.
    """
    global _indexer
    if _indexer is not None:
        return _indexer

    with _indexer_lock:
        if _indexer is None:
            from src.indexer import RepoLensIndexer

            logger.info("Initialising RepoLensIndexer (skip_initial_index=%s)", _SKIP_INITIAL_INDEX)
            _indexer = RepoLensIndexer(chroma_path=CHROMA_PATH)

            if _SKIP_INITIAL_INDEX:
                logger.info(
                    "REPOLENS_SKIP_INITIAL_INDEX=1: skipping index_repository('%s').",
                    REPO_PATH,
                )
            else:
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

@mcp.tool()
def index_github_repo(repo_url: str) -> str:
    """Clones a public GitHub repository, parses its AST structure, and indexes code chunks into the RepoLens ChromaDB vector store.

    Indexing is performed asynchronously in a background thread so this tool
    returns immediately. Call ``get_index_status`` with the same ``repo_url``
    to poll for completion before running ``search_codebase``.

    Parameters
    ----------
    repo_url:
        The full Git clone URL of a public GitHub repository.
        Must start with ``https://github.com/`` or end with ``.git``.
        Example: ``https://github.com/user/repo.git``

    Returns
    -------
    str
        A message confirming the background job has started, including the
        clone directory. Check status with ``get_index_status``.
    """
    if not repo_url or not repo_url.strip():
        return "Error: repo_url must be a non-empty string."

    repo_url = repo_url.strip()

    # --- Validation -----------------------------------------------------------
    is_github = repo_url.startswith("https://github.com/")
    is_git_url = repo_url.endswith(".git")
    if not (is_github or is_git_url):
        return (
            "Error: invalid repo_url. "
            "It must start with 'https://github.com/' or end with '.git'. "
            f"Received: '{repo_url}'"
        )

    # --- Guard: reject if a job is already running for this URL ---------------
    with _index_jobs_lock:
        existing = _index_jobs.get(repo_url)
        if existing and existing["status"] == "running":
            return (
                f"Indexing already in progress for '{repo_url}'.\n"
                f"Call get_index_status(repo_url='{repo_url}') to check progress."
            )

    # --- Derive a stable clone directory from the URL -------------------------
    url_slug = repo_url.rstrip("/").rstrip(".git").rsplit("/", 2)
    repo_name_part = "_".join(url_slug[-2:]) if len(url_slug) >= 2 else url_slug[-1]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in repo_name_part)
    clone_dir = os.path.join(tempfile.gettempdir(), f"repolens_clone_{safe_name}")

    # --- Background worker ---------------------------------------------------
    def _worker() -> None:
        with _index_jobs_lock:
            _index_jobs[repo_url] = {"status": "running", "message": "Cloning repository…", "count": 0}

        try:
            from git import Repo, GitCommandError, GitCommandNotFound  # type: ignore
        except ImportError:
            with _index_jobs_lock:
                _index_jobs[repo_url] = {
                    "status": "error",
                    "message": "GitPython is not installed. Run: pip install GitPython",
                    "count": 0,
                }
            return

        # Clean up any stale clone
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, ignore_errors=True)

        logger.info("[index_job] Cloning '%s' → %s", repo_url, clone_dir)
        try:
            # Disable terminal prompts so git fails immediately instead of hanging on auth
            Repo.clone_from(repo_url, clone_dir, env={"GIT_TERMINAL_PROMPT": "0"})
        except GitCommandNotFound:
            with _index_jobs_lock:
                _index_jobs[repo_url] = {
                    "status": "error",
                    "message": "'git' binary not found. Install Git and ensure it is in PATH.",
                    "count": 0,
                }
            return
        except GitCommandError as exc:
            shutil.rmtree(clone_dir, ignore_errors=True)
            with _index_jobs_lock:
                _index_jobs[repo_url] = {
                    "status": "error",
                    "message": f"Clone failed: {exc.stderr.strip() if exc.stderr else exc}",
                    "count": 0,
                }
            return
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(clone_dir, ignore_errors=True)
            with _index_jobs_lock:
                _index_jobs[repo_url] = {
                    "status": "error",
                    "message": f"Clone failed: {exc}",
                    "count": 0,
                }
            return

        with _index_jobs_lock:
            _index_jobs[repo_url]["message"] = "Clone complete. Indexing code chunks…"

        try:
            from src.indexer import RepoLensIndexer
            primary = _get_indexer()
            github_indexer = RepoLensIndexer(
                chroma_path=primary.chroma_path,
                collection_name=primary._collection_name,
                model=primary._model,
            )
            count = github_indexer.index_repository(clone_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("[index_job] Indexing error: %s", exc, exc_info=True)
            with _index_jobs_lock:
                _index_jobs[repo_url] = {
                    "status": "error",
                    "message": f"Cloned OK but indexing failed: {exc}",
                    "count": 0,
                }
            return

        with _index_jobs_lock:
            _index_jobs[repo_url] = {
                "status": "done",
                "message": (
                    f"Cloned to {clone_dir}. "
                    f"Indexed into ChromaDB at {primary.chroma_path}."
                ),
                "count": count,
            }
        logger.info("[index_job] '%s' complete — %d chunks.", repo_url, count)

    # --- Launch and return immediately ----------------------------------------
    with _index_jobs_lock:
        _index_jobs[repo_url] = {"status": "running", "message": "Job queued.", "count": 0}
    threading.Thread(target=_worker, daemon=True, name=f"index:{repo_url}").start()

    return (
        f"✅ Indexing job started for '{repo_url}'.\n"
        f"   Clone target : {clone_dir}\n"
        f"   This runs in the background. Call:\n"
        f"   get_index_status(repo_url='{repo_url}')\n"
        f"   to check progress before calling search_codebase."
    )


@mcp.tool()
def get_index_status(repo_url: str) -> str:
    """Check the status of a background repository indexing job started by index_github_repo.

    Parameters
    ----------
    repo_url:
        The same Git clone URL passed to ``index_github_repo``.

    Returns
    -------
    str
        Current job status: running, done (with chunk count), or error details.
        When status is 'done', you may call ``search_codebase`` immediately.
    """
    if not repo_url or not repo_url.strip():
        return "Error: repo_url must be a non-empty string."

    repo_url = repo_url.strip()
    with _index_jobs_lock:
        job = _index_jobs.get(repo_url)

    if job is None:
        return (
            f"No indexing job found for '{repo_url}'.\n"
            "Call index_github_repo first."
        )

    status  = job["status"]
    message = job["message"]
    count   = job.get("count", 0)

    if status == "running":
        return f"⏳ Indexing in progress for '{repo_url}'…\n   {message}"
    if status == "done":
        return (
            f"✅ Indexing complete for '{repo_url}'.\n"
            f"   {message}\n"
            f"   Chunks indexed: {count}\n"
            f"   You can now call search_codebase to query this repository."
        )
    if status == "error":
        return f"❌ Indexing failed for '{repo_url}'.\n   Error: {message}"
    return f"Status: {status} — {message}"


# ---------------------------------------------------------------------------
# Background Pre-loading
# ---------------------------------------------------------------------------

def _preload_indexer() -> None:
    """Pre-load the indexer (ChromaDB + SentenceTransformer) in the background.
    This prevents the first tool call from timing out (e.g. 15s+ model load).
    Only runs when the REPOLENS_PRELOAD=1 environment variable is set to avoid
    OOM on memory-constrained hosts like Render's free tier (512 MB cap)."""
    try:
        _get_indexer()
    except Exception as exc:
        logger.error("Background pre-load failed: %s", exc)

_preload_enabled = os.environ.get("REPOLENS_PRELOAD", "0").strip() == "1"
if _preload_enabled:
    logger.info("REPOLENS_PRELOAD=1: launching background pre-load thread.")
    threading.Thread(target=_preload_indexer, daemon=True).start()
else:
    logger.info("Background pre-load disabled (set REPOLENS_PRELOAD=1 to enable).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # The indexer is eagerly loaded in a background thread above.

    # Expose a server-card.json for Smithery external registry scanning
    from starlette.responses import JSONResponse
    
    @mcp.custom_route("/.well-known/mcp/server-card.json", methods=["GET"])
    @mcp.custom_route("/sse/.well-known/mcp/server-card.json", methods=["GET"])
    async def server_card(request):
        return JSONResponse({
            "serverInfo": {
                "name": "sanjeev1412-official/repolens-mcp",
                "version": "1.0.0"
            },
            "description": "RepoLens MCP: Context Layer for Local Codebases",
            "authentication": {
                "required": False
            },
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
                },
                {
                    "name": "index_github_repo",
                    "description": "Clones a public GitHub repository, parses its AST structure, and indexes code chunks into the RepoLens ChromaDB vector store.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "repo_url": {
                                "type": "string",
                                "description": "The full Git clone URL (e.g., https://github.com/user/repo.git)."
                            }
                        },
                        "required": ["repo_url"]
                    }
                }
            ],
            "resources": [],
            "prompts": []
        })

    @mcp.custom_route("/sse", methods=["POST"])
    async def sse_post_handler(request):
        """
        Smithery's automated scanner POSTs MCP JSON-RPC messages directly to /sse
        rather than establishing a proper SSE session first. It calls, in order:
          initialize → tools/list → resources/list → prompts/list

        The original handler returned the same mock initialization payload for ALL
        methods, so tools/list received an init response with no `tools` array,
        causing Smithery's Zod validator to fail with:
          "Invalid input: expected array, received undefined"

        This handler dispatches on `method` and returns the correct MCP response
        for each lifecycle call. Proper SSE sessions from real clients (Claude, etc.)
        always use GET /sse + the message endpoint provided in the SSE stream, so
        this handler does not interfere with live tool calls.
        """
        try:
            body = await request.json()
            req_id = body.get("id", 1)
            method = body.get("method", "")
        except Exception:
            req_id = 1
            method = ""

        _server_info = {
            "name": "sanjeev1412-official/repolens-mcp",
            "version": "1.0.0",
        }

        # ── initialize ────────────────────────────────────────────────────────
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools":     {},
                        "resources": {},
                        "prompts":   {},
                    },
                    "serverInfo": _server_info,
                },
            })

        # ── tools/list ────────────────────────────────────────────────────────
        if method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "search_codebase",
                            "description": (
                                "Search the indexed repository codebase using semantic "
                                "vector similarity. Returns ranked code chunks (functions, "
                                "classes, or module-level code) with similarity scores, "
                                "file paths, line ranges, and source snippets."
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Natural-language description or code snippet to search for.",
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "Maximum number of results to return (default 5).",
                                        "default": 5,
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "read_file_content",
                            "description": (
                                "Read a slice of a source file from the indexed repository. "
                                "Returns the requested line range with 1-indexed line numbers."
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path relative to the repository root (e.g. src/utils.py).",
                                    },
                                    "start_line": {
                                        "type": "integer",
                                        "description": "1-indexed line to begin reading from (default 1).",
                                        "default": 1,
                                    },
                                    "end_line": {
                                        "type": "integer",
                                        "description": "1-indexed line to stop reading at (default 200).",
                                        "default": 200,
                                    },
                                },
                                "required": ["file_path"],
                            },
                        },
                        {
                            "name": "get_file_history",
                            "description": (
                                "Retrieve the 10 most-recent Git commits that touched a file, "
                                "including short hash, date, author, and commit message."
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path relative to the repository root (e.g. src/chunker.py).",
                                    },
                                },
                                "required": ["file_path"],
                            },
                        },
                        {
                            "name": "index_github_repo",
                            "description": (
                                "Clones a public GitHub repository in the background and indexes "
                                "code chunks into the RepoLens ChromaDB vector store. Returns "
                                "immediately. Call get_index_status to poll for completion."
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "repo_url": {
                                        "type": "string",
                                        "description": "The full Git clone URL (e.g., https://github.com/user/repo.git).",
                                    },
                                },
                                "required": ["repo_url"],
                            },
                        },
                        {
                            "name": "get_index_status",
                            "description": (
                                "Check the status of a background indexing job started by "
                                "index_github_repo. Returns running / done (with chunk count) / "
                                "error. When done, you may immediately call search_codebase."
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "repo_url": {
                                        "type": "string",
                                        "description": "The same Git clone URL passed to index_github_repo.",
                                    },
                                },
                                "required": ["repo_url"],
                            },
                        },
                    ]
                },
            })

        # ── tools/call ───────────────────────────────────────────────────────
        # Smithery's proxy routes ALL MCP messages (including live tool calls)
        # as plain POSTs to /sse instead of using the real SSE stream.
        # We dispatch directly to the Python tool functions here so Smithery
        # clients can execute tools without a persistent SSE connection.
        if method == "tools/call":
            params    = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            _tool_dispatch = {
                "search_codebase":   search_codebase,
                "read_file_content": read_file_content,
                "get_file_history":  get_file_history,
                "index_github_repo": index_github_repo,
                "get_index_status":  get_index_status,
            }

            if tool_name not in _tool_dispatch:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": f"Unknown tool: '{tool_name}'. "
                                   f"Available: {list(_tool_dispatch)}",
                    },
                })

            try:
                # Run synchronous tool in a thread pool so the async event loop
                # is never blocked (important for fast response on other requests).
                result_text = await asyncio.to_thread(
                    _tool_dispatch[tool_name], **arguments
                )
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(result_text)}],
                        "isError": False,
                    },
                })
            except TypeError as exc:
                # Bad arguments (missing required param, wrong type, etc.)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Invalid arguments for '{tool_name}': {exc}"}],
                        "isError": True,
                    },
                })
            except Exception as exc:  # noqa: BLE001
                logger.error("tools/call '%s' error: %s", tool_name, exc, exc_info=True)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error running '{tool_name}': {exc}"}],
                        "isError": True,
                    },
                })

        # ── resources/list ───────────────────────────────────────────────────
        if method == "resources/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"resources": []},
            })

        # ── prompts/list ─────────────────────────────────────────────────────
        if method == "prompts/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"prompts": []},
            })

        # ── notifications/* (fire-and-forget, no response required) ──────────
        if method.startswith("notifications/"):
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

        # ── unknown method ────────────────────────────────────────────────────
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' is not supported on this endpoint.",
            },
        })

    @mcp.custom_route("/", methods=["GET"])
    async def root_handler(request):
        """
        Friendly root endpoint so users don't see 'Not Found' when visiting the base URL in a browser.
        """
        return JSONResponse({
            "status": "online",
            "message": "RepoLens MCP Server is running.",
            "sse_endpoint": "/sse",
            "version": "1.0.0"
        })

    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse":
        port = int(os.environ.get("PORT", 8000))
        logger.info("Starting FastMCP server with SSE transport on port %d", port)
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        logger.info("Starting FastMCP server with stdio transport")
        mcp.run(transport="stdio")
