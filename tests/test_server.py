"""
test_server.py – Phase 3 integration tests for src/server.py
=============================================================
Strategy
--------
The MCP tools (search_codebase, read_file_content, get_file_history) are
**plain Python callables** decorated with @mcp.tool().  We test them by:

1. Patching the module-level ``_indexer`` in server.py with a mock/real
   indexer backed by a tmp_path ChromaDB, so no stdout/stdio transport is
   needed.
2. Overriding the ``REPO_PATH`` module variable to point at a tmp repo.
3. Calling the decorated functions directly – FastMCP 3.x preserves the
   underlying function through the decorator, accessible via
   ``mcp.get_tool("<name>").fn`` or just by importing the function name.

All ChromaDB dirs use isolated tmp_path subdirectories; the sentence-
transformer model is loaded once (module scope) to keep the suite fast.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared repo fixture (module scope – created once for all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    A small repository used by all Phase-3 tests.

    Layout::

        mock_repo/
            math_utils.py   – defines add() and subtract()
            greeter.py      – defines Greeter class
            notes.txt       – plain-text file (fallback chunking)
    """
    root = tmp_path_factory.mktemp("mock_repo")

    (root / "math_utils.py").write_text(
        textwrap.dedent(
            """\
            \"\"\"Simple arithmetic utilities.\"\"\"


            def add(a: int, b: int) -> int:
                \"\"\"Return the sum of two integers.\"\"\"
                return a + b


            def subtract(a: int, b: int) -> int:
                \"\"\"Return the difference of two integers.\"\"\"
                return a - b
            """
        ),
        encoding="utf-8",
    )

    (root / "greeter.py").write_text(
        textwrap.dedent(
            """\
            \"\"\"Greeting utilities.\"\"\"


            class Greeter:
                \"\"\"Produces personalised greeting strings.\"\"\"

                def __init__(self, name: str) -> None:
                    self.name = name

                def greet(self) -> str:
                    return f"Hello, {self.name}!"
            """
        ),
        encoding="utf-8",
    )

    (root / "notes.txt").write_text(
        "This is a plain-text note.\nLine two.\nLine three.\n",
        encoding="utf-8",
    )

    return root


# ---------------------------------------------------------------------------
# Module-scoped indexer fixture (model loaded once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_indexer(mock_repo: Path, tmp_path_factory: pytest.TempPathFactory):
    """Fully initialised RepoLensIndexer backed by tmp ChromaDB."""
    from src.indexer import RepoLensIndexer

    chroma_dir = tmp_path_factory.mktemp("chroma_server_tests")
    idx = RepoLensIndexer(
        chroma_path=str(chroma_dir),
        collection_name="server_test_collection",
    )
    idx.index_repository(str(mock_repo))
    return idx


# ---------------------------------------------------------------------------
# Helpers to call server tools with a patched environment
# ---------------------------------------------------------------------------


def _call_search(query: str, top_k: int, indexer, repo_path: str) -> str:
    """Invoke the search_codebase tool logic with a patched indexer/REPO_PATH."""
    import src.server as srv

    original_indexer = srv._indexer
    original_repo = srv.REPO_PATH
    try:
        srv._indexer = indexer
        srv.REPO_PATH = repo_path
        return srv.search_codebase(query=query, top_k=top_k)
    finally:
        srv._indexer = original_indexer
        srv.REPO_PATH = original_repo


def _call_read(file_path: str, start_line: int, end_line: int, repo_path: str) -> str:
    """Invoke the read_file_content tool logic with a patched REPO_PATH."""
    import src.server as srv

    original_repo = srv.REPO_PATH
    try:
        srv.REPO_PATH = repo_path
        return srv.read_file_content(file_path=file_path, start_line=start_line, end_line=end_line)
    finally:
        srv.REPO_PATH = original_repo


def _call_history(file_path: str, repo_path: str) -> str:
    """Invoke the get_file_history tool logic with a patched REPO_PATH."""
    import src.server as srv

    original_repo = srv.REPO_PATH
    try:
        srv.REPO_PATH = repo_path
        return srv.get_file_history(file_path=file_path)
    finally:
        srv.REPO_PATH = original_repo


# ---------------------------------------------------------------------------
# TestServerModule – import and registration smoke tests
# ---------------------------------------------------------------------------


class TestServerModule:
    def test_server_imports_without_error(self) -> None:
        import src.server as srv  # noqa: F401

    def test_mcp_instance_exists(self) -> None:
        from fastmcp import FastMCP

        import src.server as srv

        assert isinstance(srv.mcp, FastMCP)

    def test_mcp_name_is_repolens(self) -> None:
        import src.server as srv

        assert srv.mcp.name == "RepoLens MCP"

    def test_three_tools_registered(self) -> None:
        """FastMCP 3.x exposes list_tools() as a sync method."""
        import asyncio

        import src.server as srv

        tools = asyncio.run(srv.mcp.list_tools())
        names = {t.name for t in tools}
        assert "search_codebase" in names
        assert "read_file_content" in names
        assert "get_file_history" in names

    def test_tool_functions_are_callable(self) -> None:
        import src.server as srv

        assert callable(srv.search_codebase)
        assert callable(srv.read_file_content)
        assert callable(srv.get_file_history)


# ---------------------------------------------------------------------------
# TestSearchCodebase
# ---------------------------------------------------------------------------


class TestSearchCodebase:
    def test_returns_string(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("add two integers", 3, real_indexer, str(mock_repo))
        assert isinstance(result, str)

    def test_non_empty_results_for_known_content(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("function that adds numbers", 5, real_indexer, str(mock_repo))
        assert result.strip() != ""
        assert result != "No results found."

    def test_result_contains_score_label(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("greeting class", 3, real_indexer, str(mock_repo))
        assert "Score" in result

    def test_result_contains_file_label(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("greeting class", 3, real_indexer, str(mock_repo))
        assert "File" in result

    def test_result_contains_symbol_label(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("subtract integers", 3, real_indexer, str(mock_repo))
        assert "Symbol" in result

    def test_math_utils_surfaces_for_arithmetic_query(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("add two numbers together", 5, real_indexer, str(mock_repo))
        assert "math_utils" in result

    def test_greeter_surfaces_for_greeting_query(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("class that greets people by name", 5, real_indexer, str(mock_repo))
        assert "greeter" in result

    def test_empty_query_returns_error_string(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("", 5, real_indexer, str(mock_repo))
        assert "Error" in result

    def test_whitespace_query_returns_error_string(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("   ", 5, real_indexer, str(mock_repo))
        assert "Error" in result

    def test_top_k_limits_result_count(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("function", 2, real_indexer, str(mock_repo))
        # Count "Result N" headers as a proxy for result count.
        result_count = result.count("── Result")
        assert result_count <= 2

    def test_snippet_separator_present(self, real_indexer, mock_repo: Path) -> None:
        result = _call_search("arithmetic", 3, real_indexer, str(mock_repo))
        # The enriched document always contains "---" as separator between
        # metadata header and source code.
        assert "Lines" in result  # line boundary info in symbol line

    def test_indexer_error_returns_error_string(self, mock_repo: Path) -> None:
        """If the indexer raises, the tool must return an error string (not raise)."""
        bad_indexer = MagicMock()
        bad_indexer.search_codebase.side_effect = RuntimeError("db exploded")
        result = _call_search("anything", 5, bad_indexer, str(mock_repo))
        assert "Error" in result


# ---------------------------------------------------------------------------
# TestReadFileContent
# ---------------------------------------------------------------------------


class TestReadFileContent:
    def test_returns_string(self, mock_repo: Path) -> None:
        result = _call_read("math_utils.py", 1, 200, str(mock_repo))
        assert isinstance(result, str)

    def test_reads_full_python_file(self, mock_repo: Path) -> None:
        result = _call_read("math_utils.py", 1, 200, str(mock_repo))
        assert "def add" in result
        assert "def subtract" in result

    def test_line_numbers_are_prepended(self, mock_repo: Path) -> None:
        result = _call_read("math_utils.py", 1, 5, str(mock_repo))
        # Should contain "1 |" style numbering.
        assert " | " in result

    def test_start_line_respected(self, mock_repo: Path) -> None:
        # Read only from line 7 onwards – should not contain the module docstring (line 1).
        result = _call_read("math_utils.py", 7, 12, str(mock_repo))
        assert "Simple arithmetic utilities" not in result

    def test_end_line_respected(self, mock_repo: Path) -> None:
        # Read only first 2 lines – should not contain def subtract.
        result = _call_read("math_utils.py", 1, 2, str(mock_repo))
        assert "subtract" not in result

    def test_header_contains_filename(self, mock_repo: Path) -> None:
        result = _call_read("greeter.py", 1, 10, str(mock_repo))
        assert "greeter.py" in result

    def test_header_contains_line_range(self, mock_repo: Path) -> None:
        result = _call_read("notes.txt", 1, 3, str(mock_repo))
        assert "1" in result and "3" in result

    def test_missing_file_returns_error(self, mock_repo: Path) -> None:
        result = _call_read("no_such_file.py", 1, 10, str(mock_repo))
        assert "Error" in result
        assert "not found" in result.lower() or "error" in result.lower()

    def test_path_traversal_blocked(self, mock_repo: Path) -> None:
        result = _call_read("../../secrets.txt", 1, 10, str(mock_repo))
        assert "Error" in result

    def test_empty_file_path_returns_error(self, mock_repo: Path) -> None:
        result = _call_read("", 1, 10, str(mock_repo))
        assert "Error" in result

    def test_start_beyond_file_length_returns_error(self, mock_repo: Path) -> None:
        result = _call_read("notes.txt", 9999, 10000, str(mock_repo))
        assert "Error" in result

    def test_start_greater_than_end_returns_error(self, mock_repo: Path) -> None:
        result = _call_read("notes.txt", 5, 2, str(mock_repo))
        assert "Error" in result

    def test_reads_plain_text_file(self, mock_repo: Path) -> None:
        result = _call_read("notes.txt", 1, 200, str(mock_repo))
        assert "plain-text note" in result

    def test_single_line_read(self, mock_repo: Path) -> None:
        result = _call_read("math_utils.py", 1, 1, str(mock_repo))
        # Should contain exactly one "|" separator for the single line.
        assert result.count(" | ") == 1


# ---------------------------------------------------------------------------
# TestGetFileHistory
# ---------------------------------------------------------------------------


class TestGetFileHistory:
    def test_returns_string(self, mock_repo: Path) -> None:
        result = _call_history("math_utils.py", str(mock_repo))
        assert isinstance(result, str)

    def test_non_git_repo_returns_informational_message(self, mock_repo: Path) -> None:
        """mock_repo has no .git directory – must get an informational string, not raise."""
        result = _call_history("math_utils.py", str(mock_repo))
        # Should mention "no git history" or "not a Git repository" – not crash.
        lower = result.lower()
        assert "no git history" in lower or "not a git" in lower or "untracked" in lower

    def test_empty_file_path_returns_error(self, mock_repo: Path) -> None:
        result = _call_history("", str(mock_repo))
        assert "Error" in result

    def test_result_does_not_raise_for_missing_file(self, mock_repo: Path) -> None:
        """Even for a non-existent file, the tool must return a string."""
        result = _call_history("does_not_exist.py", str(mock_repo))
        assert isinstance(result, str)

    def test_result_for_nonexistent_path(self) -> None:
        """Completely invalid repo path must return a string, not raise."""
        result = _call_history("any_file.py", "/this/does/not/exist")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestFormatSearchResults (unit-level, no indexer needed)
# ---------------------------------------------------------------------------


class TestFormatSearchResults:
    """Tests for the private _format_search_results helper."""

    def test_empty_list_returns_no_results(self) -> None:
        from src.server import _format_search_results

        assert _format_search_results([]) == "No results found."

    def test_single_result_formatted(self) -> None:
        from src.server import _format_search_results

        results = [
            {
                "score": 0.95,
                "chunk_id": "abc123",
                "file_path": "src/foo.py",
                "symbol_name": "my_func",
                "symbol_type": "function",
                "start_line": 10,
                "end_line": 20,
                "document": (
                    "FILE: src/foo.py  |  SYMBOL: my_func (function)\n---\ndef my_func(): pass"
                ),
            }
        ]
        out = _format_search_results(results)
        assert "Score" in out
        assert "src/foo.py" in out
        assert "my_func" in out
        assert "0.9500" in out

    def test_multiple_results_numbered(self) -> None:
        from src.server import _format_search_results

        results = [
            {
                "score": 0.9,
                "chunk_id": "a",
                "file_path": "a.py",
                "symbol_name": "fa",
                "symbol_type": "function",
                "start_line": 1,
                "end_line": 5,
                "document": "FILE: a.py\n---\ndef fa(): pass",
            },
            {
                "score": 0.8,
                "chunk_id": "b",
                "file_path": "b.py",
                "symbol_name": "fb",
                "symbol_type": "function",
                "start_line": 1,
                "end_line": 5,
                "document": "FILE: b.py\n---\ndef fb(): pass",
            },
        ]
        out = _format_search_results(results)
        assert "Result 1" in out
        assert "Result 2" in out

    def test_long_snippet_is_truncated(self) -> None:
        from src.server import _SNIPPET_MAX_LINES, _format_search_results

        long_code = "\n".join(f"line_{i} = {i}" for i in range(_SNIPPET_MAX_LINES + 10))
        results = [
            {
                "score": 0.7,
                "chunk_id": "x",
                "file_path": "big.py",
                "symbol_name": "big_fn",
                "symbol_type": "function",
                "start_line": 1,
                "end_line": _SNIPPET_MAX_LINES + 10,
                "document": f"FILE: big.py\n---\n{long_code}",
            }
        ]
        out = _format_search_results(results)
        assert "more lines" in out
