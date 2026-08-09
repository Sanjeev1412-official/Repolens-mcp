"""
test_live_mcp.py – Phase 4: Live MCP Client Integration Tests
=============================================================
Strategy
--------
FastMCP 3.x ``Client`` accepts a ``FastMCP`` instance directly as its
transport argument, running everything **in-process** with no subprocess
or stdio pipe.  This lets us exercise the full MCP request/response cycle
(tool listing, argument serialisation, content extraction) against a real
server without spawning an OS process.

Fixtures
--------
* ``mock_repo``       – tiny on-disk repo (module scope, written once).
* ``indexed_server``  – the ``src.server.mcp`` FastMCP instance with its
  ``_indexer`` already pointed at the tmp repo, returned ready to accept
  Client connections.

Test classes
------------
* ``TestLiveMCPToolDiscovery``  – protocol-level tool registration checks.
* ``TestLiveSearchCodebase``    – end-to-end semantic search via MCP.
* ``TestLiveReadFileContent``   – file-read slice round-trip via MCP.
* ``TestLiveGetFileHistory``    – git-history round-trip via MCP.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_from_result(result) -> str:
    """
    Extract the plain-text string from a ``CallToolResult``.

    FastMCP 3.x returns a ``CallToolResult`` dataclass whose ``.data``
    field holds the deserialised tool return value.  When the tool returns
    a plain ``str``, ``.data`` is that string.  The ``.content`` list
    contains ``TextContent`` blocks as a fallback.
    """
    # Prefer the deserialised .data field (str for our tools).
    if isinstance(result.data, str):
        return result.data

    # Fall back to concatenating TextContent blocks from .content.
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Create a minimal repository tree with Python sources and a plain-text file.

    Layout::

        live_repo/
            math_ops.py   – defines add(), multiply()
            shapes.py     – defines Rectangle class
            readme.txt    – plain-text file
    """
    root = tmp_path_factory.mktemp("live_repo")

    (root / "math_ops.py").write_text(
        textwrap.dedent(
            """\
            \"\"\"Basic arithmetic operations.\"\"\"


            def add(x: float, y: float) -> float:
                \"\"\"Return the sum of x and y.\"\"\"
                return x + y


            def multiply(x: float, y: float) -> float:
                \"\"\"Return the product of x and y.\"\"\"
                return x * y


            def divide(x: float, y: float) -> float:
                \"\"\"Return x divided by y; raises ZeroDivisionError if y is 0.\"\"\"
                if y == 0:
                    raise ZeroDivisionError("Cannot divide by zero.")
                return x / y
            """
        ),
        encoding="utf-8",
    )

    (root / "shapes.py").write_text(
        textwrap.dedent(
            """\
            \"\"\"Geometric shape utilities.\"\"\"


            class Rectangle:
                \"\"\"Represents a 2-D rectangle.\"\"\"

                def __init__(self, width: float, height: float) -> None:
                    self.width = width
                    self.height = height

                def area(self) -> float:
                    \"\"\"Return the area of the rectangle.\"\"\"
                    return self.width * self.height

                def perimeter(self) -> float:
                    \"\"\"Return the perimeter of the rectangle.\"\"\"
                    return 2 * (self.width + self.height)
            """
        ),
        encoding="utf-8",
    )

    (root / "readme.txt").write_text(
        "RepoLens live-test sample repository.\nLine two.\nLine three.\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture(scope="module")
def indexed_server(mock_repo: Path, tmp_path_factory: pytest.TempPathFactory):
    """
    Return the ``src.server.mcp`` FastMCP instance with its module-level
    ``_indexer`` set up against the tmp repo.

    Because ``src.server`` is a module singleton, we patch its ``_indexer``
    and ``REPO_PATH`` for the duration of the test session and restore them
    afterwards.
    """
    import src.server as srv
    from src.indexer import RepoLensIndexer

    chroma_dir = tmp_path_factory.mktemp("live_chroma")
    idx = RepoLensIndexer(
        chroma_path=str(chroma_dir),
        collection_name="live_test_collection",
    )
    idx.index_repository(str(mock_repo))

    # Patch module globals.
    original_indexer = srv._indexer
    original_repo = srv.REPO_PATH
    srv._indexer = idx
    srv.REPO_PATH = str(mock_repo)

    yield srv.mcp  # hand the FastMCP instance to tests

    # Restore originals after the module-scope session ends.
    srv._indexer = original_indexer
    srv.REPO_PATH = original_repo


# ---------------------------------------------------------------------------
# Async helper – run a single async coroutine in a test
# ---------------------------------------------------------------------------


def _run(coro):
    """Run *coro* on a fresh event loop (Python 3.12+ compatible)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# TestLiveMCPToolDiscovery
# ---------------------------------------------------------------------------


class TestLiveMCPToolDiscovery:
    """Verify tool registration via the MCP protocol layer."""

    def test_list_tools_returns_three_tools(self, indexed_server) -> None:
        from fastmcp import Client

        async def _check():
            async with Client(indexed_server) as client:
                tools = await client.list_tools()
                return {t.name for t in tools}

        names = _run(_check())
        assert "search_codebase" in names
        assert "read_file_content" in names
        assert "get_file_history" in names

    def test_search_codebase_tool_has_description(self, indexed_server) -> None:
        from fastmcp import Client

        async def _check():
            async with Client(indexed_server) as client:
                tools = await client.list_tools()
                return {t.name: t for t in tools}

        tools = _run(_check())
        sc = tools.get("search_codebase")
        assert sc is not None
        assert sc.description and len(sc.description) > 10

    def test_read_file_content_tool_has_description(self, indexed_server) -> None:
        from fastmcp import Client

        async def _check():
            async with Client(indexed_server) as client:
                tools = await client.list_tools()
                return {t.name: t for t in tools}

        tools = _run(_check())
        rfc = tools.get("read_file_content")
        assert rfc is not None
        assert rfc.description and len(rfc.description) > 10

    def test_get_file_history_tool_has_description(self, indexed_server) -> None:
        from fastmcp import Client

        async def _check():
            async with Client(indexed_server) as client:
                tools = await client.list_tools()
                return {t.name: t for t in tools}

        tools = _run(_check())
        gfh = tools.get("get_file_history")
        assert gfh is not None
        assert gfh.description and len(gfh.description) > 10

    def test_tools_have_input_schemas(self, indexed_server) -> None:
        from fastmcp import Client

        async def _check():
            async with Client(indexed_server) as client:
                tools = await client.list_tools()
                return tools

        tools = _run(_check())
        for tool in tools:
            assert tool.inputSchema is not None, f"Tool '{tool.name}' has no inputSchema"


# ---------------------------------------------------------------------------
# TestLiveSearchCodebase
# ---------------------------------------------------------------------------


class TestLiveSearchCodebase:
    """Full MCP round-trip for the search_codebase tool."""

    def _search(self, server, query: str, top_k: int = 5) -> str:
        from fastmcp import Client

        async def _call():
            async with Client(server) as client:
                result = await client.call_tool(
                    "search_codebase",
                    {"query": query, "top_k": top_k},
                )
                return _text_from_result(result)

        return _run(_call())

    def test_returns_non_empty_string(self, indexed_server) -> None:
        out = self._search(indexed_server, "arithmetic function")
        assert isinstance(out, str) and out.strip()

    def test_not_error_payload_for_valid_query(self, indexed_server) -> None:
        out = self._search(indexed_server, "add two numbers")
        assert not out.startswith("Error")

    def test_math_ops_surfaces_for_arithmetic_query(self, indexed_server) -> None:
        out = self._search(indexed_server, "function that adds two numbers", top_k=5)
        assert "math_ops" in out

    def test_shapes_surfaces_for_geometry_query(self, indexed_server) -> None:
        out = self._search(indexed_server, "class that represents a rectangle", top_k=5)
        assert "shapes" in out

    def test_result_contains_score_label(self, indexed_server) -> None:
        out = self._search(indexed_server, "multiplication")
        assert "Score" in out

    def test_result_contains_file_label(self, indexed_server) -> None:
        out = self._search(indexed_server, "area of rectangle")
        assert "File" in out

    def test_result_contains_symbol_label(self, indexed_server) -> None:
        out = self._search(indexed_server, "divide numbers")
        assert "Symbol" in out

    def test_empty_query_returns_error_string(self, indexed_server) -> None:
        out = self._search(indexed_server, "")
        assert "Error" in out

    def test_top_k_one_returns_single_result(self, indexed_server) -> None:
        out = self._search(indexed_server, "function", top_k=1)
        assert out.count("── Result") == 1

    def test_top_k_two_returns_at_most_two_results(self, indexed_server) -> None:
        out = self._search(indexed_server, "class method", top_k=2)
        assert out.count("── Result") <= 2

    def test_is_error_false_for_valid_query(self, indexed_server) -> None:
        from fastmcp import Client

        async def _call():
            async with Client(indexed_server) as client:
                result = await client.call_tool(
                    "search_codebase",
                    {"query": "geometric shape", "top_k": 3},
                )
                return result.is_error

        assert _run(_call()) is False


# ---------------------------------------------------------------------------
# TestLiveReadFileContent
# ---------------------------------------------------------------------------


class TestLiveReadFileContent:
    """Full MCP round-trip for the read_file_content tool."""

    def _read(
        self,
        server,
        file_path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        from fastmcp import Client

        async def _call():
            async with Client(server) as client:
                result = await client.call_tool(
                    "read_file_content",
                    {
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
                return _text_from_result(result)

        return _run(_call())

    def test_reads_python_file_successfully(self, indexed_server) -> None:
        out = self._read(indexed_server, "math_ops.py")
        assert "def add" in out
        assert "def multiply" in out

    def test_returns_non_empty_string(self, indexed_server) -> None:
        out = self._read(indexed_server, "shapes.py")
        assert isinstance(out, str) and out.strip()

    def test_line_numbers_prepended(self, indexed_server) -> None:
        out = self._read(indexed_server, "math_ops.py", 1, 5)
        assert " | " in out

    def test_header_contains_filename(self, indexed_server) -> None:
        out = self._read(indexed_server, "readme.txt")
        assert "readme.txt" in out

    def test_start_line_respected(self, indexed_server) -> None:
        # Starting at line 8 skips the module docstring and first blank lines.
        out = self._read(indexed_server, "math_ops.py", 8, 15)
        assert "Basic arithmetic operations" not in out

    def test_end_line_respected(self, indexed_server) -> None:
        # Only first 2 lines – should not contain def multiply.
        out = self._read(indexed_server, "math_ops.py", 1, 2)
        assert "multiply" not in out

    def test_missing_file_returns_error_string(self, indexed_server) -> None:
        out = self._read(indexed_server, "no_such_file.py")
        assert "Error" in out

    def test_path_traversal_returns_error_string(self, indexed_server) -> None:
        out = self._read(indexed_server, "../../etc/passwd")
        assert "Error" in out

    def test_empty_path_returns_error_string(self, indexed_server) -> None:
        out = self._read(indexed_server, "")
        assert "Error" in out

    def test_inverted_range_returns_error_string(self, indexed_server) -> None:
        out = self._read(indexed_server, "math_ops.py", 50, 5)
        assert "Error" in out

    def test_reads_plain_text_file(self, indexed_server) -> None:
        out = self._read(indexed_server, "readme.txt", 1, 10)
        assert "RepoLens" in out

    def test_is_error_false_for_valid_file(self, indexed_server) -> None:
        from fastmcp import Client

        async def _call():
            async with Client(indexed_server) as client:
                result = await client.call_tool(
                    "read_file_content",
                    {"file_path": "shapes.py", "start_line": 1, "end_line": 10},
                )
                return result.is_error

        assert _run(_call()) is False


# ---------------------------------------------------------------------------
# TestLiveGetFileHistory
# ---------------------------------------------------------------------------


class TestLiveGetFileHistory:
    """Full MCP round-trip for the get_file_history tool."""

    def _history(self, server, file_path: str) -> str:
        from fastmcp import Client

        async def _call():
            async with Client(server) as client:
                result = await client.call_tool(
                    "get_file_history",
                    {"file_path": file_path},
                )
                return _text_from_result(result)

        return _run(_call())

    def test_returns_non_empty_string(self, indexed_server) -> None:
        out = self._history(indexed_server, "math_ops.py")
        assert isinstance(out, str) and out.strip()

    def test_no_git_repo_returns_informational_message(self, indexed_server) -> None:
        """mock_repo has no .git dir – should get an informational string, not an exception."""
        out = self._history(indexed_server, "math_ops.py")
        lower = out.lower()
        assert "no git history" in lower or "not a git" in lower or "untracked" in lower, (
            f"Unexpected response: {out}"
        )

    def test_empty_file_path_returns_error_string(self, indexed_server) -> None:
        out = self._history(indexed_server, "")
        assert "Error" in out

    def test_nonexistent_file_does_not_raise(self, indexed_server) -> None:
        out = self._history(indexed_server, "does_not_exist.py")
        assert isinstance(out, str)

    def test_is_error_false_for_valid_call(self, indexed_server) -> None:
        from fastmcp import Client

        async def _call():
            async with Client(indexed_server) as client:
                result = await client.call_tool(
                    "get_file_history",
                    {"file_path": "shapes.py"},
                )
                return result.is_error

        assert _run(_call()) is False
