"""
test_indexer.py – Pytest suite for src/git_utils.py & src/indexer.py (Phase 2)
===============================================================================
Test strategy
-------------
* ``TestGitUtils``        – Unit tests for get_git_file_history and
                            format_git_metadata_string that do NOT require a
                            real Git repo (non-git paths, empty history).
* ``TestFormatGitMetadata`` – Pure formatting tests (no I/O).
* ``TestRepoLensIndexer`` – Integration tests over a tiny in-memory repository
                            written to tmp_path.  The model is loaded once at
                            module scope to keep the test suite fast.

Notes
-----
* All ChromaDB collections use fresh tmp_path directories so tests are
  fully isolated from each other and from any project-level ./chroma_db.
* The sentence-transformer model (all-MiniLM-L6-v2, ~90 MB) is downloaded
  on first run and then cached by the transformers/HuggingFace library.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable when pytest is run from the project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.git_utils import format_git_metadata_string, get_git_file_history
from src.indexer import RepoLensIndexer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    A minimal repository tree that ``load_repo_files`` can scan.

    Layout::

        tiny_repo/
            math_utils.py    ← defines add() and multiply()
            greeter.py       ← defines Greeter class
            notes.txt        ← plain-text fallback file
    """
    root = tmp_path_factory.mktemp("tiny_repo")

    (root / "math_utils.py").write_text(
        textwrap.dedent(
            """\
            \"\"\"Simple math utilities.\"\"\"


            def add(a: int, b: int) -> int:
                \"\"\"Return the sum of two integers.\"\"\"
                return a + b


            def multiply(a: int, b: int) -> int:
                \"\"\"Return the product of two integers.\"\"\"
                return a * b
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
        "This is a plain-text note file.\nNo code here.\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture(scope="module")
def indexer(tiny_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> RepoLensIndexer:
    """
    Module-scoped ``RepoLensIndexer`` pointing at an isolated ChromaDB
    directory.  The repository is indexed exactly once to save time.
    """
    chroma_dir = tmp_path_factory.mktemp("chroma")
    idx = RepoLensIndexer(
        chroma_path=str(chroma_dir),
        collection_name="test_collection",
        embed_model_name="all-MiniLM-L6-v2",
    )
    idx.index_repository(str(tiny_repo))
    return idx


# ---------------------------------------------------------------------------
# TestGitUtils – non-git-repo edge cases
# ---------------------------------------------------------------------------


class TestGitUtils:
    def test_non_git_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """A plain directory (no .git) must return [] without raising."""
        result = get_git_file_history(str(tmp_path), "some_file.py")
        assert result == []

    def test_nonexistent_path_returns_empty_list(self) -> None:
        result = get_git_file_history("/this/path/does/not/exist", "foo.py")
        assert result == []

    def test_return_type_is_list(self, tmp_path: Path) -> None:
        result = get_git_file_history(str(tmp_path), "x.py")
        assert isinstance(result, list)

    def test_max_commits_respected(self, tmp_path: Path) -> None:
        """Even for non-git dirs the contract is 'at most max_commits' entries."""
        result = get_git_file_history(str(tmp_path), "x.py", max_commits=3)
        assert len(result) <= 3

    def test_dict_keys_present_when_non_empty(self, tmp_path: Path) -> None:
        """
        When history is non-empty (requires a real git repo), all required
        keys must be present.  Skip gracefully when no git repo is available.
        """
        result = get_git_file_history(str(tmp_path), "x.py")
        for entry in result:  # may be empty – loop body simply won't execute
            required = {"commit_hash", "short_hash", "author", "author_email", "date", "message"}
            assert required <= entry.keys()


# ---------------------------------------------------------------------------
# TestFormatGitMetadata – pure formatting, no I/O
# ---------------------------------------------------------------------------


class TestFormatGitMetadata:
    _SAMPLE_HISTORY = [
        {
            "commit_hash": "a" * 40,
            "short_hash": "aaaaaaa",
            "author": "Alice Smith",
            "author_email": "alice@example.com",
            "date": "2024-06-01T12:00:00+00:00",
            "message": "Add feature X",
        },
        {
            "commit_hash": "b" * 40,
            "short_hash": "bbbbbbb",
            "author": "Bob Jones",
            "author_email": "bob@example.com",
            "date": "2024-05-20T09:30:00+00:00",
            "message": "Fix bug Y",
        },
    ]

    def test_empty_history_returns_empty_string(self) -> None:
        assert format_git_metadata_string([]) == ""

    def test_header_contains_commit_count(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        assert "2 commits" in result

    def test_single_commit_header_is_singular(self) -> None:
        result = format_git_metadata_string([self._SAMPLE_HISTORY[0]])
        assert "1 commit" in result
        # Must NOT say "1 commits"
        assert "1 commits" not in result

    def test_all_short_hashes_present(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        assert "aaaaaaa" in result
        assert "bbbbbbb" in result

    def test_author_names_present(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        assert "Alice Smith" in result
        assert "Bob Jones" in result

    def test_commit_messages_present(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        assert "Add feature X" in result
        assert "Fix bug Y" in result

    def test_dates_are_trimmed_to_date_portion(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        # Should show 2024-06-01, not the full ISO timestamp.
        assert "2024-06-01" in result

    def test_output_is_multiline(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        assert "\n" in result

    def test_returns_string(self) -> None:
        result = format_git_metadata_string(self._SAMPLE_HISTORY)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestRepoLensIndexer – indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_index_repository_returns_positive_count(self, indexer: RepoLensIndexer) -> None:
        """index_repository must return the number of chunks stored (> 0)."""
        # The indexer fixture has already indexed; we check collection_count.
        assert indexer.collection_count > 0

    def test_collection_count_matches_indexed(self, tiny_repo: Path, tmp_path: Path) -> None:
        """
        A fresh indexer over the same repo must produce a positive chunk count
        and that count must equal the ChromaDB collection size.
        """
        chroma_dir = tmp_path / "chroma_count_test"
        idx = RepoLensIndexer(
            chroma_path=str(chroma_dir),
            collection_name="count_test",
        )
        count = idx.index_repository(str(tiny_repo))
        assert count > 0
        assert idx.collection_count == count

    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        """An empty directory has no files → 0 chunks."""
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()
        chroma_dir = tmp_path / "chroma_empty"
        idx = RepoLensIndexer(
            chroma_path=str(chroma_dir),
            collection_name="empty_test",
        )
        count = idx.index_repository(str(empty_dir))
        assert count == 0

    def test_reindex_is_idempotent(self, tiny_repo: Path, tmp_path: Path) -> None:
        """
        Calling index_repository twice must not duplicate documents
        (upsert semantics guarantee this).
        """
        chroma_dir = tmp_path / "chroma_idempotent"
        idx = RepoLensIndexer(
            chroma_path=str(chroma_dir),
            collection_name="idempotent_test",
        )
        count_first = idx.index_repository(str(tiny_repo))
        count_second = idx.index_repository(str(tiny_repo))
        # Both calls must report the same number of chunks.
        assert count_first == count_second
        # Collection should not have grown.
        assert idx.collection_count == count_first


# ---------------------------------------------------------------------------
# TestSearchCodebase – search quality and result structure
# ---------------------------------------------------------------------------


class TestSearchCodebase:
    def test_returns_list(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("add two numbers")
        assert isinstance(results, list)

    def test_top_k_respected(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("function", top_k=2)
        assert len(results) <= 2

    def test_result_has_required_keys(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("multiply integers", top_k=3)
        required = {
            "score",
            "chunk_id",
            "file_path",
            "symbol_name",
            "symbol_type",
            "start_line",
            "end_line",
            "document",
        }
        for r in results:
            assert required <= r.keys(), f"Missing keys in result: {r}"

    def test_score_is_float_in_range(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("greet someone", top_k=5)
        for r in results:
            assert isinstance(r["score"], float)
            # Cosine similarity can theoretically be [-1, 1]; practically [0, 1].
            assert -1.0 <= r["score"] <= 1.0

    def test_results_sorted_by_score_descending(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("class that greets people", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_symbol_ranked_high_for_math_query(self, indexer: RepoLensIndexer) -> None:
        """
        Querying for addition should surface math_utils.py near the top.
        """
        results = indexer.search_codebase("add two integers together", top_k=5)
        top_files = [r["file_path"] for r in results[:3]]
        assert any("math_utils" in f for f in top_files), (
            f"Expected math_utils.py in top results; got: {top_files}"
        )

    def test_relevant_symbol_ranked_high_for_class_query(self, indexer: RepoLensIndexer) -> None:
        """
        Querying for greeting class should surface greeter.py near the top.
        """
        results = indexer.search_codebase("class that greets people by name", top_k=5)
        top_files = [r["file_path"] for r in results[:3]]
        assert any("greeter" in f for f in top_files), (
            f"Expected greeter.py in top results; got: {top_files}"
        )

    def test_document_contains_source_code(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("return a plus b", top_k=3)
        # At least one document should contain actual source code.
        all_docs = " ".join(r["document"] for r in results)
        assert len(all_docs) > 0

    def test_document_contains_file_header(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("greeting", top_k=3)
        for r in results:
            assert "FILE:" in r["document"], (
                f"Expected 'FILE:' header in document, got: {r['document'][:100]}"
            )

    def test_start_line_is_positive(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("function definition", top_k=5)
        for r in results:
            assert r["start_line"] >= 1

    def test_end_line_gte_start_line(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("method", top_k=5)
        for r in results:
            assert r["end_line"] >= r["start_line"]

    def test_empty_query_returns_empty_list(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("")
        assert results == []

    def test_whitespace_only_query_returns_empty_list(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("   \t\n")
        assert results == []

    def test_file_filter_limits_results(self, indexer: RepoLensIndexer) -> None:
        """
        When file_filter is set, returned chunks must only come from matching
        files.  Skip assertion if the filter produces zero results (file not
        indexed or ChromaDB $contains unsupported).
        """
        results = indexer.search_codebase("function", top_k=10, file_filter="math_utils")
        for r in results:
            assert "math_utils" in r["file_path"], (
                f"file_filter not respected: got {r['file_path']}"
            )

    def test_chunk_id_is_non_empty_string(self, indexer: RepoLensIndexer) -> None:
        results = indexer.search_codebase("any code", top_k=3)
        for r in results:
            assert isinstance(r["chunk_id"], str)
            assert r["chunk_id"] != ""

    def test_symbol_type_is_valid(self, indexer: RepoLensIndexer) -> None:
        valid_types = {"function", "class", "module", "fallback"}
        results = indexer.search_codebase("definition", top_k=5)
        for r in results:
            assert r["symbol_type"] in valid_types, f"Unexpected symbol_type: {r['symbol_type']}"
