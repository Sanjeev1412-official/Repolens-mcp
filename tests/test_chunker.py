"""
test_chunker.py – Pytest suite for src/chunker.py (Phase 1)
============================================================
Tests cover:
* CodeChunk model field population and ID determinism.
* load_repo_files – file scanning, binary exclusion, ignored-dir pruning.
* _chunk_with_lines – fallback line-windowed chunking correctness.
* chunk_file – AST path (Python) and fallback path.
* chunk_repo – end-to-end integration over a temporary repository tree.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Adjust import path so tests can find the src package regardless of how
# pytest is invoked (from repo root or from within tests/).
# ---------------------------------------------------------------------------
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import (
    CodeChunk,
    _chunk_with_lines,
    chunk_file,
    chunk_repo,
    load_repo_files,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PYTHON_SOURCE = textwrap.dedent(
    """\
    \"\"\"A sample module for testing the chunker.\"\"\"

    import os
    import sys

    MODULE_CONSTANT = 42


    class Greeter:
        \"\"\"Greets people.\"\"\"

        def __init__(self, name: str) -> None:
            self.name = name

        def greet(self) -> str:
            return f"Hello, {self.name}!"


    def add(a: int, b: int) -> int:
        \"\"\"Return the sum of a and b.\"\"\"
        return a + b


    def multiply(a: int, b: int) -> int:
        \"\"\"Return the product of a and b.\"\"\"
        return a * b


    if __name__ == "__main__":
        g = Greeter("World")
        print(g.greet())
    """
)


@pytest.fixture()
def sample_python_file_desc() -> dict:
    """Return a file descriptor dict for the in-memory sample Python source."""
    return {
        "abs_path": Path("/fake/repo/sample.py"),
        "rel_path": "sample.py",
        "extension": ".py",
        "content": SAMPLE_PYTHON_SOURCE,
    }


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """
    Create a small temporary repository tree:

        tmp_repo/
            src/
                main.py          # Python – AST parsed
                utils.py         # Python – AST parsed
            data/
                schema.json      # JSON – fallback
            .git/
                config           # should be ignored
            node_modules/
                lodash/index.js  # should be ignored
            README.md            # Markdown – fallback
    """
    # Python source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "main.py").write_text(
        textwrap.dedent(
            """\
            def run():
                print("running")

            class App:
                pass
            """
        ),
        encoding="utf-8",
    )

    (src_dir / "utils.py").write_text(
        textwrap.dedent(
            """\
            def helper(x):
                return x * 2
            """
        ),
        encoding="utf-8",
    )

    # JSON file (fallback)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "schema.json").write_text('{"version": 1}', encoding="utf-8")

    # Markdown (fallback)
    (tmp_path / "README.md").write_text("# My Project\n\nDescription here.", encoding="utf-8")

    # Files that must be ignored
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0", encoding="utf-8")

    nm_dir = tmp_path / "node_modules" / "lodash"
    nm_dir.mkdir(parents=True)
    (nm_dir / "index.js").write_text("module.exports = {};", encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# CodeChunk model tests
# ---------------------------------------------------------------------------


class TestCodeChunkModel:
    def test_fields_populated(self) -> None:
        chunk = CodeChunk(
            id="abc123",
            file_path="src/foo.py",
            symbol_name="my_func",
            symbol_type="function",
            start_line=10,
            end_line=20,
            code="def my_func(): pass",
        )
        assert chunk.file_path == "src/foo.py"
        assert chunk.symbol_name == "my_func"
        assert chunk.symbol_type == "function"
        assert chunk.start_line == 10
        assert chunk.end_line == 20
        assert "my_func" in chunk.code

    def test_make_id_is_deterministic(self) -> None:
        id1 = CodeChunk.make_id("src/foo.py", "my_func", 10)
        id2 = CodeChunk.make_id("src/foo.py", "my_func", 10)
        assert id1 == id2

    def test_make_id_differs_for_different_inputs(self) -> None:
        id1 = CodeChunk.make_id("src/foo.py", "my_func", 10)
        id2 = CodeChunk.make_id("src/bar.py", "my_func", 10)
        id3 = CodeChunk.make_id("src/foo.py", "other_func", 10)
        id4 = CodeChunk.make_id("src/foo.py", "my_func", 99)
        assert len({id1, id2, id3, id4}) == 4

    def test_id_length(self) -> None:
        chunk_id = CodeChunk.make_id("a", "b", 1)
        assert len(chunk_id) == 16


# ---------------------------------------------------------------------------
# load_repo_files tests
# ---------------------------------------------------------------------------


class TestLoadRepoFiles:
    def test_returns_list_of_dicts(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        assert isinstance(files, list)
        assert len(files) > 0

    def test_all_results_have_required_keys(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        required_keys = {"abs_path", "rel_path", "extension", "content"}
        for f in files:
            assert required_keys <= f.keys(), f"Missing keys in {f}"

    def test_git_dir_excluded(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        rel_paths = {f["rel_path"] for f in files}
        assert not any(".git" in p for p in rel_paths)

    def test_node_modules_excluded(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        rel_paths = {f["rel_path"] for f in files}
        assert not any("node_modules" in p for p in rel_paths)

    def test_python_files_included(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        py_files = [f for f in files if f["extension"] == ".py"]
        assert len(py_files) >= 2

    def test_binary_file_excluded(self, tmp_repo: Path) -> None:
        # Write a small binary blob.
        (tmp_repo / "binary.bin").write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        files = load_repo_files(str(tmp_repo))
        rel_paths = {f["rel_path"] for f in files}
        assert "binary.bin" not in rel_paths

    def test_content_is_string(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        for f in files:
            assert isinstance(f["content"], str)

    def test_rel_path_uses_forward_slashes(self, tmp_repo: Path) -> None:
        files = load_repo_files(str(tmp_repo))
        for f in files:
            assert "\\" not in f["rel_path"], (
                f"rel_path should use forward slashes, got: {f['rel_path']}"
            )


# ---------------------------------------------------------------------------
# Fallback line-chunking tests
# ---------------------------------------------------------------------------


class TestChunkWithLines:
    def _make_desc(self, content: str, rel_path: str = "fake.rb") -> dict:
        return {
            "abs_path": Path(f"/fake/{rel_path}"),
            "rel_path": rel_path,
            "extension": Path(rel_path).suffix.lower(),
            "content": content,
        }

    def test_single_chunk_for_short_file(self) -> None:
        content = "\n".join(f"line {i}" for i in range(10))
        chunks = _chunk_with_lines(self._make_desc(content), chunk_size=50, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "fallback"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10

    def test_multiple_chunks_for_long_file(self) -> None:
        content = "\n".join(f"line {i}" for i in range(100))
        chunks = _chunk_with_lines(self._make_desc(content), chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_overlap_means_lines_shared(self) -> None:
        content = "\n".join(str(i) for i in range(60))
        chunks = _chunk_with_lines(self._make_desc(content), chunk_size=50, overlap=10)
        assert len(chunks) >= 2
        # Second chunk should start before line 51 (overlap of 10).
        assert chunks[1].start_line <= 41

    def test_empty_file_returns_no_chunks(self) -> None:
        chunks = _chunk_with_lines(self._make_desc(""))
        assert chunks == []

    def test_symbol_names_are_sequential(self) -> None:
        content = "\n".join(f"line {i}" for i in range(120))
        chunks = _chunk_with_lines(self._make_desc(content), chunk_size=50, overlap=10)
        for idx, chunk in enumerate(chunks):
            assert chunk.symbol_name == f"chunk_{idx}"

    def test_line_boundaries_are_one_indexed(self) -> None:
        content = "only one line"
        chunks = _chunk_with_lines(self._make_desc(content))
        assert chunks[0].start_line == 1

    def test_end_line_does_not_exceed_total(self) -> None:
        content = "\n".join(f"line {i}" for i in range(47))
        chunks = _chunk_with_lines(self._make_desc(content), chunk_size=50, overlap=10)
        assert chunks[-1].end_line == 47

    def test_rel_path_stored_correctly(self) -> None:
        content = "hello"
        chunks = _chunk_with_lines(self._make_desc(content, "src/foo.rb"))
        assert chunks[0].file_path == "src/foo.rb"


# ---------------------------------------------------------------------------
# chunk_file – AST path (Python)
# ---------------------------------------------------------------------------


class TestChunkFilePython:
    def test_returns_list_of_code_chunks(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, CodeChunk) for c in chunks)

    def test_class_chunk_extracted(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        class_chunks = [c for c in chunks if c.symbol_type == "class"]
        assert len(class_chunks) >= 1, "Expected at least one class chunk"
        names = {c.symbol_name for c in class_chunks}
        assert "Greeter" in names

    def test_function_chunks_extracted(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        func_chunks = [c for c in chunks if c.symbol_type == "function"]
        assert len(func_chunks) >= 2, "Expected at least two function chunks"
        names = {c.symbol_name for c in func_chunks}
        assert "add" in names
        assert "multiply" in names

    def test_module_chunk_extracted(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        module_chunks = [c for c in chunks if c.symbol_type == "module"]
        # Module-level code (imports, constant) should produce a module chunk.
        assert len(module_chunks) >= 1

    def test_line_boundaries_are_positive(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_code_field_non_empty(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        for c in chunks:
            assert c.code.strip() != ""

    def test_ids_are_unique(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique within a file"

    def test_file_path_matches_descriptor(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        for c in chunks:
            assert c.file_path == "sample.py"

    def test_greeter_class_contains_correct_source(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        greeter = next((c for c in chunks if c.symbol_name == "Greeter"), None)
        assert greeter is not None
        assert "def greet" in greeter.code

    def test_add_function_line_boundaries(self, sample_python_file_desc: dict) -> None:
        chunks = chunk_file(sample_python_file_desc)
        add_chunk = next((c for c in chunks if c.symbol_name == "add"), None)
        assert add_chunk is not None
        assert add_chunk.start_line < add_chunk.end_line


# ---------------------------------------------------------------------------
# chunk_file – fallback path (unsupported extension)
# ---------------------------------------------------------------------------


class TestChunkFileFallback:
    def _ruby_desc(self, content: str) -> dict:
        return {
            "abs_path": Path("/fake/repo/script.rb"),
            "rel_path": "script.rb",
            "extension": ".rb",
            "content": content,
        }

    def test_fallback_used_for_ruby(self) -> None:
        content = "\n".join(f"puts {i}" for i in range(10))
        chunks = chunk_file(self._ruby_desc(content))
        assert all(c.symbol_type == "fallback" for c in chunks)

    def test_fallback_chunk_ids_non_empty(self) -> None:
        content = "line 1\nline 2"
        chunks = chunk_file(self._ruby_desc(content))
        for c in chunks:
            assert c.id


# ---------------------------------------------------------------------------
# chunk_repo – end-to-end integration
# ---------------------------------------------------------------------------


class TestChunkRepo:
    def test_returns_code_chunks(self, tmp_repo: Path) -> None:
        chunks = chunk_repo(str(tmp_repo))
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, CodeChunk) for c in chunks)

    def test_git_dir_not_in_chunks(self, tmp_repo: Path) -> None:
        chunks = chunk_repo(str(tmp_repo))
        for c in chunks:
            assert ".git" not in c.file_path

    def test_node_modules_not_in_chunks(self, tmp_repo: Path) -> None:
        chunks = chunk_repo(str(tmp_repo))
        for c in chunks:
            assert "node_modules" not in c.file_path

    def test_python_symbols_present(self, tmp_repo: Path) -> None:
        chunks = chunk_repo(str(tmp_repo))
        sym_names = {c.symbol_name for c in chunks}
        # main.py defines `run` and `App`; utils.py defines `helper`.
        assert "run" in sym_names or "helper" in sym_names, (
            f"Expected Python symbols; got: {sym_names}"
        )

    def test_all_chunks_have_valid_line_boundaries(self, tmp_repo: Path) -> None:
        chunks = chunk_repo(str(tmp_repo))
        for c in chunks:
            assert c.start_line >= 1, f"start_line < 1 for {c.file_path}:{c.symbol_name}"
            assert c.end_line >= c.start_line, (
                f"end_line < start_line for {c.file_path}:{c.symbol_name}"
            )

    def test_no_duplicate_ids_across_repo(self, tmp_repo: Path) -> None:
        chunks = chunk_repo(str(tmp_repo))
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found across repository"
