"""
chunker.py – Phase 1: File Ingestion & AST-Based Code Chunking
==============================================================
Responsibilities
----------------
* Recursively load source files from a local repository, filtering out
  binary blobs, VCS internals, dependency caches, and build artefacts.
* Parse Python (primary) and JavaScript / TypeScript (secondary) files
  with tree-sitter, extracting class and function boundaries as discrete
  :class:`CodeChunk` objects with rich metadata.
* Fall back to fixed-size line-overlap chunking for unsupported languages
  or files that cannot be parsed.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

#: Directories that should never be traversed.
_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "coverage",
        ".tox",
        "site-packages",
    }
)

#: File name patterns that should be skipped regardless of extension.
_IGNORED_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Pipfile.lock",
        "poetry.lock",
        "composer.lock",
        "Cargo.lock",
        "go.sum",
    }
)

#: Extensions considered plain-text / source code (non-binary).
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Python
        ".py",
        ".pyi",
        # JavaScript / TypeScript
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        # Web
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".svg",
        # Config / data
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        # Shell
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        # Docs / markup
        ".md",
        ".rst",
        ".txt",
        # Other common source
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".swift",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".lua",
        ".r",
    }
)

#: Fallback chunking parameters (lines per chunk, overlap).
_FALLBACK_CHUNK_SIZE: int = 50
_FALLBACK_OVERLAP: int = 10

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CodeChunk(BaseModel):
    """A single logical unit of source code extracted from a repository file."""

    id: str = Field(description="Stable SHA-256 derived identifier for this chunk.")
    file_path: str = Field(description="Relative path from the repository root.")
    symbol_name: str = Field(
        description="Name of the function, class, or a synthetic label for module-level code."
    )
    symbol_type: str = Field(description="One of: 'function', 'class', 'module', or 'fallback'.")
    start_line: int = Field(description="1-indexed first line of the chunk (inclusive).")
    end_line: int = Field(description="1-indexed last line of the chunk (inclusive).")
    code: str = Field(description="Raw source text of the chunk.")

    @staticmethod
    def make_id(file_path: str, symbol_name: str, start_line: int) -> str:
        """Generate a deterministic, content-independent chunk ID."""
        raw = f"{file_path}::{symbol_name}::{start_line}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------


def _is_binary(file_path: Path, sample_bytes: int = 8192) -> bool:
    """Return *True* if *file_path* looks like a binary file."""
    try:
        with file_path.open("rb") as fh:
            chunk = fh.read(sample_bytes)
        if b"\x00" in chunk:
            return True
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False
    except OSError:
        return True


def load_repo_files(repo_path: str) -> list[dict]:
    """
    Recursively scan *repo_path* and return a list of file descriptor dicts.

    Each dict contains:
    * ``abs_path``  – absolute :class:`pathlib.Path` to the file.
    * ``rel_path``  – path string relative to *repo_path*.
    * ``extension`` – lower-cased file extension including the leading dot.
    * ``content``   – decoded text content of the file.

    Files are silently skipped if they are:
    * Located inside an ignored directory.
    * A known lock-file / build-artefact.
    * Binary (detected by null-byte heuristic).
    * Not decodable as UTF-8.
    """
    root = Path(repo_path).resolve()
    results: list[dict] = []

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        # Prune ignored directories *in-place* so os.walk skips them entirely.
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]

        for filename in filenames:
            abs_path = Path(dirpath_str) / filename

            if filename in _IGNORED_FILENAMES:
                logger.debug("Skipping artefact file: %s", abs_path)
                continue

            ext = abs_path.suffix.lower()

            # If the extension is not in our allow-list, run a binary check first.
            if ext not in _TEXT_EXTENSIONS and _is_binary(abs_path):
                logger.debug("Skipping binary file: %s", abs_path)
                continue

            if _is_binary(abs_path):
                logger.debug("Skipping binary file: %s", abs_path)
                continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", abs_path, exc)
                continue

            rel_path = abs_path.relative_to(root).as_posix()
            results.append(
                {
                    "abs_path": abs_path,
                    "rel_path": rel_path,
                    "extension": ext,
                    "content": content,
                }
            )
            logger.debug("Loaded: %s", rel_path)

    logger.info("Loaded %d files from %s", len(results), repo_path)
    return results


# ---------------------------------------------------------------------------
# tree-sitter helpers
# ---------------------------------------------------------------------------

# Lazily populated parser cache keyed by language name.
_PARSERS: dict[str, object] = {}


def _get_parser(language_name: str):  # noqa: ANN201
    """
    Return a cached tree-sitter Parser for *language_name*.

    Tries ``tree-sitter-languages`` first (bundled grammars), then the
    standalone ``tree-sitter-python`` package.  Returns ``None`` when no
    grammar is available so callers can use the fallback chunker.
    """
    if language_name in _PARSERS:
        return _PARSERS[language_name]

    # Attempt 1: tree-sitter-languages (preferred – covers many languages).
    try:
        from tree_sitter_languages import get_parser  # type: ignore

        parser = get_parser(language_name)
        _PARSERS[language_name] = parser
        return parser
    except Exception:
        pass

    # Attempt 2: standalone tree-sitter-python package (Python only).
    try:
        import tree_sitter_python as tspython  # type: ignore
        from tree_sitter import Language, Parser  # type: ignore

        if language_name == "python":
            lang = Language(tspython.language())
            parser = Parser(lang)
            _PARSERS[language_name] = parser
            return parser
    except Exception:
        pass

    logger.warning("tree-sitter parser unavailable for language: %s", language_name)
    _PARSERS[language_name] = None
    return None


def _extension_to_language(ext: str) -> str | None:
    """Map a file extension to the tree-sitter language name."""
    mapping: dict[str, str] = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
    }
    return mapping.get(ext)


# ---------------------------------------------------------------------------
# AST-based chunking
# ---------------------------------------------------------------------------

_CLASS_NODE_TYPES: frozenset[str] = frozenset({"class_definition", "class_declaration"})

_FUNCTION_NODE_TYPES: frozenset[str] = frozenset(
    {
        "function_definition",
        "function_declaration",
        "method_definition",
        "arrow_function",
        "async_function_declaration",
    }
)


def _node_name(node, source_lines: list[str]) -> str:
    """Extract the symbol name from an AST node via its ``identifier`` child."""
    for child in node.children:
        if child.type in ("identifier", "name"):
            row = child.start_point[0]
            col_s = child.start_point[1]
            col_e = child.end_point[1]
            line_text = source_lines[row] if row < len(source_lines) else ""
            return line_text[col_s:col_e]
    return ""


def _iter_top_level_nodes(
    tree_root, source_lines: list[str]
) -> Iterator[tuple[str, str, int, int]]:
    """
    Yield ``(symbol_type, symbol_name, start_line, end_line)`` for every
    top-level class/function under *tree_root*.  Lines are 1-indexed, inclusive.
    """
    for node in tree_root.children:
        if node.type in _CLASS_NODE_TYPES:
            name = _node_name(node, source_lines) or "<anonymous_class>"
            yield "class", name, node.start_point[0] + 1, node.end_point[0] + 1
        elif node.type in _FUNCTION_NODE_TYPES:
            name = _node_name(node, source_lines) or "<anonymous_function>"
            yield "function", name, node.start_point[0] + 1, node.end_point[0] + 1


def _chunk_with_ast(file_desc: dict, parser) -> list[CodeChunk]:
    """
    Parse *file_desc* with *parser* and return CodeChunks for every
    top-level class/function.  Uncovered lines become a ``module`` chunk.
    """
    rel_path: str = file_desc["rel_path"]
    content: str = file_desc["content"]
    source_lines = content.splitlines()
    total_lines = len(source_lines)

    tree = parser.parse(content.encode("utf-8"))
    root = tree.root_node

    chunks: list[CodeChunk] = []
    covered: set[int] = set()

    for sym_type, sym_name, start_l, end_l in _iter_top_level_nodes(root, source_lines):
        code_slice = "\n".join(source_lines[start_l - 1 : end_l])
        covered.update(range(start_l, end_l + 1))
        chunks.append(
            CodeChunk(
                id=CodeChunk.make_id(rel_path, sym_name, start_l),
                file_path=rel_path,
                symbol_name=sym_name,
                symbol_type=sym_type,
                start_line=start_l,
                end_line=end_l,
                code=code_slice,
            )
        )

    # Gather module-level lines not owned by any symbol.
    module_lines = [source_lines[i - 1] for i in range(1, total_lines + 1) if i not in covered]
    if module_lines:
        module_code = "\n".join(module_lines)
        first_uncovered = next((i for i in range(1, total_lines + 1) if i not in covered), 1)
        last_uncovered = next(
            (i for i in range(total_lines, 0, -1) if i not in covered), total_lines
        )
        chunks.append(
            CodeChunk(
                id=CodeChunk.make_id(rel_path, "<module>", first_uncovered),
                file_path=rel_path,
                symbol_name="<module>",
                symbol_type="module",
                start_line=first_uncovered,
                end_line=last_uncovered,
                code=module_code,
            )
        )

    logger.debug("AST chunked %s → %d chunk(s)", rel_path, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Line-based fallback chunking
# ---------------------------------------------------------------------------


def _chunk_with_lines(
    file_desc: dict,
    chunk_size: int = _FALLBACK_CHUNK_SIZE,
    overlap: int = _FALLBACK_OVERLAP,
) -> list[CodeChunk]:
    """
    Divide *file_desc* into overlapping fixed-size line windows.

    Each chunk has ``symbol_type='fallback'`` and a synthetic
    ``symbol_name`` of the form ``chunk_N``.
    """
    rel_path: str = file_desc["rel_path"]
    source_lines = file_desc["content"].splitlines()
    total_lines = len(source_lines)

    if total_lines == 0:
        return []

    chunks: list[CodeChunk] = []
    chunk_index = 0
    pos = 0

    while pos < total_lines:
        end_pos = min(pos + chunk_size, total_lines)
        start_line = pos + 1
        end_line = end_pos
        code_slice = "\n".join(source_lines[pos:end_pos])
        sym_name = f"chunk_{chunk_index}"

        chunks.append(
            CodeChunk(
                id=CodeChunk.make_id(rel_path, sym_name, start_line),
                file_path=rel_path,
                symbol_name=sym_name,
                symbol_type="fallback",
                start_line=start_line,
                end_line=end_line,
                code=code_slice,
            )
        )

        chunk_index += 1
        step = max(chunk_size - overlap, 1)
        pos += step

    logger.debug("Fallback chunked %s → %d chunk(s)", rel_path, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(file_desc: dict) -> list[CodeChunk]:
    """
    Produce :class:`CodeChunk` objects for a single file descriptor (as
    returned by :func:`load_repo_files`).

    Strategy:
    1. Attempt AST parsing via tree-sitter if a grammar exists for the extension.
    2. Fall back to fixed-size line windowing otherwise.
    """
    ext: str = file_desc["extension"]
    rel_path: str = file_desc["rel_path"]

    lang = _extension_to_language(ext)
    if lang is not None:
        parser = _get_parser(lang)
        if parser is not None:
            try:
                return _chunk_with_ast(file_desc, parser)
            except Exception as exc:
                logger.warning(
                    "AST parsing failed for %s (%s); using fallback. Error: %s",
                    rel_path,
                    lang,
                    exc,
                )

    return _chunk_with_lines(file_desc)


def chunk_repo(repo_path: str) -> list[CodeChunk]:
    """
    End-to-end convenience: load all files in *repo_path* and chunk each one.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the root of the local repository.

    Returns
    -------
    list[CodeChunk]
        All code chunks extracted from every readable source file.
    """
    files = load_repo_files(repo_path)
    all_chunks: list[CodeChunk] = []

    for file_desc in files:
        try:
            all_chunks.extend(chunk_file(file_desc))
        except Exception as exc:
            logger.error("Unexpected error chunking %s: %s", file_desc["rel_path"], exc)

    logger.info(
        "Produced %d total chunks from %d files in %s",
        len(all_chunks),
        len(files),
        repo_path,
    )
    return all_chunks
