"""
indexer.py – Phase 2: Vector Embeddings & ChromaDB Storage
===========================================================
``RepoLensIndexer`` is the single class responsible for:

1. Generating sentence-transformer embeddings for code chunks.
2. Storing/retrieving those embeddings in a persistent ChromaDB collection.
3. Enriching each chunk document with file metadata and Git provenance.

Public API
----------
* ``RepoLensIndexer.__init__``   – Initialise ChromaDB client and embedding model.
* ``RepoLensIndexer.index_repository`` – Walk repo, chunk, embed, upsert.
* ``RepoLensIndexer.search_codebase``  – Semantic query → ranked result dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_COLLECTION_NAME: str = "repolens_codebase"
_DEFAULT_EMBED_MODEL: str = "all-MiniLM-L6-v2"
_DEFAULT_CHROMA_PATH: str = "./chroma_db"


# ---------------------------------------------------------------------------
# RepoLensIndexer
# ---------------------------------------------------------------------------


class RepoLensIndexer:
    """
    Manages embedding generation and vector storage for a local code repository.

    Parameters
    ----------
    chroma_path:
        Directory where ChromaDB will persist its data.  Defaults to
        ``./chroma_db`` relative to the current working directory.
    collection_name:
        Name of the ChromaDB collection to create or reuse.
    embed_model_name:
        Sentence-transformers model identifier.  The model is downloaded on
        first use and cached by the ``sentence-transformers`` library.
    """

    def __init__(
        self,
        chroma_path: str = _DEFAULT_CHROMA_PATH,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        embed_model_name: str = _DEFAULT_EMBED_MODEL,
    ) -> None:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._embed_model_name = embed_model_name
        self._collection_name = collection_name

        # Persistent ChromaDB client.
        chroma_abs = str(Path(chroma_path).resolve())
        self.chroma_path = chroma_abs
        logger.info("Initialising ChromaDB at %s", chroma_abs)
        self._client = chromadb.PersistentClient(path=chroma_abs)

        # Get-or-create the collection with cosine similarity.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Load the embedding model once and reuse it across calls.
        logger.info("Loading embedding model: %s", embed_model_name)
        self._model = SentenceTransformer(embed_model_name)
        logger.info("RepoLensIndexer ready (collection=%s)", collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings as plain Python float lists (ChromaDB-compatible)."""
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    @staticmethod
    def _build_document(
        chunk,  # CodeChunk
        git_summary: str,
    ) -> str:
        """
        Compose the enriched text string that will be embedded and stored.

        Layout::

            FILE: <rel_path>  |  SYMBOL: <name> (<type>)  |  LINES: start-end
            <git_summary>
            ---
            <source code>
        """
        header = (
            f"FILE: {chunk.file_path}"
            f"  |  SYMBOL: {chunk.symbol_name} ({chunk.symbol_type})"
            f"  |  LINES: {chunk.start_line}-{chunk.end_line}"
        )
        parts = [header]
        if git_summary:
            parts.append(git_summary)
        parts.append("---")
        parts.append(chunk.code)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_repository(self, repo_path: str) -> int:
        """
        Index all source files in *repo_path* into ChromaDB.

        Steps:
        1. Load and chunk files via ``src.chunker``.
        2. Fetch per-file Git history via ``src.git_utils``.
        3. Build enriched document strings.
        4. Batch-embed and upsert into ChromaDB.

        Returns
        -------
        int
            Total number of code chunks successfully indexed.
        """
        from src.chunker import chunk_file, load_repo_files
        from src.git_utils import format_git_metadata_string, get_git_file_history
        import os
        import json

        logger.info("Starting indexing for repository: %s", repo_path)

        files = load_repo_files(repo_path)
        if not files:
            logger.warning("No files found in %s", repo_path)
            return 0

        # --- Check mtimes to skip indexing if nothing changed ---
        current_mtimes = {
            f["rel_path"]: int(os.path.getmtime(f["abs_path"])) for f in files
        }
        mtimes_path = Path(self.chroma_path) / "mtimes.json"
        if mtimes_path.exists():
            try:
                cached_mtimes = json.loads(mtimes_path.read_text(encoding="utf-8"))
                if cached_mtimes == current_mtimes:
                    count = self._collection.count()
                    logger.info("No files modified. Skipping indexing. (chunks=%d)", count)
                    return count
            except Exception as exc:
                logger.warning("Failed to read mtimes cache: %s", exc)

        # Cache git history per file path to avoid redundant git calls.
        _git_cache: dict[str, str] = {}

        all_ids: list[str] = []
        all_docs: list[str] = []
        all_embeddings: list[list[float]] = []
        all_metadatas: list[dict[str, Any]] = []

        for file_desc in files:
            rel_path: str = file_desc["rel_path"]

            # Fetch git history (cached per file).
            if rel_path not in _git_cache:
                history = get_git_file_history(repo_path, rel_path)
                _git_cache[rel_path] = format_git_metadata_string(history)
            git_summary = _git_cache[rel_path]

            # Chunk the file.
            try:
                chunks = chunk_file(file_desc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to chunk %s: %s", rel_path, exc)
                continue

            for chunk in chunks:
                doc = self._build_document(chunk, git_summary)
                all_ids.append(chunk.id)
                all_docs.append(doc)
                all_metadatas.append(
                    {
                        "file_path": chunk.file_path,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }
                )

        if not all_ids:
            logger.warning("No chunks produced for %s", repo_path)
            return 0

        # Embed in one batch for efficiency.
        logger.info("Embedding %d chunks …", len(all_ids))
        all_embeddings = self._embed(all_docs)

        # ChromaDB has a default batch limit; upsert in pages of 5 000.
        _BATCH = 5_000
        for start in range(0, len(all_ids), _BATCH):
            end = start + _BATCH
            self._collection.upsert(
                ids=all_ids[start:end],
                documents=all_docs[start:end],
                embeddings=all_embeddings[start:end],
                metadatas=all_metadatas[start:end],
            )

        total = len(all_ids)
        logger.info("Indexed %d chunks from %s", total, repo_path)
        
        # Update mtime cache on success
        try:
            mtimes_path.write_text(json.dumps(current_mtimes), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write mtimes cache: %s", exc)
            
        return total

    def search_codebase(
        self,
        query: str,
        top_k: int = 5,
        file_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform a semantic similarity search over the indexed codebase.

        Parameters
        ----------
        query:
            Natural-language or code snippet to search for.
        top_k:
            Maximum number of results to return.
        file_filter:
            Optional substring filter on ``file_path`` metadata.  Only chunks
            whose ``file_path`` contains this string are returned.

        Returns
        -------
        list[dict]
            Each dict has keys:
            ``score``, ``chunk_id``, ``file_path``, ``symbol_name``,
            ``symbol_type``, ``start_line``, ``end_line``, ``document``.
        """
        if not query.strip():
            return []

        query_embedding = self._embed([query])[0]

        # Build the optional where clause for ChromaDB metadata filtering.
        where: dict[str, Any] | None = None
        if file_filter:
            where = {"file_path": {"$contains": file_filter}}

        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ChromaDB query failed: %s", exc)
            return []

        matches: list[dict[str, Any]] = []

        ids_list = result.get("ids", [[]])[0]
        docs_list = result.get("documents", [[]])[0]
        metas_list = result.get("metadatas", [[]])[0]
        dists_list = result.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(
            ids_list, docs_list, metas_list, dists_list, strict=False
        ):
            # ChromaDB cosine distance → similarity score (higher = more similar).
            score = 1.0 - float(dist)
            matches.append(
                {
                    "score": round(score, 6),
                    "chunk_id": chunk_id,
                    "file_path": meta.get("file_path", ""),
                    "symbol_name": meta.get("symbol_name", ""),
                    "symbol_type": meta.get("symbol_type", ""),
                    "start_line": meta.get("start_line", 0),
                    "end_line": meta.get("end_line", 0),
                    "document": doc,
                }
            )

        # Sort descending by score (closest match first).
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset_collection(self) -> None:
        """Drop and recreate the ChromaDB collection (useful for re-indexing)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' has been reset.", self._collection_name)

    @property
    def collection_count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        return self._collection.count()
