"""
git_utils.py – Phase 2: Git History Extraction
===============================================
Provides lightweight wrappers around GitPython to surface commit provenance
data for individual files inside a local repository.

Public API
----------
* ``get_git_file_history``  – Retrieve recent commit records for a file.
* ``format_git_metadata_string`` – Render a commit list as a compact text block.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_git_file_history(
    repo_path: str,
    rel_file_path: str,
    max_commits: int = 5,
) -> list[dict[str, Any]]:
    """
    Return the *max_commits* most-recent Git commits that touched *rel_file_path*.

    Each entry in the returned list is a plain dict with keys:

    * ``commit_hash``  – Full 40-character SHA-1.
    * ``short_hash``   – First 7 characters of the SHA-1.
    * ``author``       – Author name (str).
    * ``author_email`` – Author e-mail address (str).
    * ``date``         – Commit timestamp as an ISO-8601 string (UTC).
    * ``message``      – First line of the commit message (str).

    Returns an empty list when:
    * The directory is not a Git repository.
    * The file has no recorded commits (e.g., untracked or new).
    * GitPython is not installed.
    * Any other error that would otherwise propagate to the caller.
    """
    try:
        import git  # GitPython
    except ImportError:
        logger.warning(
            "GitPython is not installed; git history unavailable for %s",
            rel_file_path,
        )
        return []

    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        logger.debug("Not a Git repository (or no .git found above): %s", repo_path)
        return []
    except git.exc.NoSuchPathError:
        logger.warning("Repository path does not exist: %s", repo_path)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error opening repo %s: %s", repo_path, exc)
        return []

    history: list[dict[str, Any]] = []

    try:
        commits = list(repo.iter_commits(paths=rel_file_path, max_count=max_commits))
    except git.exc.GitCommandError as exc:
        logger.warning("git log failed for %s in %s: %s", rel_file_path, repo_path, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error fetching commits for %s: %s", rel_file_path, exc)
        return []

    for commit in commits:
        # Convert the authored_date (Unix timestamp) to a UTC ISO-8601 string.
        try:
            authored_dt = datetime.fromtimestamp(commit.authored_date, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            authored_dt = "unknown"

        history.append(
            {
                "commit_hash": commit.hexsha,
                "short_hash": commit.hexsha[:7],
                "author": commit.author.name or "",
                "author_email": commit.author.email or "",
                "date": authored_dt,
                "message": commit.message.strip().splitlines()[0] if commit.message else "",
            }
        )

    return history


def format_git_metadata_string(history: list[dict[str, Any]]) -> str:
    """
    Render a list of commit dicts (as returned by :func:`get_git_file_history`)
    into a compact, human-readable text block suitable for embedding alongside
    source code.

    Returns an empty string when *history* is empty.

    Example output::

        Git History (3 commits):
        [a1b2c3d] 2024-01-15 | Alice Smith <alice@example.com>: Add retry logic
        [d4e5f6g] 2024-01-10 | Bob Jones <bob@example.com>: Fix edge case
        [h7i8j9k] 2024-01-05 | Alice Smith <alice@example.com>: Initial implementation
    """
    if not history:
        return ""

    lines: list[str] = [f"Git History ({len(history)} commit{'s' if len(history) != 1 else ''}):"]

    for entry in history:
        short = entry.get("short_hash", "???????")
        # Trim the ISO date to just the date portion for brevity.
        raw_date: str = entry.get("date", "")
        date_str = raw_date[:10] if len(raw_date) >= 10 else raw_date
        author = entry.get("author", "Unknown")
        email = entry.get("author_email", "")
        message = entry.get("message", "")

        author_part = f"{author} <{email}>" if email else author
        lines.append(f"  [{short}] {date_str} | {author_part}: {message}")

    return "\n".join(lines)
