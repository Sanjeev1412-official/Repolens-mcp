"""
test_eval.py – Phase 5: Evaluation Pipeline Unit Tests
=======================================================
Tests the metric calculations, dataset loading, and report generation in
``eval/run_eval.py`` without requiring a full repository index.

Strategy
--------
* A tiny 4-case mock dataset is used throughout to keep the tests fast.
* A module-scoped ``tmp_repo`` + ``indexer`` fixture indexes 3 small Python
  files so that retrieval tests have real vectors to query against.
* The LLM judge is forced to rule-based mode in all tests (no API keys needed).
* ChromaDB dirs are isolated via ``tmp_path_factory`` to prevent cross-test
  pollution.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ and eval/ are importable.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_ROOT))

from eval.run_eval import (  # noqa: E402
    EvalCase,
    EvalReport,
    _rule_based_judge,
    compute_hit,
    compute_recall,
    generate_answer,
    judge_answer,
    load_dataset,
    print_report,
    run_evaluation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_CASES: list[dict] = [
    {
        "id": "mock_001",
        "category": "function_lookup",
        "question": "Which function loads repository files?",
        "expected_files": ["src/chunker.py"],
        "expected_symbols": ["load_repo_files"],
        "ground_truth_answer": (
            "The `load_repo_files` function in `src/chunker.py` loads repository files."
        ),
    },
    {
        "id": "mock_002",
        "category": "architecture",
        "question": "How does the indexer store embeddings?",
        "expected_files": ["src/indexer.py"],
        "expected_symbols": ["RepoLensIndexer"],
        "ground_truth_answer": (
            "RepoLensIndexer stores embeddings in ChromaDB using a persistent client."
        ),
    },
    {
        "id": "mock_003",
        "category": "git_history",
        "question": "How does git history extraction handle non-git directories?",
        "expected_files": ["src/git_utils.py"],
        "expected_symbols": ["get_git_file_history"],
        "ground_truth_answer": (
            "get_git_file_history catches InvalidGitRepositoryError and returns an empty list."
        ),
    },
    {
        "id": "mock_004",
        "category": "data_flow",
        "question": "What is the data flow for a search request?",
        "expected_files": ["src/server.py", "src/indexer.py"],
        "expected_symbols": ["search_codebase"],
        "ground_truth_answer": (
            "The server search_codebase tool delegates to RepoLensIndexer.search_codebase."
        ),
    },
]


@pytest.fixture(scope="module")
def dataset_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the mock dataset to a temp JSON file."""
    p = tmp_path_factory.mktemp("eval_data") / "mock_dataset.json"
    p.write_text(json.dumps({"cases": _MOCK_CASES}), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def tiny_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3-file Python repo for integration tests."""
    root = tmp_path_factory.mktemp("eval_repo")

    (root / "chunker.py").write_text(
        textwrap.dedent("""\
            def load_repo_files(repo_path: str) -> list:
                \"\"\"Load all source files from the repository.\"\"\"
                return []

            def chunk_file(file_desc: dict) -> list:
                \"\"\"Chunk a single file into CodeChunk objects.\"\"\"
                return []
        """),
        encoding="utf-8",
    )
    (root / "indexer.py").write_text(
        textwrap.dedent("""\
            class RepoLensIndexer:
                \"\"\"Manages embeddings and ChromaDB storage.\"\"\"

                def index_repository(self, repo_path: str) -> int:
                    \"\"\"Index all source files.\"\"\"
                    return 0

                def search_codebase(self, query: str, top_k: int = 5) -> list:
                    \"\"\"Semantic search over the indexed codebase.\"\"\"
                    return []
        """),
        encoding="utf-8",
    )
    (root / "git_utils.py").write_text(
        textwrap.dedent("""\
            def get_git_file_history(repo_path: str, rel_path: str) -> list:
                \"\"\"Return recent git commits for a file.\"\"\"
                return []
        """),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def eval_indexer(tiny_repo: Path, tmp_path_factory: pytest.TempPathFactory):
    """Module-scoped indexer over tiny_repo for retrieval tests."""
    from src.indexer import RepoLensIndexer

    chroma_dir = tmp_path_factory.mktemp("eval_chroma")
    idx = RepoLensIndexer(
        chroma_path=str(chroma_dir),
        collection_name="eval_unit_test",
    )
    idx.index_repository(str(tiny_repo))
    return idx


# ---------------------------------------------------------------------------
# TestDatasetLoader
# ---------------------------------------------------------------------------


class TestDatasetLoader:
    def test_load_returns_list_of_eval_cases(self, dataset_file: Path) -> None:
        cases = load_dataset(str(dataset_file))
        assert isinstance(cases, list)
        assert all(isinstance(c, EvalCase) for c in cases)

    def test_load_correct_count(self, dataset_file: Path) -> None:
        cases = load_dataset(str(dataset_file))
        assert len(cases) == 4

    def test_case_fields_populated(self, dataset_file: Path) -> None:
        cases = load_dataset(str(dataset_file))
        c = cases[0]
        assert c.id == "mock_001"
        assert c.category == "function_lookup"
        assert "load" in c.question.lower()
        assert c.expected_files == ["src/chunker.py"]
        assert c.expected_symbols == ["load_repo_files"]
        assert c.ground_truth_answer

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset("/this/does/not/exist/dataset.json")

    def test_load_flat_list_format(self, tmp_path: Path) -> None:
        """Dataset can also be a bare list (no 'cases' key)."""
        data = [
            {
                "id": "x_001",
                "category": "architecture",
                "question": "How does X work?",
                "expected_files": ["a.py"],
                "expected_symbols": ["f"],
                "ground_truth_answer": "X works by doing Y.",
            }
        ]
        p = tmp_path / "flat.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        cases = load_dataset(str(p))
        assert len(cases) == 1
        assert cases[0].id == "x_001"

    def test_malformed_case_is_skipped(self, tmp_path: Path) -> None:
        """Cases missing the 'question' key should be skipped without crashing."""
        data = {
            "cases": [
                {"id": "bad_001", "category": "x"},  # missing 'question'
                {
                    "id": "good_001",
                    "category": "function_lookup",
                    "question": "Valid question?",
                    "expected_files": [],
                    "expected_symbols": [],
                    "ground_truth_answer": "Valid.",
                },
            ]
        }
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        cases = load_dataset(str(p))
        assert len(cases) == 1
        assert cases[0].id == "good_001"


# ---------------------------------------------------------------------------
# TestComputeRecall
# ---------------------------------------------------------------------------


class TestComputeRecall:
    def test_perfect_recall(self) -> None:
        assert compute_recall(["src/chunker.py"], ["src/chunker.py"]) == 1.0

    def test_zero_recall(self) -> None:
        assert compute_recall(["src/chunker.py"], ["src/indexer.py"]) == 0.0

    def test_partial_recall(self) -> None:
        r = compute_recall(["a.py", "b.py"], ["a.py"])
        assert r == pytest.approx(0.5)

    def test_empty_expected_returns_one(self) -> None:
        assert compute_recall([], ["anything.py"]) == 1.0

    def test_empty_retrieved_with_expected_returns_zero(self) -> None:
        assert compute_recall(["a.py"], []) == 0.0

    def test_substring_matching(self) -> None:
        """'chunker.py' in expected should match 'src/chunker.py' in retrieved."""
        assert compute_recall(["chunker.py"], ["src/chunker.py"]) == 1.0

    def test_case_insensitive(self) -> None:
        assert compute_recall(["CHUNKER.PY"], ["src/chunker.py"]) == 1.0

    def test_all_retrieved_with_multiple_expected(self) -> None:
        r = compute_recall(["a.py", "b.py"], ["a.py", "b.py", "c.py"])
        assert r == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestComputeHit
# ---------------------------------------------------------------------------


class TestComputeHit:
    def test_hit_when_file_present(self) -> None:
        assert compute_hit(["src/chunker.py"], ["src/chunker.py"]) == 1

    def test_miss_when_file_absent(self) -> None:
        assert compute_hit(["src/chunker.py"], ["src/indexer.py"]) == 0

    def test_empty_expected_is_hit(self) -> None:
        assert compute_hit([], ["x.py"]) == 1

    def test_hit_with_one_of_two_expected(self) -> None:
        assert compute_hit(["a.py", "b.py"], ["b.py"]) == 1

    def test_miss_with_empty_retrieved(self) -> None:
        assert compute_hit(["a.py"], []) == 0


# ---------------------------------------------------------------------------
# TestRuleBasedJudge
# ---------------------------------------------------------------------------


class TestRuleBasedJudge:
    def test_returns_score_and_rationale(self) -> None:
        score, rationale = _rule_based_judge("q?", "ground truth answer", "ground truth answer")
        assert isinstance(score, float)
        assert isinstance(rationale, str)

    def test_score_in_valid_range(self) -> None:
        score, _ = _rule_based_judge("q?", "hello world test", "random answer")
        assert 1.0 <= score <= 5.0

    def test_high_overlap_gives_high_score(self) -> None:
        gt = "load repo files function chunker python scanner"
        gen = "load repo files function chunker python scanner additional"
        score, _ = _rule_based_judge("q?", gt, gen)
        assert score >= 4.0

    def test_no_overlap_gives_low_score(self) -> None:
        gt = "the quick brown fox jumps over the lazy dog"
        gen = "completely unrelated answer about databases and schemas"
        score, _ = _rule_based_judge("q?", gt, gen)
        assert score <= 2.0

    def test_empty_generated_gives_score_one(self) -> None:
        score, _ = _rule_based_judge("q?", "ground truth", "")
        assert score == 1.0

    def test_empty_ground_truth_returns_three(self) -> None:
        score, _ = _rule_based_judge("q?", "", "some answer")
        assert score == 3.0

    def test_judge_answer_with_force_rule_based(self) -> None:
        score, rationale = judge_answer("q?", "ground truth", "answer", force_rule_based=True)
        assert 1.0 <= score <= 5.0
        assert isinstance(rationale, str)


# ---------------------------------------------------------------------------
# TestGenerateAnswer
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    def test_returns_string(self) -> None:
        results = [
            {
                "file_path": "src/chunker.py",
                "symbol_name": "load_repo_files",
                "symbol_type": "function",
                "score": 0.9,
                "document": "FILE: src/chunker.py\n---\ndef load_repo_files(): pass",
            }
        ]
        out = generate_answer("question?", results, "ground truth")
        assert isinstance(out, str) and out.strip()

    def test_empty_results_returns_no_results_message(self) -> None:
        out = generate_answer("question?", [], "ground truth")
        assert "No relevant" in out

    def test_contains_file_path(self) -> None:
        results = [
            {
                "file_path": "src/indexer.py",
                "symbol_name": "RepoLensIndexer",
                "symbol_type": "class",
                "score": 0.8,
                "document": "FILE: src/indexer.py\n---\nclass RepoLensIndexer: pass",
            }
        ]
        out = generate_answer("question?", results, "ground truth")
        assert "src/indexer.py" in out

    def test_caps_at_three_results(self) -> None:
        results = [
            {
                "file_path": f"file_{i}.py",
                "symbol_name": f"func_{i}",
                "symbol_type": "function",
                "score": 0.9 - i * 0.1,
                "document": f"FILE: file_{i}.py\n---\ndef func_{i}(): pass",
            }
            for i in range(6)
        ]
        out = generate_answer("question?", results, "gt")
        # Only top-3 should appear.
        assert "file_0.py" in out
        assert "file_2.py" in out
        assert "file_3.py" not in out


# ---------------------------------------------------------------------------
# TestRunEvaluation – integration (uses real indexer)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_report(tiny_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> EvalReport:
    """
    Run ``run_evaluation`` exactly ONCE (module scope) to avoid reloading the
    sentence-transformer model on every test, which causes a Windows access
    violation when PyTorch tries to materialise model weights in multiple
    concurrent-ish loads.
    """
    chroma = tmp_path_factory.mktemp("run_eval_shared_chroma")
    cases = [
        EvalCase(
            id="t_001",
            category="function_lookup",
            question="Which function loads repository files?",
            expected_files=["chunker.py"],
            expected_symbols=["load_repo_files"],
            ground_truth_answer="load_repo_files loads repository files.",
        ),
        EvalCase(
            id="t_002",
            category="architecture",
            question="How does the indexer class store embeddings?",
            expected_files=["indexer.py"],
            expected_symbols=["RepoLensIndexer"],
            ground_truth_answer="RepoLensIndexer stores embeddings in ChromaDB.",
        ),
        EvalCase(
            id="t_003",
            category="git_history",
            question="How does git history utility work?",
            expected_files=["git_utils.py"],
            expected_symbols=["get_git_file_history"],
            ground_truth_answer="get_git_file_history returns commit history.",
        ),
    ]
    return run_evaluation(
        cases,
        str(tiny_repo),
        top_k=5,
        chroma_path=str(chroma),
        force_rule_based=True,
    )


class TestRunEvaluation:
    """
    All tests consume the shared module-scoped ``eval_report`` fixture so the
    sentence-transformer model is only loaded once across the entire class.
    """

    def test_returns_eval_report(self, eval_report: EvalReport) -> None:
        assert isinstance(eval_report, EvalReport)

    def test_total_cases_matches_input(self, eval_report: EvalReport) -> None:
        assert eval_report.total_cases == 3

    def test_avg_file_recall_is_float_in_range(self, eval_report: EvalReport) -> None:
        assert 0.0 <= eval_report.avg_file_recall <= 1.0

    def test_hit_rate_is_percentage(self, eval_report: EvalReport) -> None:
        assert 0.0 <= eval_report.file_hit_rate <= 100.0
        assert 0.0 <= eval_report.symbol_hit_rate <= 100.0

    def test_avg_llm_score_in_valid_range(self, eval_report: EvalReport) -> None:
        assert 1.0 <= eval_report.avg_llm_score <= 5.0

    def test_category_stats_populated(self, eval_report: EvalReport) -> None:
        expected_cats = {"function_lookup", "architecture", "git_history"}
        assert expected_cats <= eval_report.category_stats.keys()

    def test_case_results_length_matches(self, eval_report: EvalReport) -> None:
        assert len(eval_report.case_results) == 3

    def test_case_result_has_required_keys(self, eval_report: EvalReport) -> None:
        required = {
            "id",
            "category",
            "question",
            "file_recall",
            "symbol_recall",
            "file_hit",
            "symbol_hit",
            "generated_answer",
            "llm_score",
            "llm_rationale",
        }
        for cr in eval_report.case_results:
            assert required <= cr.keys(), f"Missing keys in case result: {cr['id']}"

    def test_positive_chunks_indexed(self, eval_indexer) -> None:
        """Reuse the already-loaded eval_indexer to verify chunk count > 0."""
        assert eval_indexer.collection_count > 0


# ---------------------------------------------------------------------------
# TestPrintReport – smoke test (just ensure no exceptions raised)
# ---------------------------------------------------------------------------


class TestPrintReport:
    def _make_report(self) -> EvalReport:
        return EvalReport(
            repo_path="/fake/repo",
            dataset_path="/fake/dataset.json",
            top_k=5,
            total_cases=4,
            avg_file_recall=0.75,
            avg_symbol_recall=0.60,
            file_hit_rate=75.0,
            symbol_hit_rate=60.0,
            avg_llm_score=3.5,
            category_stats={
                "function_lookup": {
                    "count": 2,
                    "avg_file_recall": 0.8,
                    "avg_symbol_recall": 0.7,
                    "file_hit_rate_%": 80.0,
                    "symbol_hit_rate_%": 70.0,
                    "avg_llm_score": 3.8,
                },
                "architecture": {
                    "count": 2,
                    "avg_file_recall": 0.7,
                    "avg_symbol_recall": 0.5,
                    "file_hit_rate_%": 70.0,
                    "symbol_hit_rate_%": 50.0,
                    "avg_llm_score": 3.2,
                },
            },
            case_results=[
                {
                    "id": "t_001",
                    "category": "function_lookup",
                    "question": "Test?",
                    "file_recall": 1.0,
                    "symbol_recall": 1.0,
                    "file_hit": 1,
                    "symbol_hit": 1,
                    "generated_answer": "answer",
                    "llm_score": 4.0,
                    "llm_rationale": "Good.",
                    "retrieved_files": ["src/chunker.py"],
                    "retrieved_symbols": ["load_repo_files"],
                    "latency_ms": 12.0,
                    "error": "",
                },
                {
                    "id": "t_002",
                    "category": "architecture",
                    "question": "Architecture?",
                    "file_recall": 0.0,
                    "symbol_recall": 0.0,
                    "file_hit": 0,
                    "symbol_hit": 0,
                    "generated_answer": "",
                    "llm_score": 1.0,
                    "llm_rationale": "Not found.",
                    "retrieved_files": [],
                    "retrieved_symbols": [],
                    "latency_ms": 8.0,
                    "error": "",
                },
            ],
            total_latency_ms=20.0,
        )

    def test_print_report_does_not_raise(self, capsys) -> None:
        report = self._make_report()
        print_report(report)  # should not raise
        captured = capsys.readouterr()
        assert "RepoLens MCP" in captured.out

    def test_print_report_shows_hit_rate(self, capsys) -> None:
        report = self._make_report()
        print_report(report)
        captured = capsys.readouterr()
        assert "75.0%" in captured.out or "Hit Rate" in captured.out

    def test_print_report_shows_category_breakdown(self, capsys) -> None:
        report = self._make_report()
        print_report(report)
        captured = capsys.readouterr()
        assert "function_lookup" in captured.out
        assert "architecture" in captured.out

    def test_print_report_shows_failure_section(self, capsys) -> None:
        report = self._make_report()
        print_report(report)
        captured = capsys.readouterr()
        # t_002 has file_hit=0 → should appear in failures.
        assert "t_002" in captured.out
