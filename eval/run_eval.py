"""
run_eval.py – Phase 5: Automated Evaluation Suite
==================================================
Measures the retrieval and answer quality of the RepoLens MCP system
against a ground-truth benchmark dataset.

Metrics
-------
* **Recall@K**        – fraction of expected_files/expected_symbols present
                        in the top-K results.
* **File Hit Rate**   – binary: 1 if ≥1 expected file appears in top-K.
* **Symbol Hit Rate** – binary: 1 if ≥1 expected symbol appears in top-K.
* **LLM Correctness** – 1–5 score from an LLM judge (or rule-based fallback).

Usage
-----
    python eval/run_eval.py --repo .
    python eval/run_eval.py --repo /path/to/repo --top-k 10 --output results.json
    python eval/run_eval.py --dataset eval/dataset.json --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap – allow running from project root or from eval/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("repolens.eval")

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _bold(t: str) -> str:
    return _c("1", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _dim(t: str) -> str:
    return _c("2", t)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One benchmark test case loaded from dataset.json."""

    id: str
    category: str
    question: str
    expected_files: list[str]
    expected_symbols: list[str]
    ground_truth_answer: str


@dataclass
class CaseResult:
    """Metrics computed for a single EvalCase."""

    id: str
    category: str
    question: str
    # Retrieval
    file_recall: float  # fraction of expected_files found in top-K
    symbol_recall: float  # fraction of expected_symbols found in top-K
    file_hit: int  # 1 if ≥1 expected file retrieved
    symbol_hit: int  # 1 if ≥1 expected symbol retrieved
    # Answer quality
    generated_answer: str
    llm_score: float  # 1.0–5.0
    llm_rationale: str
    # Diagnostic
    retrieved_files: list[str] = field(default_factory=list)
    retrieved_symbols: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class EvalReport:
    """Aggregate benchmark report."""

    repo_path: str
    dataset_path: str
    top_k: int
    total_cases: int
    # Overall averages
    avg_file_recall: float
    avg_symbol_recall: float
    file_hit_rate: float  # percentage
    symbol_hit_rate: float  # percentage
    avg_llm_score: float
    # Per-category breakdown
    category_stats: dict[str, dict[str, float]]
    # Individual results
    case_results: list[dict[str, Any]]
    # Timing
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> list[EvalCase]:
    """Load and validate the JSON benchmark dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []

    items = raw.get("cases", raw) if isinstance(raw, dict) else raw

    for item in items:
        try:
            cases.append(
                EvalCase(
                    id=item["id"],
                    category=item.get("category", "unknown"),
                    question=item["question"],
                    expected_files=item.get("expected_files", []),
                    expected_symbols=item.get("expected_symbols", []),
                    ground_truth_answer=item.get("ground_truth_answer", ""),
                )
            )
        except KeyError as exc:
            logger.warning("Skipping malformed case (missing key %s): %s", exc, item.get("id", "?"))

    return cases


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def compute_recall(
    expected: list[str],
    retrieved: list[str],
) -> float:
    """
    Recall = |expected ∩ retrieved| / |expected|.
    Returns 1.0 when expected is empty (nothing to recall).
    Matching is case-insensitive substring containment: an expected value is
    considered retrieved if any retrieved value contains it as a substring.
    """
    if not expected:
        return 1.0
    hits = sum(1 for exp in expected if any(exp.lower() in ret.lower() for ret in retrieved))
    return hits / len(expected)


def compute_hit(expected: list[str], retrieved: list[str]) -> int:
    """Binary hit: 1 if at least one expected item appears in retrieved."""
    if not expected:
        return 1
    return int(any(any(exp.lower() in ret.lower() for ret in retrieved) for exp in expected))


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict technical evaluator for a code search system.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{generated}

Rate the generated answer on a scale from 1 to 5 using these criteria:
5 - Completely correct, well-grounded, no hallucinations.
4 - Mostly correct with minor omissions or imprecisions.
3 - Partially correct; some key facts right but missing important details.
2 - Mostly incorrect or contains significant hallucinations.
1 - Completely wrong or irrelevant.

Respond with exactly two lines:
SCORE: <integer 1-5>
RATIONALE: <one sentence explanation>
"""


def _rule_based_judge(
    question: str,
    ground_truth: str,
    generated: str,
) -> tuple[float, str]:
    """
    Deterministic rule-based judge used when no LLM API key is available.

    Strategy: keyword overlap between ground_truth and generated answer.
    """
    if not generated or not generated.strip():
        return 1.0, "No answer generated."

    # Extract content words (>3 chars, lower-cased).
    def _words(text: str) -> set[str]:
        import re

        return {w.lower() for w in re.findall(r"\b\w{4,}\b", text)}

    gt_words = _words(ground_truth)
    gen_words = _words(generated)

    if not gt_words:
        return 3.0, "Ground truth is empty; cannot evaluate."

    overlap = len(gt_words & gen_words) / len(gt_words)

    if overlap >= 0.55:
        score, rationale = 5.0, f"High keyword overlap ({overlap:.0%}) with ground truth."
    elif overlap >= 0.40:
        score, rationale = 4.0, f"Good keyword overlap ({overlap:.0%}) with ground truth."
    elif overlap >= 0.25:
        score, rationale = 3.0, f"Moderate keyword overlap ({overlap:.0%}) with ground truth."
    elif overlap >= 0.10:
        score, rationale = 2.0, f"Low keyword overlap ({overlap:.0%}) with ground truth."
    else:
        score, rationale = 1.0, f"Very low keyword overlap ({overlap:.0%}) with ground truth."

    return score, rationale


def _openai_judge(
    question: str,
    ground_truth: str,
    generated: str,
    model: str = "gpt-4o-mini",
) -> tuple[float, str]:
    """Call OpenAI to score the generated answer."""
    import openai  # type: ignore

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        generated=generated,
    )
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    text = response.choices[0].message.content or ""
    return _parse_judge_response(text)


def _anthropic_judge(
    question: str,
    ground_truth: str,
    generated: str,
    model: str = "claude-3-haiku-20240307",
) -> tuple[float, str]:
    """Call Anthropic Claude to score the generated answer."""
    import anthropic  # type: ignore

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        generated=generated,
    )
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text if message.content else ""
    return _parse_judge_response(text)


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract SCORE and RATIONALE from a judge LLM response."""
    score = 3.0
    rationale = "Could not parse judge response."

    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
                score = max(1.0, min(5.0, score))
            except ValueError:
                pass
        elif line.upper().startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()

    return score, rationale


def judge_answer(
    question: str,
    ground_truth: str,
    generated: str,
    force_rule_based: bool = False,
) -> tuple[float, str]:
    """
    Score *generated* against *ground_truth* using the best available judge.

    Priority order:
    1. OpenAI (if OPENAI_API_KEY is set and ``openai`` is installed).
    2. Anthropic (if ANTHROPIC_API_KEY is set and ``anthropic`` is installed).
    3. Rule-based keyword overlap (always available, deterministic).
    """
    if force_rule_based:
        return _rule_based_judge(question, ground_truth, generated)

    # --- Try OpenAI ---
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _openai_judge(question, ground_truth, generated)
        except Exception as exc:
            logger.warning("OpenAI judge failed: %s – falling back.", exc)

    # --- Try Anthropic ---
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_judge(question, ground_truth, generated)
        except Exception as exc:
            logger.warning("Anthropic judge failed: %s – falling back.", exc)

    return _rule_based_judge(question, ground_truth, generated)


# ---------------------------------------------------------------------------
# Answer generation (RAG-style: format top-K chunks as context)
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    search_results: list[dict[str, Any]],
    ground_truth: str,
) -> str:
    """
    Produce a generated answer from retrieval results.

    In evaluation mode we use the retrieved chunk documents as context and
    construct a structured summary without requiring a generative LLM.  If an
    LLM API is available it could be called here for a full RAG answer; the
    current implementation produces a rule-based extraction for reproducibility.
    """
    if not search_results:
        return "No relevant code chunks were retrieved for this question."

    # Extract key facts from the top-3 chunks.
    lines: list[str] = []
    for r in search_results[:3]:
        file_path = r.get("file_path", "")
        symbol = r.get("symbol_name", "")
        sym_type = r.get("symbol_type", "")
        score = r.get("score", 0.0)
        doc = r.get("document", "")

        # Pull the code portion (after the "---" separator).
        sep = "---\n"
        code = doc[doc.find(sep) + len(sep) :].strip() if sep in doc else doc.strip()
        code_preview = "\n".join(code.splitlines()[:6])

        lines.append(f"[{file_path} | {sym_type}: {symbol} | score={score:.3f}]\n{code_preview}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    cases: list[EvalCase],
    repo_path: str,
    top_k: int = 5,
    chroma_path: str | None = None,
    force_rule_based: bool = False,
    verbose: bool = False,
) -> EvalReport:
    """
    Run the full evaluation loop and return an :class:`EvalReport`.

    Parameters
    ----------
    cases       : Benchmark test cases.
    repo_path   : Repository to index.
    top_k       : Number of search results to retrieve per query.
    chroma_path : Override ChromaDB storage path (defaults to temp-adjacent dir).
    force_rule_based : Skip LLM judge even if API keys are present.
    verbose     : Print per-case progress.
    """
    from src.indexer import RepoLensIndexer

    # --- Set up indexer ---
    chroma_dir = chroma_path or str(Path(repo_path) / ".eval_chroma_db")
    print(_cyan(f"\n  Initialising indexer → ChromaDB at {chroma_dir}"))
    indexer = RepoLensIndexer(chroma_path=chroma_dir, collection_name="eval_run")

    print(_cyan(f"  Indexing repository: {repo_path}"))
    t0 = time.perf_counter()
    chunk_count = indexer.index_repository(repo_path)
    index_time = time.perf_counter() - t0
    print(_cyan(f"  Indexed {chunk_count} chunks in {index_time:.1f}s\n"))

    # --- Per-case loop ---
    results: list[CaseResult] = []
    total_latency = 0.0

    for i, case in enumerate(cases, start=1):
        prefix = f"  [{i:>2}/{len(cases)}] {case.id}"
        if verbose:
            print(_dim(f"{prefix} – {case.question[:70]}…"))

        try:
            t_start = time.perf_counter()
            search_results = indexer.search_codebase(case.question, top_k=top_k)
            latency_ms = (time.perf_counter() - t_start) * 1000
        except Exception as exc:
            logger.error("Search failed for %s: %s", case.id, exc)
            results.append(
                CaseResult(
                    id=case.id,
                    category=case.category,
                    question=case.question,
                    file_recall=0.0,
                    symbol_recall=0.0,
                    file_hit=0,
                    symbol_hit=0,
                    generated_answer="",
                    llm_score=1.0,
                    llm_rationale="Search raised an exception.",
                    error=str(exc),
                )
            )
            continue

        retrieved_files = [r["file_path"] for r in search_results]
        retrieved_symbols = [r["symbol_name"] for r in search_results]

        file_recall = compute_recall(case.expected_files, retrieved_files)
        symbol_recall = compute_recall(case.expected_symbols, retrieved_symbols)
        file_hit = compute_hit(case.expected_files, retrieved_files)
        symbol_hit = compute_hit(case.expected_symbols, retrieved_symbols)

        # Generate answer from retrieved context.
        generated = generate_answer(case.question, search_results, case.ground_truth_answer)

        # Score the answer.
        try:
            llm_score, llm_rationale = judge_answer(
                case.question,
                case.ground_truth_answer,
                generated,
                force_rule_based=force_rule_based,
            )
        except Exception as exc:
            logger.warning("Judge failed for %s: %s", case.id, exc)
            llm_score, llm_rationale = 1.0, f"Judge error: {exc}"

        total_latency += latency_ms

        cr = CaseResult(
            id=case.id,
            category=case.category,
            question=case.question,
            file_recall=round(file_recall, 4),
            symbol_recall=round(symbol_recall, 4),
            file_hit=file_hit,
            symbol_hit=symbol_hit,
            generated_answer=generated,
            llm_score=llm_score,
            llm_rationale=llm_rationale,
            retrieved_files=retrieved_files,
            retrieved_symbols=retrieved_symbols,
            latency_ms=round(latency_ms, 1),
        )
        results.append(cr)

        if verbose:
            status = _green("✓") if file_hit else _red("✗")
            print(
                f"  {status} file_recall={file_recall:.2f}  "
                f"sym_recall={symbol_recall:.2f}  "
                f"llm={llm_score:.1f}  "
                f"({latency_ms:.0f}ms)"
            )

    # --- Aggregate ---
    n = len(results)
    avg_file_recall = sum(r.file_recall for r in results) / n if n else 0.0
    avg_symbol_recall = sum(r.symbol_recall for r in results) / n if n else 0.0
    file_hit_rate = (sum(r.file_hit for r in results) / n * 100) if n else 0.0
    symbol_hit_rate = (sum(r.symbol_hit for r in results) / n * 100) if n else 0.0
    avg_llm_score = sum(r.llm_score for r in results) / n if n else 0.0

    # Per-category stats.
    categories: dict[str, list[CaseResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_stats: dict[str, dict[str, float]] = {}
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        category_stats[cat] = {
            "count": cn,
            "avg_file_recall": round(sum(r.file_recall for r in cat_results) / cn, 4),
            "avg_symbol_recall": round(sum(r.symbol_recall for r in cat_results) / cn, 4),
            "file_hit_rate_%": round(sum(r.file_hit for r in cat_results) / cn * 100, 1),
            "symbol_hit_rate_%": round(sum(r.symbol_hit for r in cat_results) / cn * 100, 1),
            "avg_llm_score": round(sum(r.llm_score for r in cat_results) / cn, 2),
        }

    # Clean up eval ChromaDB.
    try:
        import shutil

        shutil.rmtree(chroma_dir, ignore_errors=True)
    except Exception:
        pass

    return EvalReport(
        repo_path=str(Path(repo_path).resolve()),
        dataset_path="",
        top_k=top_k,
        total_cases=n,
        avg_file_recall=round(avg_file_recall, 4),
        avg_symbol_recall=round(avg_symbol_recall, 4),
        file_hit_rate=round(file_hit_rate, 1),
        symbol_hit_rate=round(symbol_hit_rate, 1),
        avg_llm_score=round(avg_llm_score, 2),
        category_stats=category_stats,
        case_results=[asdict(r) for r in results],
        total_latency_ms=round(total_latency, 1),
    )


# ---------------------------------------------------------------------------
# Console report printer
# ---------------------------------------------------------------------------


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction * 100:.1f}%"


def print_report(report: EvalReport) -> None:
    """Print a rich terminal summary table."""
    W = 65
    print()
    print(_bold(_cyan("╔" + "═" * W + "╗")))
    print(_bold(_cyan("║" + "  RepoLens MCP – Evaluation Report".center(W) + "║")))
    print(_bold(_cyan("╚" + "═" * W + "╝")))
    print()

    # ── Overall metrics ────────────────────────────────────────────────
    print(_bold("  Overall Metrics"))
    print("  " + "─" * (W - 2))

    def _row(label: str, value: str) -> None:
        print(f"  {_bold(label):<38} {value}")

    _row("Total cases evaluated:", str(report.total_cases))
    _row("Top-K:", str(report.top_k))
    _row(
        f"Recall@{report.top_k} (files):",
        _bar(report.avg_file_recall),
    )
    _row(
        f"Recall@{report.top_k} (symbols):",
        _bar(report.avg_symbol_recall),
    )

    fhr_colour = (
        _green if report.file_hit_rate >= 70 else (_yellow if report.file_hit_rate >= 40 else _red)
    )
    _row("File Hit Rate:", fhr_colour(f"{report.file_hit_rate:.1f}%"))

    shr_colour = (
        _green
        if report.symbol_hit_rate >= 70
        else (_yellow if report.symbol_hit_rate >= 40 else _red)
    )
    _row("Symbol Hit Rate:", shr_colour(f"{report.symbol_hit_rate:.1f}%"))

    llm_colour = (
        _green if report.avg_llm_score >= 4 else (_yellow if report.avg_llm_score >= 3 else _red)
    )
    _row("Avg LLM Correctness Score:", llm_colour(f"{report.avg_llm_score:.2f} / 5.00"))
    _row("Total search latency:", f"{report.total_latency_ms:.0f} ms")
    _row("Avg latency per query:", f"{report.total_latency_ms / max(report.total_cases, 1):.1f} ms")

    # ── Per-category breakdown ─────────────────────────────────────────
    print()
    print(_bold("  Category Breakdown"))
    print("  " + "─" * (W - 2))

    # Header
    col_w = [20, 7, 12, 12, 10, 10]
    headers = ["Category", "Count", "FileRecall", "SymRecall", "FileHit%", "LLM"]
    header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w, strict=False))
    print(_bold(header_line))
    print("  " + "─" * (W - 2))

    for cat, stats in sorted(report.category_stats.items()):
        fr = stats["avg_file_recall"]
        sr = stats["avg_symbol_recall"]
        fh = stats["file_hit_rate_%"]
        llm = stats["avg_llm_score"]
        row = [
            cat,
            str(int(stats["count"])),
            f"{fr:.2f}",
            f"{sr:.2f}",
            f"{fh:.1f}%",
            f"{llm:.2f}",
        ]
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, col_w, strict=False)))

    # ── Failures ───────────────────────────────────────────────────────
    failures = [r for r in report.case_results if r["file_hit"] == 0]
    if failures:
        print()
        print(
            _bold(
                _red(f"  Failed Cases (file not retrieved) – {len(failures)}/{report.total_cases}")
            )
        )
        print("  " + "─" * (W - 2))
        for r in failures:
            print(
                f"  {_red('✗')} [{r['id']}] {r['category']:15s}  "
                f"score={r['llm_score']:.1f}  "
                f"{r['question'][:48]}…"
            )
    else:
        print()
        print(_green("  ✓ All cases retrieved at least one expected file."))

    print()
    print(_bold(_cyan("─" * W)))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="RepoLens MCP automated evaluation benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python eval/run_eval.py
              python eval/run_eval.py --repo /path/to/repo --top-k 10
              python eval/run_eval.py --output results.json --verbose
            """
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(_HERE / "dataset.json"),
        metavar="PATH",
        help="Path to the benchmark dataset JSON (default: eval/dataset.json).",
    )
    parser.add_argument(
        "--repo",
        default=str(_ROOT),
        metavar="PATH",
        help="Repository to index and evaluate against (default: project root).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        metavar="N",
        help="Number of chunks to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save the full benchmark report as JSON to this path.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-case progress.",
    )
    parser.add_argument(
        "--rule-based-judge",
        action="store_true",
        dest="rule_based",
        help="Force the rule-based judge even if LLM API keys are set.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    print(_bold(_cyan("\n  RepoLens MCP – Automated Evaluation Suite")))
    print(_dim(f"  Dataset : {args.dataset}"))
    print(_dim(f"  Repo    : {args.repo}"))
    print(_dim(f"  Top-K   : {args.top_k}"))

    try:
        cases = load_dataset(args.dataset)
    except FileNotFoundError as exc:
        print(_red(f"\n  Error: {exc}"))
        return 1

    if not cases:
        print(_red("\n  Error: dataset is empty."))
        return 1

    print(_dim(f"  Cases   : {len(cases)}"))

    report = run_evaluation(
        cases=cases,
        repo_path=args.repo,
        top_k=args.top_k,
        force_rule_based=args.rule_based,
        verbose=args.verbose,
    )
    report.dataset_path = args.dataset

    print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(asdict(report), indent=2, default=str),
            encoding="utf-8",
        )
        print(_cyan(f"  Report saved to: {out_path}\n"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
