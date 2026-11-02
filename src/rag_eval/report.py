"""Markdown rendering of experiment results and comparisons."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.results import METRIC_ORDER, ExperimentResult, MetricComparison, MetricSummary

LABELS = {
    "hit@k": "Hit@k",
    "recall@k": "Recall@k",
    "mrr": "MRR",
    "ndcg@k": "nDCG@k",
    "citation_precision": "Cite P",
    "citation_recall": "Cite R",
    "token_f1": "F1",
    "exact_match": "EM",
    "faithfulness": "Faithful",
    "relevance": "Relevant",
}

# Metrics that also get a confidence-interval column in the summary table.
CI_METRICS = ("ndcg@k", "token_f1")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _ci(summary: MetricSummary) -> str:
    return f"[{summary.low:.2f}, {summary.high:.2f}]"


def render_report(result: ExperimentResult) -> str:
    """One row per run, best mean per metric in bold."""
    present = {name for run in result.runs for name in run.summary}
    metrics = [m for m in METRIC_ORDER if m in present]
    best: dict[str, str] = {}
    for metric in metrics:
        means = {f"{r.summary[metric].mean:.3f}" for r in result.runs if metric in r.summary}
        # Bold only when there is something to distinguish.
        if len(means) > 1:
            best[metric] = max(means, key=float)

    header = ["Run", "Chunks"]
    for metric in metrics:
        header.append(LABELS[metric])
        if metric in CI_METRICS:
            header.append(f"{LABELS[metric]} 95% CI")
    rows: list[list[str]] = []
    for run in result.runs:
        row = [f"`{run.id}`", str(run.num_chunks)]
        for metric in metrics:
            summary = run.summary.get(metric)
            if summary is None:
                row.append("-")
                if metric in CI_METRICS:
                    row.append("-")
                continue
            cell = f"{summary.mean:.3f}"
            row.append(f"**{cell}**" if best.get(metric) == cell else cell)
            if metric in CI_METRICS:
                row.append(_ci(summary))
        rows.append(row)

    questions = len(result.runs[0].questions) if result.runs else 0
    return (
        f"## Experiment `{result.name}`\n\n"
        f"{len(result.runs)} runs over {questions} questions. "
        "Retrieval metrics are document-level; k is the number of chunks retrieved. "
        "Best value per column in bold.\n\n" + _table(header, rows) + "\n"
    )


def render_comparison(run_a: str, run_b: str, comparisons: Sequence[MetricComparison]) -> str:
    rows = []
    for c in comparisons:
        r = c.result
        rows.append(
            [
                LABELS[c.metric],
                str(c.questions),
                f"{r.mean_a:.3f}",
                f"{r.mean_b:.3f}",
                f"{r.delta.mean:+.3f}",
                f"[{r.delta.low:+.3f}, {r.delta.high:+.3f}]",
                f"{r.p_value:.3f}",
            ]
        )
    header = ["Metric", "n", "A", "B", "Δ (B - A)", "95% CI of Δ", "p"]
    return (
        f"**A** = `{run_a}`  \n**B** = `{run_b}`\n\n"
        + _table(header, rows)
        + "\n\nΔ CI from a paired bootstrap; p from a two-sided sign-flip permutation test.\n"
    )
