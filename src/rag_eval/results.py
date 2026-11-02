"""Serialisable experiment results and paired comparison between runs."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from rag_eval.stats import PairedComparison, paired_comparison

# Display order for reports; metrics absent from a run are skipped.
METRIC_ORDER = (
    "hit@k",
    "recall@k",
    "mrr",
    "ndcg@k",
    "citation_precision",
    "citation_recall",
    "token_f1",
    "exact_match",
    "faithfulness",
    "relevance",
)


class MetricSummary(BaseModel):
    mean: float
    low: float
    high: float


class QuestionResult(BaseModel):
    question_id: str
    retrieved: list[str]
    """Retrieved document ids in rank order."""
    answer: str
    metrics: dict[str, float]


class RunResult(BaseModel):
    id: str
    chunker: str
    retriever: str
    k: int
    num_chunks: int
    summary: dict[str, MetricSummary]
    questions: list[QuestionResult]


class ExperimentResult(BaseModel):
    name: str
    version: str
    config: dict[str, object]
    runs: list[RunResult]

    def run(self, run_id: str | None) -> RunResult:
        """Look up a run by id; with no id, the result must contain exactly one run."""
        if run_id is None:
            if len(self.runs) == 1:
                return self.runs[0]
            raise KeyError(f"{self.name!r} has {len(self.runs)} runs; choose one: {self._ids()}")
        for run in self.runs:
            if run.id == run_id:
                return run
        raise KeyError(f"no run {run_id!r} in {self.name!r}; available: {self._ids()}")

    def _ids(self) -> str:
        return ", ".join(run.id for run in self.runs)


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    questions: int
    result: PairedComparison


def compare_runs(
    a: RunResult, b: RunResult, *, resamples: int = 10_000, seed: int = 0
) -> list[MetricComparison]:
    """Paired comparison on every metric both runs report, over their shared questions."""
    by_id_a = {q.question_id: q.metrics for q in a.questions}
    by_id_b = {q.question_id: q.metrics for q in b.questions}
    shared = [qid for qid in by_id_a if qid in by_id_b]
    if not shared:
        raise ValueError(f"runs {a.id!r} and {b.id!r} share no questions")
    comparisons: list[MetricComparison] = []
    for metric in METRIC_ORDER:
        paired = [
            (by_id_a[qid][metric], by_id_b[qid][metric])
            for qid in shared
            if metric in by_id_a[qid] and metric in by_id_b[qid]
        ]
        if paired:
            xs, ys = zip(*paired, strict=True)
            comparisons.append(
                MetricComparison(
                    metric, len(paired), paired_comparison(xs, ys, resamples=resamples, seed=seed)
                )
            )
    return comparisons
