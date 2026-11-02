"""Executes an experiment grid and scores every run."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from importlib.metadata import version

import httpx

from rag_eval.chunking import chunk_corpus
from rag_eval.config import ExperimentConfig, RunSpec, expand_grid
from rag_eval.generation import Generator
from rag_eval.ingest import load_corpus, load_questions
from rag_eval.judge import LLMJudge
from rag_eval.metrics import (
    citation_precision,
    citation_recall,
    exact_match,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from rag_eval.pipeline import build_chunker, build_generator, build_judge, build_retriever
from rag_eval.results import ExperimentResult, MetricSummary, QuestionResult, RunResult
from rag_eval.retrievers import Retriever
from rag_eval.schema import Chunk, Question
from rag_eval.stats import bootstrap_ci


def evaluate_question(
    question: Question,
    retriever: Retriever,
    generator: Generator,
    k: int,
    judge: LLMJudge | None = None,
) -> QuestionResult:
    passages = [hit.chunk for hit in retriever.retrieve(question.question, k)]
    # Metrics are document-level: several chunks of one relevant document count once,
    # which keeps scores comparable across chunkers that cut documents differently.
    ranked = list(dict.fromkeys(p.doc_id for p in passages))
    relevant = question.relevant
    answer = generator.generate(question.question, passages)
    cited = [passages[n - 1].doc_id if 1 <= n <= len(passages) else None for n in answer.citations]

    metrics = {
        "hit@k": hit_at_k(ranked, relevant, k),
        "recall@k": recall_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        "ndcg@k": ndcg_at_k(ranked, relevant, k),
        "citation_precision": citation_precision(cited, relevant),
        "citation_recall": citation_recall(cited, relevant),
    }
    if question.answer is not None:
        metrics["token_f1"] = token_f1(answer.text, question.answer)
        metrics["exact_match"] = exact_match(answer.text, question.answer)
    if judge is not None:
        verdict = judge.judge(question.question, passages, answer.text)
        # Map the 1-5 rubric onto [0, 1] so every metric shares a scale.
        metrics["faithfulness"] = (verdict.faithfulness - 1) / 4
        metrics["relevance"] = (verdict.relevance - 1) / 4
    return QuestionResult(
        question_id=question.id, retrieved=ranked, answer=answer.text, metrics=metrics
    )


def summarize(
    results: Sequence[QuestionResult], *, resamples: int, seed: int
) -> dict[str, MetricSummary]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in result.metrics.items():
            values[name].append(value)
    summary: dict[str, MetricSummary] = {}
    for name, sample in values.items():
        ci = bootstrap_ci(sample, resamples=resamples, seed=seed)
        summary[name] = MetricSummary(mean=ci.mean, low=ci.low, high=ci.high)
    return summary


def run_experiment(
    config: ExperimentConfig,
    *,
    http_client: httpx.Client | None = None,
    on_run: Callable[[int, int, RunSpec], None] | None = None,
) -> ExperimentResult:
    docs = load_corpus(config.corpus)
    questions = load_questions(config.questions)
    doc_ids = {doc.id for doc in docs}
    for question in questions:
        unknown = set(question.relevant) - doc_ids
        if unknown:
            raise ValueError(f"question {question.id!r} labels unknown documents: {unknown}")

    generator = build_generator(config.generator, http_client)
    judge = build_judge(config.judge, http_client) if config.judge else None
    chunk_cache: dict[str, list[Chunk]] = {}
    # The k axis is innermost in the grid, so keeping only the latest index means each
    # (chunker, retriever) pair is indexed exactly once.
    current: tuple[tuple[str, str], Retriever] | None = None
    specs = expand_grid(config)
    runs: list[RunResult] = []
    for position, spec in enumerate(specs, start=1):
        if on_run:
            on_run(position, len(specs), spec)
        if spec.chunker not in chunk_cache:
            chunker = build_chunker(config.chunkers[spec.chunker])
            chunk_cache[spec.chunker] = chunk_corpus(chunker, docs)
        chunks = chunk_cache[spec.chunker]
        key = (spec.chunker, spec.retriever)
        if current is None or current[0] != key:
            retriever = build_retriever(spec.retriever, config)
            retriever.index(chunks)
            current = (key, retriever)
        per_question = [
            evaluate_question(q, current[1], generator, spec.k, judge) for q in questions
        ]
        runs.append(
            RunResult(
                id=spec.id,
                chunker=spec.chunker,
                retriever=spec.retriever,
                k=spec.k,
                num_chunks=len(chunks),
                summary=summarize(
                    per_question, resamples=config.bootstrap_resamples, seed=config.seed
                ),
                questions=per_question,
            )
        )
    return ExperimentResult(
        name=config.name,
        version=version("py-rag-eval"),
        config=config.model_dump(mode="json"),
        runs=runs,
    )
