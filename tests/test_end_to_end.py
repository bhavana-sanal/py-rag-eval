import json
from pathlib import Path

import httpx
import pytest

from rag_eval.cli import main
from rag_eval.config import GridConfig, JudgeConfig, load_experiment
from rag_eval.report import render_report
from rag_eval.results import ExperimentResult, compare_runs
from rag_eval.runner import run_experiment

RETRIEVAL = {"hit@k", "recall@k", "mrr", "ndcg@k"}
GENERATION = {"citation_precision", "citation_recall", "token_f1", "exact_match"}


def test_example_experiment_runs_offline_and_is_deterministic(example_config: Path) -> None:
    config = load_experiment(example_config)
    result = run_experiment(config)

    assert len(result.runs) == 24
    for run in result.runs:
        assert set(run.summary) == RETRIEVAL | GENERATION
        assert len(run.questions) == 43
        for summary in run.summary.values():
            assert 0.0 <= summary.low <= summary.mean <= summary.high <= 1.0
        for question in run.questions:
            assert len(question.retrieved) <= run.k

    # Retrieval on this corpus should be far better than chance (k / 26 documents).
    best = max(run.summary["recall@k"].mean for run in result.runs)
    assert best > 0.8
    assert run_experiment(config) == result


def test_judge_scores_are_added_and_normalised(example_config: Path, tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        verdict = '{"faithfulness": 5, "relevance": 3, "reasoning": "ok"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": verdict}}]})

    config = load_experiment(example_config).model_copy(
        update={
            "grid": GridConfig(chunker=["recursive"], retriever=["bm25"], k=[3]),
            "judge": JudgeConfig(
                model="judge-model", base_url="http://judge.test/v1", cache_dir=tmp_path
            ),
        }
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    (run,) = run_experiment(config, http_client=client).runs

    assert run.summary["faithfulness"].mean == 1.0
    assert run.summary["relevance"].mean == 0.5
    assert calls == 43
    run_experiment(config, http_client=client)
    assert calls == 43, "second run should be served entirely from the judge cache"


def test_unknown_labelled_documents_are_rejected(example_config: Path, tmp_path: Path) -> None:
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"id": "q", "question": "?", "relevant": ["missing"]}))
    config = load_experiment(example_config).model_copy(update={"questions": questions})
    with pytest.raises(ValueError, match="unknown documents"):
        run_experiment(config)


def test_cli_run_and_compare(
    example_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(example_config), "--out", str(tmp_path)]) == 0
    results = tmp_path / "example.json"
    report = (tmp_path / "example.md").read_text()
    assert "| `recursive/hybrid/k=5` |" in report
    assert report in capsys.readouterr().out

    args = ["compare", str(results), str(results), "--resamples", "500"]
    run_args = ["--run-a", "fixed/bm25/k=3", "--run-b", "fixed/hybrid/k=5"]
    assert main(args + run_args) == 0
    out = capsys.readouterr().out
    assert "| nDCG@k | 43 |" in out

    assert main(args) == 1
    assert "choose one" in capsys.readouterr().err


def test_compare_runs_pairs_by_question_id(example_config: Path) -> None:
    result = run_experiment(load_experiment(example_config))
    a, b = result.run("fixed/bm25/k=3"), result.run("recursive/hybrid/k=5")
    trimmed = b.model_copy(update={"questions": b.questions[:10]})
    comparisons = {c.metric: c for c in compare_runs(a, trimmed, resamples=200)}
    assert comparisons["ndcg@k"].questions == 10
    self_compare = compare_runs(a, a, resamples=200)
    assert all(c.result.delta.mean == 0 and c.result.p_value == 1 for c in self_compare)


def test_results_round_trip_through_json(example_config: Path) -> None:
    result = run_experiment(load_experiment(example_config))
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert render_report(restored) == render_report(result)
