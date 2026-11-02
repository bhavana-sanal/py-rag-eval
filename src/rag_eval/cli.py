"""Command-line interface: ``rag-eval run`` and ``rag-eval compare``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_eval.config import RunSpec, load_experiment
from rag_eval.report import render_comparison, render_report
from rag_eval.results import ExperimentResult, compare_runs
from rag_eval.runner import run_experiment


def _progress(position: int, total: int, spec: RunSpec) -> None:
    print(f"[{position}/{total}] {spec.id}", file=sys.stderr)


def _run(args: argparse.Namespace) -> None:
    config = load_experiment(args.config)
    result = run_experiment(config, on_run=_progress)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{config.name}.json"
    md_path = args.out / f"{config.name}.md"
    json_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report = render_report(result)
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {json_path} and {md_path}", file=sys.stderr)


def _load(path: Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json(path.read_text(encoding="utf-8"))


def _compare(args: argparse.Namespace) -> None:
    run_a = _load(args.a).run(args.run_a)
    run_b = _load(args.b).run(args.run_b)
    comparisons = compare_runs(run_a, run_b, resamples=args.resamples, seed=args.seed)
    print(render_comparison(run_a.id, run_b.id, comparisons))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval", description="Build, measure and compare RAG pipelines."
    )
    commands = parser.add_subparsers(required=True, metavar="COMMAND")

    run = commands.add_parser("run", help="run an experiment grid from a TOML file")
    run.add_argument("config", type=Path, help="experiment TOML file")
    run.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    run.set_defaults(handler=_run)

    compare = commands.add_parser("compare", help="paired comparison of two runs")
    compare.add_argument("a", type=Path, help="results JSON for system A")
    compare.add_argument("b", type=Path, help="results JSON for system B (may equal A)")
    compare.add_argument("--run-a", help="run id within A (needed if A has several runs)")
    compare.add_argument("--run-b", help="run id within B (needed if B has several runs)")
    compare.add_argument("--resamples", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (OSError, KeyError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"rag-eval: error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
