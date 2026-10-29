"""Opt-in, terminal-based walkthrough from analysis to local model training."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from datalens.training import TrainingResult


class _ExitWorkflow(Exception):
    pass


class _SkipTraining(Exception):
    pass


def _choose(
    title: str, options: dict[str, str], *, default: str,
    ask: Callable[[str], str], show: Callable[[str], None],
) -> str:
    while True:
        show(title)
        for key, label in options.items():
            show(f"  {key}) {label}")
        show("  x) Exit workflow")
        answer = ask(f"Choice [{default}]: ").strip().lower() or default
        if answer in {"x", "exit", "q", "quit"}:
            raise _ExitWorkflow
        if answer in options:
            return answer
        show("Please enter one of the listed choices.")


def _select_target(
    columns: list[str], proposed: str | None, *,
    ask: Callable[[str], str], show: Callable[[str], None],
) -> str:
    if proposed is not None:
        if proposed not in columns:
            raise ValueError(f"Target column {proposed!r} was not found")
        choice = _choose(
            f"Use {proposed!r} as the column to predict?",
            {"y": "Yes", "s": "Skip training"}, default="y", ask=ask, show=show,
        )
        if choice == "s":
            raise _SkipTraining
        return proposed
    show("Choose the target column to predict (you must confirm it):")
    for number, name in enumerate(columns, 1):
        show(f"  {number}) {name}")
    while True:
        answer = ask("Column name or number [x to exit]: ").strip()
        if answer.lower() in {"x", "exit", "q", "quit"}:
            raise _ExitWorkflow
        if answer in columns:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(columns):
            return columns[int(answer) - 1]
        show("Enter an exact column name or one of the listed numbers.")


def _destination(
    suggested: Path, *, ask: Callable[[str], str], show: Callable[[str], None]
) -> Path | None:
    if suggested.suffix.lower() != ".joblib":
        raise ValueError("model output must end in .joblib")
    destination = suggested
    while destination.exists() or destination.with_suffix(".json").exists():
        show(f"Output already exists: {destination}")
        answer = ask("Enter a new .joblib path, or s to skip saving, x to exit: ").strip()
        if answer.lower() in {"x", "exit", "q", "quit"}:
            raise _ExitWorkflow
        if answer.lower() in {"s", "skip"}:
            return None
        destination = Path(answer)
        if destination.suffix.lower() != ".joblib":
            show("The model path must end in .joblib.")
            destination = suggested
    return destination


def guide_dataframe(
    dataframe: pd.DataFrame, *, target: str | None = None,
    output: str | Path | None = None, source_name: str = "dataset",
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> TrainingResult | None:
    """Show profile, chart ideas, cleaning choices, and optional training.

    Interactive choices affect a fitted pipeline; the supplied DataFrame is
    never changed. A skipped or exited training stage returns None.
    """
    from datalens.api import analyse, train
    from datalens.cli import format_report, format_training_result

    ask = input_fn or input
    show = output_fn or print
    report = analyse(dataframe)
    show(format_report(report))

    try:
        if report.recommendations:
            choice = _choose(
                f"View all {len(report.recommendations)} chart ideas?",
                {"y": "Yes", "s": "Skip"}, default="y", ask=ask, show=show,
            )
            if choice == "y":
                for item in report.recommendations:
                    show(f"  {item.chart_type}: {item.title} [{', '.join(item.columns)}]")
                    show(f"    Why: {item.reason}")

        choice = _choose(
            "Continue from data analysis to guided classification training?",
            {"y": "Yes", "s": "Skip training"}, default="y", ask=ask, show=show,
        )
        if choice == "s":
            show("Analysis complete. Training skipped.")
            return None

        selected_target = _select_target(
            list(dataframe.columns), target, ask=ask, show=show
        )
        drop_duplicates = False
        if report.duplicate_row_count:
            choice = _choose(
                f"Found {report.duplicate_row_count} exact duplicate row(s). Drop them before splitting?",
                {"y": "Yes (recommended: avoids duplicates crossing train/test)",
                 "s": "Skip; keep all rows"},
                default="y", ask=ask, show=show,
            )
            drop_duplicates = choice == "y"

        numeric_imputers: dict[str, str] = {}
        categorical_imputers: dict[str, str] = {}
        excluded_columns: list[str] = []
        for suggestion in report.cleaning_suggestions:
            name = suggestion.column
            if name is None or name == selected_target:
                continue
            if suggestion.recommended_action == "exclude_column":
                show(f"{suggestion.reason} Excluding this feature from training.")
                excluded_columns.append(name)
                continue
            show(suggestion.reason)
            if suggestion.recommended_action == "median":
                choice = _choose(
                    f"Numeric cleaning for {name}:",
                    {"1": "Median imputation (recommended)",
                     "2": "Mean imputation", "s": "Skip this column"},
                    default="1", ask=ask, show=show,
                )
                if choice == "s":
                    excluded_columns.append(name)
                else:
                    numeric_imputers[name] = "median" if choice == "1" else "mean"
            else:
                choice = _choose(
                    f"Categorical cleaning for {name}:",
                    {"1": "Separate missing category (recommended)",
                     "2": "Most frequent value", "s": "Skip this column"},
                    default="1", ask=ask, show=show,
                )
                if choice == "s":
                    excluded_columns.append(name)
                else:
                    categorical_imputers[name] = "constant" if choice == "1" else "most_frequent"

        for issue in report.quality_issues:
            if issue.code == "mixed_types" and selected_target not in issue.columns:
                show(f"Review: {issue.message} Values are converted to text before categorical encoding.")

        numeric_features = [
            item.name for item in report.columns
            if item.kind == "numeric" and item.name != selected_target
            and item.name not in excluded_columns
        ]
        scaler = "standard"
        if numeric_features:
            choice = _choose(
                "Scaling for logistic regression (random forest does not use scaling):",
                {"1": "StandardScaler (recommended)", "2": "MinMaxScaler",
                 "3": "RobustScaler", "4": "No scaling"},
                default="1", ask=ask, show=show,
            )
            scaler = {"1": "standard", "2": "minmax", "3": "robust", "4": "none"}[choice]

        choice = _choose(
            "Model selection:",
            {"1": "Compare logistic regression and random forest (recommended)",
             "2": "Logistic regression only", "3": "Random forest only"},
            default="1", ask=ask, show=show,
        )
        model = {"1": "auto", "2": "logistic_regression", "3": "random_forest"}[choice]
        show(
            f"Plan: target={selected_target}, model={model}, scaler={scaler}, "
            f"drop_duplicates={drop_duplicates}, skipped_columns={excluded_columns or 'none'}"
        )
        show(f"Numeric imputers: {numeric_imputers or 'default median where needed'}")
        show(f"Categorical imputers: {categorical_imputers or 'default missing category where needed'}")
        choice = _choose(
            "Run training with these choices?",
            {"y": "Yes", "s": "Skip training"}, default="y", ask=ask, show=show,
        )
        if choice == "s":
            show("Training skipped. The input data was not changed.")
            return None

        result = train(
            dataframe, target=selected_target, numeric_imputers=numeric_imputers,
            categorical_imputers=categorical_imputers,
            exclude_columns=excluded_columns, scaler=scaler, model=model,
            drop_duplicates=drop_duplicates,
        )
        show(format_training_result(result))
        safe_target = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in selected_target
        )
        source = Path(source_name)
        suggested = Path(output) if output is not None else source.with_name(
            f"{source.stem}-{safe_target}.joblib"
        )
        choice = _choose(
            f"Save the fitted pipeline and JSON run report to {suggested}?",
            {"y": "Yes", "s": "Skip saving"}, default="y", ask=ask, show=show,
        )
        if choice == "y":
            destination = _destination(suggested, ask=ask, show=show)
            if destination is not None:
                saved = result.save(destination)
                show(f"Saved pipeline: {saved}")
                show(f"Run report: {saved.with_suffix('.json')}")
        return result
    except _SkipTraining:
        show("Analysis complete. Training skipped.")
        return None
    except (EOFError, KeyboardInterrupt, _ExitWorkflow):
        show("Workflow exited. The input data was not changed.")
        return None
