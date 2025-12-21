from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Сколько top-значений сохранять для категориальных признаков."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта (первая строка report.md)."),
    min_missing_share: float = typer.Option(
        0.10,
        help="Порог доли пропусков: выше порога колонка считается проблемной и попадает в список в отчёте.",
        min=0.0,
        max=1.0,
    ),
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # важно: df передаем, чтобы новые эвристики считались
    quality_flags = compute_quality_flags(summary, missing_df, df=df)

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # использование min_missing_share
    problem_missing_df = pd.DataFrame()
    if not missing_df.empty:
        problem_missing_df = missing_df[missing_df["missing_share"] >= float(min_missing_share)].copy()
        if not problem_missing_df.empty:
            problem_missing_df.to_csv(out_root / "problem_missing.csv", index=True)

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Параметры генерации отчёта\n\n")
        f.write(f"- max_hist_columns = **{max_hist_columns}**\n")
        f.write(f"- top_k_categories = **{top_k_categories}**\n")
        f.write(f"- min_missing_share = **{min_missing_share:.0%}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- quality_score: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- max_missing_share: **{quality_flags['max_missing_share']:.2%}**\n\n")

        f.write("### Дополнительные эвристики (HW03)\n\n")
        const_cols = quality_flags.get("constant_columns", [])
        f.write(f"- constant_columns: {', '.join(const_cols) if const_cols else 'нет'}\n")

        high_cols = quality_flags.get("high_cardinality_categoricals", [])
        thr = quality_flags.get("high_cardinality_threshold", 50)
        f.write(f"- high_cardinality_categoricals (threshold={thr}): {', '.join(high_cols) if high_cols else 'нет'}\n")

        id_col = quality_flags.get("id_column_checked", "user_id")
        id_exists = quality_flags.get("id_column_exists", False)
        if not id_exists:
            f.write(f"- suspicious_id_duplicates: колонка {id_col} не найдена (проверка пропущена)\n")
        else:
            f.write(f"- suspicious_id_duplicates (id_column={id_col}): {quality_flags.get('suspicious_id_duplicates', False)}\n")

        zero_flags = quality_flags.get("many_zero_values", {})  # dict[str,bool]
        bad_zero_cols = [c for c, v in zero_flags.items() if v]
        zthr = quality_flags.get("zero_share_threshold", 0.5)
        if bad_zero_cols:
            f.write(f"- many_zero_values (threshold={zthr:.0%}): {', '.join(bad_zero_cols)}\n")
        else:
            f.write(f"- many_zero_values (threshold={zthr:.0%}): нет\n")

        f.write("\n## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. `missing.csv` и `missing_matrix.png`.\n\n")
            f.write("### Проблемные колонки по пропускам (HW03)\n\n")
            if problem_missing_df.empty:
                f.write(f"Колонок с missing_share >= {min_missing_share:.0%} не найдено.\n\n")
            else:
                f.write(f"Порог: **{min_missing_share:.0%}**\n\n")
                f.write("Проблемные колонки:\n\n")
                for col in problem_missing_df.index.astype(str).tolist():
                    f.write(f"- {col}\n")
                f.write("\nТакже сохранено в `problem_missing.csv`.\n\n")

        f.write("## Корреляция\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные признаки не найдены.\n\n")
        else:
            f.write(f"Сохранены top-{top_k_categories} значения (см. `top_categories/`).\n\n")

        f.write("## Гистограммы\n\n")
        f.write("См. `hist_*.png`.\n")

    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Markdown: {md_path}")
    typer.echo("- Таблицы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    if not problem_missing_df.empty:
        typer.echo("- Дополнительно: problem_missing.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
