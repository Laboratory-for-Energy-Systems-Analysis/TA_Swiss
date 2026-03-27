from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


REGIONS = ["EUR", "USA", "CHA"]
IMPACT_ORDER = [
    "climate change",
    "human health",
    "ecosystem quality",
    "land occupation",
    "net use of fresh water",
    "minerals depletion",
]
KWH_PER_EJ = 1e18 / 3.6e6
LEFT_LABELS = {
    "SE - electricity - Wind Onshore": "Wind onshore",
    "SE - electricity - Solar PV Centralized": "Solar PV",
}


def short_label(variable: str) -> str:
    return variable.replace("SE - electricity - ", "")


def add_stack_label(axis, stack: pd.DataFrame, variables: list[str], variable: str, label: str) -> None:
    if variable not in stack.columns:
        return

    series = stack[variable]
    if series.max() <= 0:
        return

    year = int(series.idxmax())
    variable_index = variables.index(variable)
    lower = stack.iloc[:, :variable_index].sum(axis=1).loc[year] if variable_index > 0 else 0.0
    height = series.loc[year]
    y = lower + height / 2

    axis.text(
        year,
        y,
        label,
        ha="center",
        va="center",
        fontsize=13,
        color="black",
        path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.9)],
    )


def load_production_volumes(path: Path) -> pd.DataFrame:
    production = pd.read_csv(path, header=[0, 1])
    production.columns = [
        top if str(bottom).startswith("Unnamed") else str(bottom)
        for top, bottom in production.columns
    ]

    year_columns = [column for column in production.columns if str(column).isdigit()]
    production = production.rename(columns={"variables": "variable"})
    production = production[production["region"].isin(REGIONS)].copy()

    production = production.melt(
        id_vars=["model", "pathway", "variable", "region"],
        value_vars=year_columns,
        var_name="year",
        value_name="production_ej",
    )
    production["year"] = production["year"].astype(int)
    production["production_ej"] = pd.to_numeric(production["production_ej"], errors="coerce")

    return production[["region", "variable", "year", "production_ej"]]


def aggregate_impacts(df_path: Path, cache_path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    if cache_path.exists() and (not df_path.exists() or cache_path.stat().st_mtime >= df_path.stat().st_mtime):
        print(f"Using cached impact totals from {cache_path}")
        impacts = pd.read_csv(cache_path)
        impacts["year"] = impacts["year"].astype(int)
        return impacts

    print(f"Aggregating impacts from {df_path}")
    chunk_totals: list[pd.DataFrame] = []

    for index, chunk in enumerate(
        pd.read_csv(
            df_path,
            usecols=["variable", "year", "region", "impact_category", "value"],
            chunksize=chunksize,
        ),
        start=1,
    ):
        chunk = chunk[
            chunk["region"].isin(REGIONS) & chunk["impact_category"].isin(IMPACT_ORDER)
        ].copy()

        grouped = (
            chunk.groupby(
                ["region", "variable", "year", "impact_category"],
                as_index=False,
                observed=True,
            )["value"]
            .sum()
        )
        chunk_totals.append(grouped)
        print(f"  processed chunk {index}")

    impacts = (
        pd.concat(chunk_totals, ignore_index=True)
        .groupby(
            ["region", "variable", "year", "impact_category"],
            as_index=False,
            observed=True,
        )["value"]
        .sum()
    )
    impacts["year"] = impacts["year"].astype(int)
    impacts.to_csv(cache_path, index=False)
    print(f"Saved impact totals cache to {cache_path}")
    return impacts


def build_plot_tables(production: pd.DataFrame, impacts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    climate = (
        impacts[impacts["impact_category"] == "climate change"]
        .rename(columns={"value": "climate_kg"})
        .drop(columns=["impact_category"])
    )

    climate = production.merge(
        climate,
        on=["region", "variable", "year"],
        how="left",
    )

    zero_production = climate["production_ej"].fillna(0).eq(0)
    climate.loc[zero_production & climate["climate_kg"].isna(), "climate_kg"] = 0.0

    missing_positive = climate[climate["production_ej"].gt(0) & climate["climate_kg"].isna()]
    if not missing_positive.empty:
        print(
            "Warning: found positive production rows with missing climate impacts: "
            f"{len(missing_positive)}"
        )

    climate["climate_kg"] = climate["climate_kg"].fillna(0.0)

    climate["climate_gt"] = climate["climate_kg"] / 1e12
    climate["climate_per_kwh"] = np.where(
        climate["production_ej"].gt(0),
        climate["climate_kg"] / (climate["production_ej"] * KWH_PER_EJ),
        np.nan,
    )

    total_impacts = (
        impacts.groupby(["region", "year", "impact_category"], as_index=False, observed=True)["value"]
        .sum()
        .rename(columns={"value": "impact_total"})
    )

    baseline_2020 = (
        total_impacts[total_impacts["year"] == 2020][["region", "impact_category", "impact_total"]]
        .rename(columns={"impact_total": "impact_2020"})
    )

    indexed_impacts = total_impacts.merge(
        baseline_2020,
        on=["region", "impact_category"],
        how="left",
    )
    indexed_impacts["index_2020"] = np.where(
        indexed_impacts["impact_2020"].ne(0),
        indexed_impacts["impact_total"] / indexed_impacts["impact_2020"] * 100,
        np.nan,
    )

    return climate, indexed_impacts


def make_figure(
    production: pd.DataFrame,
    climate: pd.DataFrame,
    indexed_impacts: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 16,
        }
    )

    variables = sorted(production["variable"].unique())
    years = sorted(production["year"].unique())
    x_ticks = years[::2]

    variable_colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    variable_colors.append(plt.cm.tab20b(0))
    variable_color_map = dict(zip(variables, variable_colors, strict=True))

    category_colors = plt.cm.Dark2(np.linspace(0, 1, len(IMPACT_ORDER)))
    category_color_map = dict(zip(IMPACT_ORDER, category_colors, strict=True))

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(24, 15),
        sharex=True,
    )

    production_max = production.groupby(["region", "year"], observed=True)["production_ej"].sum().max()
    climate_max = climate.groupby(["region", "year"], observed=True)["climate_gt"].sum().max()
    intensity_max = (
        climate.groupby(["region", "year"], observed=True)[["climate_kg", "production_ej"]]
        .sum()
        .assign(
            climate_per_kwh_total=lambda df: np.where(
                df["production_ej"].gt(0),
                df["climate_kg"] / (df["production_ej"] * KWH_PER_EJ),
                np.nan,
            )
        )["climate_per_kwh_total"]
        .max()
    )

    production_ylim = (0, production_max * 1.05 if pd.notna(production_max) else 1)
    climate_ylim = (0, climate_max * 1.05 if pd.notna(climate_max) else 1)
    intensity_ylim = (0, intensity_max * 1.05 if pd.notna(intensity_max) else 1)

    column_titles = [
        "Production Volumes",
        "Climate Change Impact",
        "Impacts Relative To 2020",
    ]
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=22, fontweight="bold")

    for row_index, region in enumerate(REGIONS):
        production_region = production[production["region"] == region]
        climate_region = climate[climate["region"] == region]
        indexed_region = indexed_impacts[indexed_impacts["region"] == region]

        left_axis = axes[row_index, 0]
        middle_axis = axes[row_index, 1]
        right_axis = axes[row_index, 2]
        middle_axis_secondary = middle_axis.twinx()

        production_stack = (
            production_region.pivot(index="year", columns="variable", values="production_ej")
            .reindex(index=years, columns=variables)
            .fillna(0.0)
        )
        climate_stack = (
            climate_region.pivot(index="year", columns="variable", values="climate_gt")
            .reindex(index=years, columns=variables)
            .fillna(0.0)
        )
        total_intensity = (
            climate_region.groupby("year", as_index=True)[["climate_kg", "production_ej"]]
            .sum()
            .reindex(years)
            .fillna(0.0)
        )
        total_intensity["climate_per_kwh_total"] = np.where(
            total_intensity["production_ej"].gt(0),
            total_intensity["climate_kg"] / (total_intensity["production_ej"] * KWH_PER_EJ),
            np.nan,
        )

        left_axis.stackplot(
            years,
            production_stack.T.to_numpy(),
            colors=[variable_color_map[variable] for variable in variables],
            alpha=0.9,
            linewidth=0.0,
        )
        for variable, label in LEFT_LABELS.items():
            add_stack_label(left_axis, production_stack, variables, variable, label)
        middle_axis.stackplot(
            years,
            climate_stack.T.to_numpy(),
            colors=[variable_color_map[variable] for variable in variables],
            alpha=0.9,
            linewidth=0.0,
        )
        middle_axis_secondary.plot(
            years,
            total_intensity["climate_per_kwh_total"].to_numpy(),
            color="0.1",
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
        )

        for impact_category in IMPACT_ORDER:
            category_series = (
                indexed_region[indexed_region["impact_category"] == impact_category]
                .sort_values("year")
            )
            right_axis.plot(
                category_series["year"],
                category_series["index_2020"],
                color=category_color_map[impact_category],
                linewidth=3.2,
                label=impact_category,
            )

        right_axis.axhline(100, color="0.45", linewidth=1.0, linestyle=":")

        left_axis.annotate(
            region,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=18,
            fontweight="bold",
        )

        left_axis.set_ylabel("Production [EJ]")
        middle_axis.set_ylabel("Climate Impact [Gt CO2-eq.]")
        middle_axis_secondary.set_ylabel("Intensity [kg CO2-eq./kWh]")
        right_axis.set_ylabel("Index (2020 = 100)")

        left_axis.set_ylim(production_ylim)
        middle_axis.set_ylim(climate_ylim)
        middle_axis_secondary.set_ylim(intensity_ylim)

        left_axis.set_xticks(x_ticks)
        middle_axis.set_xticks(x_ticks)
        right_axis.set_xticks(x_ticks)

        left_axis.set_xlim(min(years), max(years))
        middle_axis.set_xlim(min(years), max(years))
        right_axis.set_xlim(min(years), max(years))

        for axis in [left_axis, middle_axis, middle_axis_secondary, right_axis]:
            axis.grid(True, alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("Year")

    technology_handles = [
        Line2D([0], [0], color=variable_color_map[variable], linewidth=2, label=short_label(variable))
        for variable in variables
    ]
    category_handles = [
        Line2D([0], [0], color=category_color_map[category], linewidth=3.2, label=category)
        for category in IMPACT_ORDER
    ]

    fig.legend(
        handles=technology_handles,
        loc="lower center",
        bbox_to_anchor=(0.30, 0.02),
        ncol=4,
        fontsize=14,
        frameon=False,
        title="Electricity Variables",
    )
    fig.legend(
        handles=category_handles,
        loc="lower center",
        bbox_to_anchor=(0.74, 0.02),
        ncol=3,
        fontsize=14,
        frameon=False,
        title="Impact Categories",
    )

    fig.tight_layout(rect=[0.04, 0.13, 0.98, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 3x3 electricity impact figure.")
    parser.add_argument(
        "--impacts",
        type=Path,
        default=Path("df_final.csv"),
        help="Path to the impact CSV export.",
    )
    parser.add_argument(
        "--production",
        type=Path,
        default=Path("production_volumes.csv"),
        help="Path to the production volumes CSV export.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("impact_totals_cache.csv"),
        help="Path to the cached annual impact totals CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("electricity_impacts_3x3.png"),
        help="Path to the output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    production = load_production_volumes(args.production)
    impacts = aggregate_impacts(args.impacts, args.cache)
    climate, indexed_impacts = build_plot_tables(production, impacts)
    make_figure(production, climate, indexed_impacts, args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
