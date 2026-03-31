from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("BRIGHTWAY2_DIR", str((Path.cwd() / ".brightway-profile").resolve()))
os.environ.setdefault(
    "BRIGHTWAY2_OUTPUT_DIR",
    str((Path.cwd() / ".brightway-profile" / "output").resolve()),
)
os.environ.setdefault("BRIGHTWAY_NO_STRUCTLOG", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pv_subshares_sensitivity_workflow import (
    DATAPACKAGE,
    PV_TECH_ORDER,
    REGIONS,
    SUBSHARES,
    SOLAR_VARIABLE,
    get_share_trajectories,
    get_production_volumes,
    get_years_and_scenario,
    initialize_pathways,
)


def build_global_shares(shares: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    merged = shares.merge(
        production[["region", "year", "production_ej"]],
        on=["region", "year"],
        how="left",
    )
    merged["production_ej"] = merged["production_ej"].fillna(0.0)
    merged["weighted_share"] = merged["share"] * merged["production_ej"]

    totals = (
        merged.groupby(["year", "iteration"], as_index=False)["production_ej"]
        .sum()
        .rename(columns={"production_ej": "total_production_ej"})
    )
    merged = merged.merge(totals, on=["year", "iteration"], how="left")

    global_weighted = (
        merged.groupby(["year", "technology", "iteration"], as_index=False)[
            ["weighted_share", "production_ej"]
        ]
        .sum()
    )
    global_weighted["share"] = np.where(
        global_weighted["production_ej"].gt(0),
        global_weighted["weighted_share"] / global_weighted["production_ej"],
        np.nan,
    )
    global_weighted["region"] = "Global"

    return global_weighted[["region", "year", "technology", "iteration", "share"]]


def summarize_shares(shares: pd.DataFrame) -> pd.DataFrame:
    summary = (
        shares.groupby(["region", "year", "technology"])["share"]
        .agg(
            q05=lambda x: x.quantile(0.05),
            median="median",
            q95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    return summary


def make_figure(summary: pd.DataFrame, years: list[int], iterations: int, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    colors = plt.cm.YlGnBu(np.linspace(0.25, 0.95, len(PV_TECH_ORDER)))
    color_map = dict(zip(PV_TECH_ORDER, colors, strict=True))

    fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.6), sharex=True, sharey=True)
    is_single_year = len(years) == 1

    x_ticks = years if is_single_year else years[::2]

    region_summary = summary[summary["region"] == "Global"]

    for technology in PV_TECH_ORDER:
        tech_summary = (
            region_summary[region_summary["technology"] == technology]
            .set_index("year")
            .reindex(years)
        )
        x_values = np.asarray(years, dtype=float)
        q05 = tech_summary["q05"].to_numpy(dtype=float)
        median = tech_summary["median"].to_numpy(dtype=float)
        q95 = tech_summary["q95"].to_numpy(dtype=float)
        valid = np.isfinite(q05) & np.isfinite(median) & np.isfinite(q95)

        if not np.any(valid):
            continue

        x_valid = x_values[valid]
        q05_valid = q05[valid]
        median_valid = median[valid]
        q95_valid = q95[valid]

        if is_single_year:
            axis.errorbar(
                x_valid,
                median_valid,
                yerr=np.vstack((median_valid - q05_valid, q95_valid - median_valid)),
                fmt="o",
                color=color_map[technology],
                markersize=7,
                linewidth=2.0,
                capsize=4,
                zorder=3,
            )
        else:
            axis.fill_between(
                x_valid,
                q05_valid,
                q95_valid,
                color=color_map[technology],
                alpha=0.22,
            )
            axis.plot(
                x_valid,
                median_valid,
                color=color_map[technology],
                linewidth=2.0,
            )

    axis.set_title("Global", fontsize=12, fontweight="bold")
    axis.set_xticks(x_ticks)
    if is_single_year:
        axis.set_xlim(years[0] - 1, years[0] + 1)
    else:
        axis.set_xlim(min(years), max(years))
    axis.set_ylim(0, 1)
    axis.set_xlabel("Year")
    axis.set_ylabel("Market share")
    axis.grid(True, alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=color_map[technology], linewidth=2.0, label=technology)
        for technology in PV_TECH_ORDER
    ]
    band_handle = Line2D(
        [0], [0], color="0.3", linewidth=6, alpha=0.22, label="5th-95th percentile"
    )

    fig.legend(
        handles=legend_handles + [band_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        title=f"Global PV subtechnologies, based on {iterations} random draws",
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact figure of random PV market shares."
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subshares", type=Path, default=SUBSHARES)
    parser.add_argument("--datapackage", type=Path, default=DATAPACKAGE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pv_subshares_iterations_1000_global.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.subshares = args.subshares.resolve()
    args.datapackage = args.datapackage.resolve()
    p = initialize_pathways(args.datapackage)
    years, scenario = get_years_and_scenario(p)
    production = get_production_volumes(
        p,
        scenario,
        regions=REGIONS,
        variable=SOLAR_VARIABLE,
    )
    shares = get_share_trajectories(
        years=years,
        iterations=args.iterations,
        subshares=args.subshares,
        seed=args.seed,
    )
    global_shares = build_global_shares(shares, production)
    summary = summarize_shares(global_shares)
    make_figure(summary, years, args.iterations, args.output)
    print(f"Using subshares: {args.subshares}")
    print(f"Using datapackage: {args.datapackage}")
    print(f"Scenario: {scenario}")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
