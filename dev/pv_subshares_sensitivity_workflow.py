from __future__ import annotations

import argparse
import os
from copy import deepcopy
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
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pathways import Pathways
from pathways.subshares import generate_samples


BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATAPACKAGE = BASE_DIR / "remind-SSP2-PkBudg1000-stem-SPS1.zip"
LEGACY_DATAPACKAGE = Path(
    "/Users/romain/Library/CloudStorage/OneDrive-PaulScherrerInstitut/TA_Swiss/"
    "remind-SSP2-PkBudg1000-stem-SPS1.zip"
)
DATAPACKAGE = LOCAL_DATAPACKAGE if LOCAL_DATAPACKAGE.exists() else LEGACY_DATAPACKAGE
SUBSHARES = Path("pv_subshares_sensitivity.yaml")
ELECTRICITY_IMPACTS_CACHE = Path("impact_totals_cache.csv")
ELECTRICITY_PRODUCTION = Path("production_volumes.csv")
SOLAR_VARIABLE = "SE - electricity - Solar PV Centralized"
REGIONS = ["EUR", "USA", "CHA"]
METHODS = [
    "EN15804+A2 - Core impact categories and indicators - climate change: total (EF v3.0 - IPCC 2013) - global warming potential (GWP100)",
    "EN15804+A2 - Core impact categories and indicators - material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves)",
    "Inventory results and indicators - resources - land occupation",
    "EN15804+A2 - Indicators describing resource use - net use of fresh water - FW",
    "ReCiPe 2016 v1.03, endpoint (H) - total: human health - human health",
    "ReCiPe 2016 v1.03, endpoint (H) - total: ecosystem quality - ecosystem quality",
]
IMPACT_ORDER = [
    "climate change",
    "human health",
    "ecosystem quality",
    "land occupation",
    "net use of fresh water",
    "minerals depletion",
]
IMPACT_LABELS = {
    "EN15804+A2 - Core impact categories and indicators - climate change: total (EF v3.0 - IPCC 2013) - global warming potential (GWP100)": "climate change",
    "EN15804+A2 - Core impact categories and indicators - material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves)": "minerals depletion",
    "Inventory results and indicators - resources - land occupation": "land occupation",
    "EN15804+A2 - Indicators describing resource use - net use of fresh water - FW": "net use of fresh water",
    "ReCiPe 2016 v1.03, endpoint (H) - total: human health - human health": "human health",
    "ReCiPe 2016 v1.03, endpoint (H) - total: ecosystem quality - ecosystem quality": "ecosystem quality",
}
PV_TECH_ORDER = ["c-Si", "CdTe", "CIGS", "a-Si", "Perovskite", "GaAs"]
KWH_PER_EJ = 1e18 / 3.6e6
RUNTIME_SUBSHARES_DIR = Path(".pathways-user-data/generated")
PATHWAYS_SHARE_KEYS = {"loc", "scale", "minimum", "maximum", "uncertainty_type"}
PV_TARGETS = [
    {
        "name": "electricity production, photovoltaic, commercial",
        "reference product": "electricity, low voltage",
        "unit": "kilowatt hour",
    }
]
PV_TECH_METADATA = {
    "c-Si": {
        "name": (
            "electricity production, photovoltaic, photovoltaic slanted-roof "
            "installation, 3 kWp, multi-Si, laminated, integrated"
        ),
        "reference product": "electricity, low voltage",
        "unit": "kilowatt hour",
    },
    "CdTe": {
        "name": (
            "electricity production, photovoltaic, photovoltaic slanted-roof "
            "installation, 3 kWp, CdTe, laminated, integrated"
        ),
        "reference product": "electricity, low voltage",
        "unit": "kilowatt hour",
    },
    "CIGS": {
        "name": (
            "electricity production, photovoltaic, photovoltaic slanted-roof "
            "installation, 3 kWp, CIS, panel, mounted"
        ),
        "reference product": "electricity, low voltage",
        "unit": "kilowatt hour",
    },
    "a-Si": {
        "name": (
            "electricity production, photovoltaic, 3kWp slanted-roof installation, "
            "a-Si, laminated, integrated"
        ),
        "reference product": "electricity, low voltage",
        "unit": "kilowatt hour",
    },
    "Perovskite": {
        "name": "electricity production, photovoltaic, 0.5kWp, perovskite-on-silicon tandem",
        "reference product": (
            "electricity production, photovoltaic, 0.5kWp, perovskite-on-silicon tandem"
        ),
        "unit": "kilowatt hour",
    },
    "GaAs": {
        "name": "electricity production, photovoltaic, 0.28kWp, GaAs",
        "reference product": "electricity production, photovoltaic, 0.28kWp, GaAs",
        "unit": "kilowatt hour",
    },
}


def initialize_pathways(datapackage: Path = DATAPACKAGE) -> Pathways:
    return Pathways(datapackage=str(datapackage), debug=False, ecoinvent_version="3.11")


def get_years_and_scenario(p: Pathways, scenario: str | None = None) -> tuple[list[int], str]:
    years = [int(year) for year in p.scenarios.coords["year"].values.tolist()]
    chosen_scenario = scenario or p.scenarios.pathway.values.tolist()[0]
    return years, chosen_scenario


def get_production_volumes(
    p: Pathways,
    scenario: str,
    regions: list[str] = REGIONS,
    variable: str = SOLAR_VARIABLE,
) -> pd.DataFrame:
    production = (
        p.scenarios.sel(pathway=scenario, variables=variable, region=regions)
        .to_dataframe("production_ej")
        .reset_index()
    )
    production["year"] = production["year"].astype(int)
    return production[["model", "pathway", "region", "year", "production_ej"]]


def _is_year_mapping(data: dict) -> bool:
    if not isinstance(data, dict) or not data:
        return False

    try:
        return all(str(key).isdigit() for key in data)
    except TypeError:
        return False


def _coerce_share_params(params: dict) -> dict:
    if not isinstance(params, dict):
        raise ValueError(f"Invalid share parameters: {params!r}")

    if any(key in params for key in PATHWAYS_SHARE_KEYS):
        return {key: deepcopy(value) for key, value in params.items() if key in PATHWAYS_SHARE_KEYS}

    anchor_values = {key: params.get(key) for key in ("p10", "p50", "p90")}
    if not any(value is not None for value in anchor_values.values()):
        raise ValueError(
            "Share parameters must use Pathways keys or define at least one of p10/p50/p90."
        )

    mode = float(
        anchor_values["p50"]
        if anchor_values["p50"] is not None
        else next(value for value in (anchor_values["p10"], anchor_values["p90"]) if value is not None)
    )
    lower = float(anchor_values["p10"] if anchor_values["p10"] is not None else mode)
    upper = float(anchor_values["p90"] if anchor_values["p90"] is not None else mode)
    lower, mode, upper = sorted((lower, mode, upper))

    if np.isclose(lower, upper):
        return {"loc": mode}

    # The updated YAML uses p10/p50/p90 narrative envelopes; Pathways needs
    # a concrete distribution, so we approximate those anchors with a triangular
    # distribution using p10 as the lower bound, p50 as the mode, and p90 as the upper bound.
    return {
        "uncertainty_type": "triangular",
        "minimum": lower,
        "loc": mode,
        "maximum": upper,
    }


def _select_share_mapping(share_data: dict, region: str) -> dict | None:
    if _is_year_mapping(share_data):
        return share_data

    if not isinstance(share_data, dict):
        return None

    if region in share_data:
        return share_data[region]

    for fallback in ("default", "GLO", "global", "World", "world"):
        if fallback in share_data:
            return share_data[fallback]

    if len(share_data) == 1:
        return next(iter(share_data.values()))

    return None


def _extend_year_mapping(year_mapping: dict[int, dict], years: list[int]) -> dict[int, dict]:
    if not year_mapping or not years:
        return dict(sorted(year_mapping.items()))

    extended = {int(year): deepcopy(params) for year, params in year_mapping.items()}
    first_year = min(extended)
    last_year = max(extended)
    target_first = min(int(year) for year in years)
    target_last = max(int(year) for year in years)

    if target_first < first_year:
        extended[target_first] = deepcopy(extended[first_year])

    if target_last > last_year:
        # Pathways does not extrapolate beyond the last defined year, so we add
        # a terminal anchor at the end of the model horizon using the last
        # available distribution parameters.
        extended[target_last] = deepcopy(extended[last_year])

    return dict(sorted(extended.items()))


def build_runtime_subshares(
    subshares: Path = SUBSHARES,
    years: list[int] | None = None,
    regions: list[str] = REGIONS,
) -> dict:
    with open(subshares, encoding="utf-8") as stream:
        source = yaml.safe_load(stream)

    pv_group = source.get("PV")
    if not isinstance(pv_group, dict):
        raise ValueError(f"Missing or invalid 'PV' group in {subshares}.")

    runtime_group = {
        "_targets": deepcopy(pv_group.get("_targets", PV_TARGETS)),
    }

    for technology in PV_TECH_ORDER:
        if technology not in pv_group:
            continue

        tech_source = pv_group[technology]
        tech_runtime = deepcopy(PV_TECH_METADATA[technology])
        tech_runtime.update(
            {
                key: deepcopy(value)
                for key, value in tech_source.items()
                if key != "share" and not key.startswith("_")
            }
        )

        share_data = tech_source.get("share", {})
        regional_shares = {}
        for region in regions:
            selected_mapping = _select_share_mapping(share_data, region)
            if selected_mapping is None:
                continue

            year_mapping = {
                int(year): _coerce_share_params(params)
                for year, params in selected_mapping.items()
            }
            regional_shares[region] = _extend_year_mapping(year_mapping, years or [])

        if not regional_shares:
            continue

        tech_runtime["share"] = regional_shares
        runtime_group[technology] = tech_runtime

    if len(runtime_group) == 1:
        raise ValueError(f"No usable PV subshare definitions found in {subshares}.")

    return {"PV": runtime_group}


def prepare_subshares_file(
    subshares: Path = SUBSHARES,
    years: list[int] | None = None,
    regions: list[str] = REGIONS,
) -> Path:
    runtime_subshares = build_runtime_subshares(subshares=subshares, years=years, regions=regions)
    runtime_path = RUNTIME_SUBSHARES_DIR / f"{Path(subshares).stem}.pathways.yaml"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(runtime_subshares, stream, sort_keys=False)
    return runtime_path


def get_share_trajectories(
    years: list[int],
    iterations: int,
    subshares: Path = SUBSHARES,
    regions: list[str] = REGIONS,
    seed: int = 42,
) -> pd.DataFrame:
    runtime_subshares = prepare_subshares_file(subshares=subshares, years=years, regions=regions)
    shares = generate_samples(
        years=years,
        iterations=iterations,
        regions=regions,
        seed=seed,
        subshares=runtime_subshares,
        groups=["PV"],
    )

    records = []
    for region in regions:
        for year, technologies in shares["PV"][region].items():
            for technology, values in technologies.items():
                sampled_values = np.atleast_1d(np.asarray(values, dtype=float))
                for iteration, share in enumerate(sampled_values, start=1):
                    records.append(
                        {
                            "region": region,
                            "year": int(year),
                            "technology": technology,
                            "iteration": iteration,
                            "share": float(share),
                        }
                    )

    trajectories = pd.DataFrame(records).sort_values(
        ["region", "year", "technology", "iteration"]
    )
    return trajectories


def run_pathways_sensitivity(
    datapackage: Path = DATAPACKAGE,
    subshares: Path = SUBSHARES,
    iterations: int = 15,
    seed: int = 42,
    multiprocessing: bool = True,
) -> tuple[Pathways, str, pd.DataFrame, pd.DataFrame]:
    p = initialize_pathways(datapackage)
    years, scenario = get_years_and_scenario(p)
    runtime_subshares = prepare_subshares_file(subshares=subshares, years=years, regions=REGIONS)
    production = get_production_volumes(p, scenario)
    shares = get_share_trajectories(years, iterations, subshares=subshares, seed=seed)

    p.calculate(
        methods=METHODS,
        scenarios=[scenario],
        regions=REGIONS,
        years=years,
        variables=[SOLAR_VARIABLE],
        use_distributions=iterations,
        subshares=str(runtime_subshares),
        subshare_groups=["PV"],
        remove_uncertainty=True,
        seed=seed,
        multiprocessing=multiprocessing,
    )

    return p, scenario, production, shares


def export_results(p: Pathways, export_base: Path) -> Path:
    export_path = p.export_results(filename=str(export_base))
    return Path(export_path)


def aggregate_lca_results(p: Pathways) -> pd.DataFrame:
    aggregated = p.lca_results.sum(dim=["act_category", "location"], skipna=True)
    frame = aggregated.to_dataframe(name="value").reset_index()
    frame["year"] = frame["year"].astype(int)
    frame["quantile"] = pd.to_numeric(frame["quantile"], errors="coerce")
    frame["impact_category"] = frame["impact_category"].replace(IMPACT_LABELS)
    return frame


def aggregate_exported_results(export_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(export_path)
    if "quantile" not in frame.columns:
        frame["quantile"] = 0.5

    frame["year"] = frame["year"].astype(int)
    frame["quantile"] = pd.to_numeric(frame["quantile"], errors="coerce")
    frame["impact_category"] = frame["impact_category"].replace(IMPACT_LABELS)

    aggregated = (
        frame.groupby(
            ["variable", "year", "region", "model", "scenario", "impact_category", "quantile"],
            as_index=False,
            observed=True,
        )["value"]
        .sum()
    )
    return aggregated


def load_deterministic_electricity_impacts(
    cache_path: Path = ELECTRICITY_IMPACTS_CACHE,
    solar_variable: str = SOLAR_VARIABLE,
) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Whole-electricity impacts cache not found: {cache_path}"
        )

    impacts = pd.read_csv(cache_path)
    impacts["year"] = impacts["year"].astype(int)
    impacts = impacts[
        impacts["region"].isin(REGIONS) & impacts["impact_category"].isin(IMPACT_ORDER)
    ].copy()

    total_impacts = (
        impacts.groupby(["region", "year", "impact_category"], as_index=False, observed=True)[
            "value"
        ]
        .sum()
        .rename(columns={"value": "electricity_total_det"})
    )
    solar_impacts = (
        impacts[impacts["variable"] == solar_variable]
        .groupby(["region", "year", "impact_category"], as_index=False, observed=True)["value"]
        .sum()
        .rename(columns={"value": "solar_total_det"})
    )

    deterministic = total_impacts.merge(
        solar_impacts,
        on=["region", "year", "impact_category"],
        how="left",
    )
    deterministic["solar_total_det"] = deterministic["solar_total_det"].fillna(0.0)
    deterministic["other_electricity_det"] = (
        deterministic["electricity_total_det"] - deterministic["solar_total_det"]
    )
    return deterministic


def load_total_electricity_production(
    production_path: Path = ELECTRICITY_PRODUCTION,
) -> pd.DataFrame:
    if not production_path.exists():
        raise FileNotFoundError(
            f"Whole-electricity production file not found: {production_path}"
        )

    production = pd.read_csv(production_path, header=[0, 1])
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

    return (
        production.groupby(["region", "year"], as_index=False, observed=True)["production_ej"]
        .sum()
        .rename(columns={"production_ej": "electricity_total_ej"})
    )


def prepare_plot_data(
    production: pd.DataFrame,
    shares: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    deterministic_electricity_impacts: pd.DataFrame,
    total_electricity_production: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = shares.merge(
        production[["region", "year", "production_ej"]],
        on=["region", "year"],
        how="left",
    )
    split["production_split_ej"] = split["share"] * split["production_ej"]
    split_summary = (
        split.groupby(["region", "year", "technology"], as_index=False)["production_split_ej"]
        .median()
        .rename(columns={"production_split_ej": "median_production_split_ej"})
    )

    climate = (
        aggregated_results[aggregated_results["impact_category"] == "climate change"]
        .groupby(["region", "year", "quantile"], as_index=False, observed=True)["value"]
        .sum()
        .rename(columns={"value": "pv_climate_kg"})
    )
    deterministic_climate = deterministic_electricity_impacts[
        deterministic_electricity_impacts["impact_category"] == "climate change"
    ][["region", "year", "other_electricity_det"]].rename(
        columns={"other_electricity_det": "other_electricity_climate_kg"}
    )
    climate = climate.merge(
        deterministic_climate,
        on=["region", "year"],
        how="left",
    )
    climate = climate.merge(
        total_electricity_production,
        on=["region", "year"],
        how="left",
    )
    climate["other_electricity_climate_kg"] = climate["other_electricity_climate_kg"].fillna(0.0)
    climate["climate_kg"] = climate["pv_climate_kg"] + climate["other_electricity_climate_kg"]
    climate["climate_gt"] = climate["climate_kg"] / 1e12
    climate["climate_per_kwh"] = np.where(
        climate["electricity_total_ej"].gt(0),
        climate["climate_kg"] / (climate["electricity_total_ej"] * KWH_PER_EJ),
        np.nan,
    )

    pv_impacts = (
        aggregated_results.groupby(
            ["year", "region", "impact_category", "quantile"],
            as_index=False,
            observed=True,
        )["value"]
        .sum()
        .rename(columns={"value": "pv_impact"})
    )
    indexed = pv_impacts.merge(
        deterministic_electricity_impacts[
            ["region", "year", "impact_category", "other_electricity_det"]
        ],
        on=["region", "year", "impact_category"],
        how="left",
    )
    indexed["other_electricity_det"] = indexed["other_electricity_det"].fillna(0.0)
    indexed["impact_total"] = indexed["other_electricity_det"] + indexed["pv_impact"]
    baseline = (
        indexed[indexed["year"] == 2020]
        .loc[:, ["region", "impact_category", "quantile", "impact_total"]]
        .rename(columns={"impact_total": "baseline_2020"})
    )
    indexed = indexed.merge(
        baseline,
        on=["region", "impact_category", "quantile"],
        how="left",
    )
    indexed["index_2020"] = np.where(
        indexed["baseline_2020"].ne(0),
        indexed["impact_total"] / indexed["baseline_2020"] * 100,
        np.nan,
    )

    return split_summary, climate, indexed


def _series_by_quantile(
    data: pd.DataFrame,
    value_column: str,
    region: str,
    impact_category: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = data[data["region"] == region].copy()
    if impact_category is not None:
        subset = subset[subset["impact_category"] == impact_category].copy()

    pivot = (
        subset.pivot_table(
            index="year",
            columns="quantile",
            values=value_column,
            aggfunc="first",
        )
        .sort_index()
    )
    median_col = 0.5 if 0.5 in pivot.columns else pivot.columns[0]
    low_col = 0.05 if 0.05 in pivot.columns else median_col
    high_col = 0.95 if 0.95 in pivot.columns else median_col
    years = pivot.index.to_numpy()
    return (
        years,
        pivot[low_col].to_numpy(),
        pivot[median_col].to_numpy(),
        pivot[high_col].to_numpy(),
    )


def make_figure(
    production_split: pd.DataFrame,
    climate: pd.DataFrame,
    indexed: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
        }
    )

    category_colors = plt.cm.Dark2(np.linspace(0, 1, len(IMPACT_ORDER)))
    category_color_map = dict(zip(IMPACT_ORDER, category_colors, strict=True))

    tech_colors = plt.cm.YlGnBu(np.linspace(0.25, 0.95, len(PV_TECH_ORDER)))
    tech_color_map = dict(zip(PV_TECH_ORDER, tech_colors, strict=True))

    years = sorted(production_split["year"].unique().tolist())
    x_ticks = years[::2]

    production_max = (
        production_split.groupby(["region", "year"], observed=True)["median_production_split_ej"]
        .sum()
        .max()
    )
    climate_max = climate["climate_gt"].max(skipna=True)
    intensity_max = climate["climate_per_kwh"].max(skipna=True)
    index_max = indexed["index_2020"].max(skipna=True)

    fig, axes = plt.subplots(3, 3, figsize=(24, 15), sharex=True)

    titles = [
        "PV Production Split",
        "Climate Change Impact",
        "Impacts Relative To 2020",
    ]
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, fontsize=18, fontweight="bold")

    for row, region in enumerate(REGIONS):
        left_axis = axes[row, 0]
        middle_axis = axes[row, 1]
        right_axis = axes[row, 2]
        middle_secondary = middle_axis.twinx()

        split_region = (
            production_split[production_split["region"] == region]
            .pivot(index="year", columns="technology", values="median_production_split_ej")
            .reindex(index=years, columns=PV_TECH_ORDER)
            .fillna(0.0)
        )
        left_axis.stackplot(
            years,
            split_region.T.to_numpy(),
            colors=[tech_color_map[tech] for tech in PV_TECH_ORDER],
            alpha=0.92,
            linewidth=0.0,
        )

        climate_years, climate_q05, climate_q50, climate_q95 = _series_by_quantile(
            climate,
            "climate_gt",
            region,
        )
        _, intensity_q05, intensity_q50, intensity_q95 = _series_by_quantile(
            climate,
            "climate_per_kwh",
            region,
        )

        middle_axis.fill_between(
            climate_years,
            climate_q05,
            climate_q95,
            color="#d94801",
            alpha=0.26,
        )
        middle_axis.plot(
            climate_years,
            climate_q50,
            color="#d94801",
            linewidth=3.1,
        )

        middle_secondary.fill_between(
            climate_years,
            intensity_q05,
            intensity_q95,
            color="0.2",
            alpha=0.16,
        )
        middle_secondary.plot(
            climate_years,
            intensity_q50,
            color="0.2",
            linewidth=2.7,
            linestyle="--",
        )

        for impact_category in IMPACT_ORDER:
            index_years, index_q05, index_q50, index_q95 = _series_by_quantile(
                indexed,
                "index_2020",
                region,
                impact_category=impact_category,
            )
            right_axis.fill_between(
                index_years,
                index_q05,
                index_q95,
                color=category_color_map[impact_category],
                alpha=0.16,
            )
            right_axis.plot(
                index_years,
                index_q50,
                color=category_color_map[impact_category],
                linewidth=3.5,
                label=impact_category,
            )

        right_axis.axhline(100, color="0.45", linewidth=1.0, linestyle=":")

        left_axis.annotate(
            region,
            xy=(-0.28, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=17,
            fontweight="bold",
        )

        left_axis.set_ylabel("Production [EJ]")
        middle_axis.set_ylabel("Climate Impact [Gt CO2-eq.]")
        middle_secondary.set_ylabel("Intensity [kg CO2-eq./kWh]")
        right_axis.set_ylabel("Index (2020 = 100)")

        left_axis.set_ylim(0, production_max * 1.05)
        middle_axis.set_ylim(0, climate_max * 1.05)
        middle_secondary.set_ylim(0, intensity_max * 1.05)
        right_axis.set_ylim(0, index_max * 1.05)

        left_axis.set_xlim(min(years), max(years))
        middle_axis.set_xlim(min(years), max(years))
        right_axis.set_xlim(min(years), max(years))

        left_axis.set_xticks(x_ticks)
        middle_axis.set_xticks(x_ticks)
        right_axis.set_xticks(x_ticks)

        for axis in [left_axis, middle_axis, middle_secondary, right_axis]:
            axis.grid(True, alpha=0.25)
            axis.tick_params(labelsize=13)

    for axis in axes[-1]:
        axis.set_xlabel("Year")

    tech_handles = [
        Patch(facecolor=tech_color_map[tech], edgecolor="none", label=tech)
        for tech in PV_TECH_ORDER
    ]
    category_handles = [
        Line2D([0], [0], color=category_color_map[category], linewidth=3.5, label=category)
        for category in IMPACT_ORDER
    ]
    uncertainty_handles = [
        Line2D([0], [0], color="#d94801", linewidth=3.1, label="Median climate impact"),
        Patch(facecolor="#d94801", alpha=0.26, edgecolor="none", label="5th-95th percentile"),
        Line2D([0], [0], color="0.2", linewidth=2.7, linestyle="--", label="Median climate intensity"),
    ]

    fig.legend(
        handles=tech_handles,
        loc="lower center",
        bbox_to_anchor=(0.26, 0.02),
        ncol=3,
        frameon=False,
        fontsize=12,
        title="PV Subtechnologies",
    )
    fig.legend(
        handles=category_handles,
        loc="lower center",
        bbox_to_anchor=(0.66, 0.02),
        ncol=3,
        frameon=False,
        fontsize=12,
        title="Impact Categories",
    )
    fig.legend(
        handles=uncertainty_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        frameon=False,
        fontsize=12,
        title="Middle Column",
    )

    fig.tight_layout(rect=[0.04, 0.13, 0.98, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PV subshare sensitivity in Pathways.")
    parser.add_argument("--datapackage", type=Path, default=DATAPACKAGE)
    parser.add_argument("--subshares", type=Path, default=SUBSHARES)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument(
        "--export-base",
        type=Path,
        default=Path("pv_subshares_sensitivity_results"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("pv_subshares_sensitivity_3x3.png"),
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Skip Pathways.calculate() and reuse an existing exported parquet/gzip file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p = initialize_pathways(args.datapackage)
    years, scenario = get_years_and_scenario(p)
    runtime_subshares = prepare_subshares_file(
        subshares=args.subshares,
        years=years,
        regions=REGIONS,
    )
    production = get_production_volumes(p, scenario)
    deterministic_electricity_impacts = load_deterministic_electricity_impacts()
    total_electricity_production = load_total_electricity_production()
    shares = get_share_trajectories(years, args.iterations, subshares=args.subshares, seed=args.seed)

    if args.reuse_export:
        export_path = args.export_base.with_suffix(".gzip")
        if not export_path.exists():
            raise FileNotFoundError(f"Export file not found: {export_path}")
        if args.subshares.exists() and args.subshares.stat().st_mtime > export_path.stat().st_mtime:
            print(
                "Warning: the export predates the current subshares YAML; "
                "reusing it will not reflect the latest PV assumptions."
            )
        aggregated_results = aggregate_exported_results(export_path)
    else:
        p.calculate(
            methods=METHODS,
            scenarios=[scenario],
            regions=REGIONS,
            years=years,
            variables=[SOLAR_VARIABLE],
            use_distributions=args.iterations,
            subshares=str(runtime_subshares),
            subshare_groups=["PV"],
            remove_uncertainty=True,
            seed=args.seed,
            multiprocessing=args.multiprocessing,
        )
        export_path = export_results(p, args.export_base)
        aggregated_results = aggregate_lca_results(p)

    production_split, climate, indexed = prepare_plot_data(
        production=production,
        shares=shares,
        aggregated_results=aggregated_results,
        deterministic_electricity_impacts=deterministic_electricity_impacts,
        total_electricity_production=total_electricity_production,
    )
    make_figure(production_split, climate, indexed, args.figure)
    print(f"Scenario: {scenario}")
    print(f"Runtime subshares: {runtime_subshares}")
    print(f"Exported results to {export_path}")
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
