import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


DATA_DIR = Path(__file__).resolve().parent / "STEM raw data"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "scenario_data" / "scenario_data.csv"
DATAPACKAGE_FILE = Path(__file__).resolve().parents[1] / "datapackage.json"
BASE_COLUMNS = ["model", "scenario", "region", "variables", "unit"]
SCENARIO_NAME_OVERRIDES = {
    "SPS1": "SPS1",
    "SPS2": "SPS2",
    "SPS4": "SPS4",
}
WASTE_MERGES = [
    (
        "Electricity generation|Wastes|Renewable|Wastes Incineration (electric only)",
        "Electricity generation|Wastes|Renewable|CHP Wastes (for District Heating)",
    ),
    (
        "Electricity generation|Wastes|Renewable|Wastes Incineration (electric only) CCS",
        "Electricity generation|Wastes|Renewable|CHP Wastes (for District Heating) CCS",
    ),
    (
        "Electricity generation|Wastes|Non Renewable|Wastes Incineration (electric only)",
        "Electricity generation|Wastes|Non Renewable|CHP Wastes (for District Heating)",
    ),
    (
        "Electricity generation|Wastes|Non Renewable|Wastes Incineration (electric only) CCS",
        "Electricity generation|Wastes|Non Renewable|CHP Wastes (for District Heating) CCS",
    ),
]
EXTRA_VARIABLES = [
    "Production|Electricity|Medium to high",
    "Production|Electricity|Low to medium",
]
LEGACY_NUCLEAR_END_YEAR = 2040
LEGACY_NUCLEAR_TOTAL = 33.2
LEGACY_NUCLEAR_SPLIT = {
    "Electricity generation|Nuclear fuel|Pressure water": 0.6,
    "Electricity generation|Nuclear fuel|Boiling water": 0.4,
}
NEW_NUCLEAR_VARIANTS = {
    "AP1": ("Electricity generation|Nuclear fuel|EPR", 2040),
    "AP2": ("Electricity generation|Nuclear fuel|EPR", 2050),
    "AP3": ("Electricity generation|Nuclear fuel|SMR", 2040),
    "AP4": ("Electricity generation|Nuclear fuel|SMR", 2050),
    "AP5": ("Electricity generation|Nuclear fuel|SMR", 2050),
}


def normalize_columns(columns):
    aliases = {
        "Model": "model",
        "Scenario": "scenario",
        "Region": "region",
        "Variable": "variables",
        "Unit": "unit",
    }
    normalized = []

    for column in columns:
        if isinstance(column, (int, np.integer)):
            normalized.append(int(column))
            continue

        if isinstance(column, float) and column.is_integer():
            normalized.append(int(column))
            continue

        if isinstance(column, str):
            stripped = column.strip()
            if stripped.isdigit():
                normalized.append(int(stripped))
            else:
                normalized.append(aliases.get(stripped, stripped))
            continue

        normalized.append(column)

    return normalized


def extract_year_columns(columns):
    return sorted(
        column
        for column in columns
        if isinstance(column, int) and 2005 <= column <= 2100
    )


def get_scenario_name(path):
    scenario_name = path.stem
    scenario_name = scenario_name.removeprefix("STEM_to_Premise_")
    scenario_name = scenario_name.removesuffix("_2035")
    return SCENARIO_NAME_OVERRIDES.get(scenario_name, scenario_name)


def load_stem_sheet(path):
    xls = pd.ExcelFile(path)
    sps_sheets = [sheet for sheet in xls.sheet_names if sheet.upper().startswith("SPS")]
    if not sps_sheets:
        raise ValueError(f"No SPS sheet found in {path}")

    df = pd.read_excel(xls, sheet_name=sps_sheets[0])
    df.columns = normalize_columns(df.columns)
    year_columns = extract_year_columns(df.columns)

    if not year_columns:
        raise ValueError(f"No year columns found in {path.name}")

    missing_columns = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path.name}: {missing_columns}")

    return df.loc[:, BASE_COLUMNS + year_columns].copy(), year_columns


def merge_variable_rows(df, source_variable, target_variable, year_columns):
    source_mask = df["variables"] == source_variable
    target_mask = df["variables"] == target_variable

    if not source_mask.any() or not target_mask.any():
        return df

    source_values = df.loc[source_mask, year_columns].sum(axis=0)
    df.loc[target_mask, year_columns] = df.loc[target_mask, year_columns].add(
        source_values, axis="columns"
    )
    return df.loc[~source_mask].copy()


def subtract_exports_from_imports(df, year_columns):
    imports_mask = df["variables"] == "Imports|Electricity"
    exports_mask = df["variables"] == "Exports|Electricity"

    if not imports_mask.any():
        return df

    imports = df.loc[imports_mask, year_columns].sum(axis=0)
    exports = df.loc[exports_mask, year_columns].sum(axis=0) if exports_mask.any() else 0
    net_imports = (imports - exports).clip(lower=0)

    df.loc[imports_mask, year_columns] = np.tile(
        net_imports.to_numpy(), (int(imports_mask.sum()), 1)
    )
    return df


def get_new_nuclear_variant(scenario_name):
    for scenario_marker, variant_info in NEW_NUCLEAR_VARIANTS.items():
        if scenario_marker in scenario_name:
            return variant_info
    return None, None


def split_nuclear_generation(df, scenario_name, year_columns):
    nuclear = df.loc[df["variables"] == "Electricity generation|Nuclear Fuel"].copy()
    if nuclear.empty:
        return df

    nuclear.loc[:, year_columns] = nuclear.loc[:, year_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    total_nuclear = nuclear.loc[:, year_columns].copy()

    legacy_nuclear = total_nuclear.copy()
    legacy_years = [year for year in year_columns if year <= LEGACY_NUCLEAR_END_YEAR]
    post_legacy_years = [year for year in year_columns if year > LEGACY_NUCLEAR_END_YEAR]

    if LEGACY_NUCLEAR_END_YEAR in legacy_years:
        legacy_nuclear.loc[:, LEGACY_NUCLEAR_END_YEAR] = np.minimum(
            legacy_nuclear.loc[:, LEGACY_NUCLEAR_END_YEAR],
            LEGACY_NUCLEAR_TOTAL,
        )

    if post_legacy_years:
        legacy_nuclear.loc[:, post_legacy_years] = 0

    split_rows = []
    for variable, share in LEGACY_NUCLEAR_SPLIT.items():
        split_row = nuclear.copy()
        split_row["variables"] = variable
        split_row.loc[:, year_columns] = legacy_nuclear.loc[:, year_columns] * share
        split_rows.append(split_row)

    new_variant, start_year = get_new_nuclear_variant(scenario_name)
    for variable in (
        "Electricity generation|Nuclear fuel|EPR",
        "Electricity generation|Nuclear fuel|SMR",
    ):
        split_row = nuclear.copy()
        split_row["variables"] = variable
        split_row.loc[:, year_columns] = 0

        if variable == new_variant:
            start_years = [year for year in year_columns if year >= start_year]
            if start_years:
                new_nuclear = total_nuclear.loc[:, start_years].copy()
                if start_year == LEGACY_NUCLEAR_END_YEAR and start_year in start_years:
                    new_nuclear.loc[:, start_year] = (
                        new_nuclear.loc[:, start_year] - LEGACY_NUCLEAR_TOTAL
                    ).clip(lower=0)
                split_row.loc[:, start_years] = new_nuclear

        split_rows.append(split_row)

    return pd.concat([df, *split_rows], ignore_index=True)


def backfill_efficiency_rows(df, year_columns):
    efficiency_mask = df["variables"].astype(str).str.startswith("Efficiency")
    if not efficiency_mask.any():
        return df

    efficiency_values = df.loc[efficiency_mask, year_columns].replace(0, np.nan)
    df.loc[efficiency_mask, year_columns] = efficiency_values.bfill(axis=1).ffill(axis=1)
    return df


def add_extra_variables(df, scenario_name, year_columns):
    existing_variables = set(df["variables"])
    extra_rows = []

    for variable in EXTRA_VARIABLES:
        if variable in existing_variables:
            continue

        row = {
            "model": "STEM",
            "scenario": scenario_name,
            "region": "CH",
            "variables": variable,
            "unit": "TWh",
        }
        row.update({year: 1 for year in year_columns})
        extra_rows.append(row)

    if not extra_rows:
        return df

    return pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


def format_file(path):
    scenario_name = get_scenario_name(path)
    df, year_columns = load_stem_sheet(path)

    df["model"] = "STEM"
    df["scenario"] = scenario_name

    df = subtract_exports_from_imports(df, year_columns)

    for source_variable, target_variable in WASTE_MERGES:
        df = merge_variable_rows(df, source_variable, target_variable, year_columns)

    df = split_nuclear_generation(df, scenario_name, year_columns)
    df = df.replace(["-", "- ", " ", "n.a."], np.nan).infer_objects(copy=False)
    df = backfill_efficiency_rows(df, year_columns)
    df = df.fillna(0)
    df = add_extra_variables(df, scenario_name, year_columns)

    return df.loc[:, BASE_COLUMNS + year_columns], year_columns


def update_datapackage_schema(year_columns):
    datapackage = json.loads(DATAPACKAGE_FILE.read_text())
    scenario_resource = next(
        resource
        for resource in datapackage["resources"]
        if resource.get("name") == "scenario_data"
    )

    fields = [
        {"name": "model", "type": "string", "format": "default"},
        {"name": "scenario", "type": "string", "format": "default"},
        {"name": "region", "type": "string", "format": "default"},
        {"name": "variables", "type": "string", "format": "default"},
        {"name": "unit", "type": "string", "format": "default"},
    ]
    fields.extend(
        {"name": str(year), "type": "number", "format": "default"}
        for year in year_columns
    )

    scenario_resource["schema"]["fields"] = fields
    DATAPACKAGE_FILE.write_text(json.dumps(datapackage, indent=4) + "\n")


def main():
    final_frames = []
    all_year_columns = set()

    for filepath in sorted(DATA_DIR.glob("*.xlsx")):
        print(filepath)
        df, year_columns = format_file(filepath)
        final_frames.append(df)
        all_year_columns.update(year_columns)

    sorted_year_columns = sorted(all_year_columns)
    final_df = pd.concat(
        [
            frame.reindex(columns=BASE_COLUMNS + sorted_year_columns, fill_value=0)
            for frame in final_frames
        ],
        ignore_index=True,
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False, sep=",")
    update_datapackage_schema(sorted_year_columns)


if __name__ == "__main__":
    main()
