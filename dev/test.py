#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATAPACKAGE = Path(
    "/Users/romain/Library/CloudStorage/OneDrive-PaulScherrerInstitut/TA_Swiss/"
    "remind-SSP2-PkBudg1000-stem-SPS1.zip"
)
DEFAULT_BRIGHTWAY_DIR = ROOT / ".brightway-profile"
DEFAULT_PROFILE_DIR = ROOT / "profile-results"
METHODS = [
    "EN15804+A2 - Core impact categories and indicators - climate change: total "
    "(EF v3.0 - IPCC 2013) - global warming potential (GWP100)",
    "EN15804+A2 - Core impact categories and indicators - material resources: "
    "metals/minerals - abiotic depletion potential (ADP): elements "
    "(ultimate reserves)",
    "Inventory results and indicators - resources - land occupation",
    "EN15804+A2 - Indicators describing resource use - net use of fresh water - FW",
    "ReCiPe 2016 v1.03, endpoint (H) - total: human health - human health",
    "ReCiPe 2016 v1.03, endpoint (H) - total: ecosystem quality - ecosystem quality",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the expensive Pathways run from "
            "`2 - indicators generation.ipynb` with cProfile."
        )
    )
    parser.add_argument(
        "--datapackage",
        type=Path,
        default=DEFAULT_DATAPACKAGE,
        help="Zip datapackage to profile.",
    )
    parser.add_argument(
        "--brightway-dir",
        type=Path,
        default=DEFAULT_BRIGHTWAY_DIR,
        help="Writable Brightway base directory used before importing pathways.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="Directory where .pstats and text summaries are written.",
    )
    parser.add_argument(
        "--variable-prefix",
        default="SE - electricity",
        help="Only variables starting with this prefix are included.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["EUR", "USA", "CHA"],
        help="IAM regions to include.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on the number of pathways to profile.",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=None,
        help="Optional cap on the number of years to profile.",
    )
    parser.add_argument(
        "--max-variables",
        type=int,
        default=None,
        help="Optional cap on the number of variables to profile.",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Enable multiprocessing in Pathways.calculate().",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export_results() if you only want init/calculate timings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Number of functions shown in each printed stats table.",
    )
    return parser


def prepare_brightway_environment(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "output").mkdir(parents=True, exist_ok=True)
    os.environ["BRIGHTWAY2_DIR"] = str(base_dir)
    os.environ["BRIGHTWAY2_OUTPUT_DIR"] = str(base_dir / "output")
    os.environ.setdefault("BRIGHTWAY_NO_STRUCTLOG", "1")


def take_first(values, limit: int | None) -> list:
    values = list(values)
    if limit is None:
        return values
    return values[:limit]


def profile_stage(
    name: str,
    output_dir: Path,
    limit: int,
    func,
):
    profiler = cProfile.Profile()
    started = time.perf_counter()
    result = profiler.runcall(func)
    elapsed = time.perf_counter() - started

    pstats_path = output_dir / f"{name}.pstats"
    summary_path = output_dir / f"{name}.txt"
    profiler.dump_stats(str(pstats_path))

    stream = io.StringIO()
    stream.write(f"[{name}] wall time: {elapsed:.2f}s\n\n")

    stream.write(f"Top {limit} functions by cumulative time\n")
    cumulative = pstats.Stats(profiler, stream=stream)
    cumulative.strip_dirs().sort_stats("cumulative").print_stats(limit)

    stream.write(f"\nTop {limit} functions by self time\n")
    self_time = pstats.Stats(profiler, stream=stream)
    self_time.strip_dirs().sort_stats("tottime").print_stats(limit)

    summary_path.write_text(stream.getvalue(), encoding="utf-8")

    print(f"[{name}] wall time: {elapsed:.2f}s")
    print(f"[{name}] wrote {pstats_path}")
    print(f"[{name}] wrote {summary_path}")

    return result, elapsed


def main() -> int:
    args = build_parser().parse_args()
    prepare_brightway_environment(args.brightway_dir.resolve())
    args.profile_dir.mkdir(parents=True, exist_ok=True)

    from pathways import Pathways

    datapackage = args.datapackage.resolve()
    if not datapackage.exists():
        raise FileNotFoundError(f"Datapackage not found: {datapackage}")

    def load_pathways() -> Pathways:
        return Pathways(
            datapackage=str(datapackage),
            debug=False,
            ecoinvent_version="3.11",
        )

    pathways_obj, init_time = profile_stage(
        "01_init",
        args.profile_dir,
        args.limit,
        load_pathways,
    )

    scenarios = take_first(pathways_obj.scenarios.pathway.values.tolist(), args.max_scenarios)
    years = take_first(pathways_obj.scenarios.year.values.tolist(), args.max_years)
    variables = [
        str(value)
        for value in pathways_obj.scenarios.coords["variables"].values
        if str(value).startswith(args.variable_prefix)
    ]
    variables = take_first(variables, args.max_variables)

    if not scenarios:
        raise ValueError("No pathways selected.")
    if not years:
        raise ValueError("No years selected.")
    if not variables:
        raise ValueError(
            f"No variables matched prefix {args.variable_prefix!r}."
        )

    print(f"Datapackage: {datapackage}")
    print(f"Scenarios selected: {len(scenarios)}")
    print(f"Years selected: {len(years)}")
    print(f"Variables selected: {len(variables)}")
    print(f"Regions selected: {len(args.regions)} -> {args.regions}")
    print(f"Init wall time: {init_time:.2f}s")

    def calculate() -> None:
        pathways_obj.calculate(
            methods=METHODS,
            regions=args.regions,
            years=years,
            scenarios=scenarios,
            variables=variables,
            multiprocessing=args.multiprocessing,
            use_distributions=0,
        )

    _, calculate_time = profile_stage(
        "02_calculate",
        args.profile_dir,
        args.limit,
        calculate,
    )

    print(f"Calculate wall time: {calculate_time:.2f}s")

    if args.skip_export:
        return 0

    export_filename = str(datapackage).replace(".zip", "").replace("remind", "results_remind")

    def export() -> str:
        return pathways_obj.export_results(filename=export_filename)

    export_path, export_time = profile_stage(
        "03_export",
        args.profile_dir,
        args.limit,
        export,
    )

    print(f"Export wall time: {export_time:.2f}s")
    print(f"Exported file: {export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
