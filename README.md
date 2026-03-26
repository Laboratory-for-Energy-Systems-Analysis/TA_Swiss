# TA-SWISS Switzerland scenario data package

Description
-----------

This repository contains a scenario data package that implements projections from the Swiss TIMES Energy Model (**STEM**) in the life cycle assessment (LCA) database **ecoinvent**.
The package was prepared for the **TA-SWISS** project assessing alternative nuclear strategies in Switzerland.

The package is designed to be used with **premise**, which combines the Swiss STEM scenarios with a global integrated assessment model pathway in order to generate scenario-specific prospective LCA databases.
In this project, the Swiss scenarios are coupled with **REMIND SSP2-PkBudg1000** for the foreign background system, so that processes outside Switzerland evolve under a decarbonizing global economy consistent with a peak cumulative CO₂ budget of 1,000 Gt by 2100.

The resulting prospective databases can be exported directly to Brightway or assembled into a data package for **pathways**, enabling time-resolved, system-level environmental assessment of final energy provision from **2020 to 2070**.

This data package contains the files required by premise to:

- read the STEM scenario trajectories,
- map STEM variables to corresponding ecoinvent activities,
- create Swiss market compositions for electricity, gaseous and liquid fuels, hydrogen, district heating, and final energy use,
- and generate scenario-specific future background databases for successive model years.


Package contents
----------------

The package includes three main components:

1. **Scenario data** (`scenario_data/scenario_data.csv`)
   
   Tabular STEM outputs in a standardized format with fields for model, scenario, region, variable, unit, and yearly values.

2. **Configuration file** (`configuration_file/config.yaml`)
   
   Mapping between STEM variables and ecoinvent datasets, including production pathways, market definitions, and regionalization instructions.

3. **Additional inventories** (`inventories/lci-sweet_sure.xlsx`)
   
   Supplementary inventories used where new or adapted datasets are needed for the Swiss scenario implementation.


Scenario coverage
-----------------

The package contains **18 Swiss scenarios**:

- SPS1
- SPS1_AP1_MD
- SPS1_AP2_MD
- SPS1_AP3_MD
- SPS1_AP4_MD
- SPS1_AP5_MD
- SPS2
- SPS2_AP1_MD
- SPS2_AP2_MD
- SPS2_AP3_MD
- SPS2_AP4_MD
- SPS2_AP5_MD
- SPS4
- SPS4_AP1_MD
- SPS4_AP2_MD
- SPS4_AP3_MD
- SPS4_AP4_MD
- SPS4_AP5_MD

The scenario file provides values for the following time steps:

**2020, 2022, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070**.


Ecoinvent compatibility
-----------------------

This package is configured for:

- **ecoinvent 3.10, cut-off**


What does this package cover?
-----------------------------

The mapping spans the main parts of the Swiss energy system and links STEM outputs to ecoinvent activities used by premise.
It covers, among others:

- electricity generation and imports,
- domestic production of hydrogen, biogas, and synthetic fuels,
- liquid and gaseous fuel imports,
- district heating and CHP supply,
- final energy use in residential buildings,
- final energy use in services,
- final energy use in industry,
- final energy use in transport.

The package also defines Swiss market datasets for energy carriers and end uses, so that consuming activities in Switzerland can be relinked to scenario-consistent energy supply mixes.

To account for decarbonization outside Switzerland, the Swiss STEM scenarios are complemented by the **REMIND SSP2-PkBudg1000** pathway for foreign background systems.
This means that imported technologies, fuels, and materials are embedded in a progressively decarbonizing global economy rather than in a static background.


How are technologies mapped?
----------------------------

The mapping between STEM variables and ecoinvent datasets is defined in:

- `configuration_file/config.yaml`

This file specifies production pathways, market creation rules, and selected regionalization steps used by premise when generating future databases.


How to use it with premise
--------------------------

The following example creates prospective ecoinvent databases for selected years by combining one Swiss STEM scenario with the REMIND **SSP2-PkBudg1000** background pathway.

```python
from premise import NewDatabase
import bw2data
from datapackage import Package

bw2data.projects.set_current("some brightway project")

swiss = Package("../datapackage.json")

scenarios = [
    {
        "model": "remind",
        "pathway": "SSP2-PkBudg1000",
        "year": 2020,
        "external scenarios": [{"scenario": "SPS1", "data": swiss}],
    },
    {
        "model": "remind",
        "pathway": "SSP2-PkBudg1000",
        "year": 2030,
        "external scenarios": [{"scenario": "SPS1", "data": swiss}],
    },
    {
        "model": "remind",
        "pathway": "SSP2-PkBudg1000",
        "year": 2050,
        "external scenarios": [{"scenario": "SPS1", "data": swiss}],
    },
    {
        "model": "remind",
        "pathway": "SSP2-PkBudg1000",
        "year": 2070,
        "external scenarios": [{"scenario": "SPS1", "data": swiss}],
    },
]

ndb = NewDatabase(
    scenarios=scenarios,
    source_db="ecoinvent-3.10-cutoff",
    source_version="3.10",
    key="xxxx",
    use_absolute_efficiency=True,
    biosphere_name="ecoinvent-3.10-biosphere",
)

ndb.update()
ndb.write_db_to_brightway()
```

Other Swiss scenarios can be used by replacing `SPS1` with any scenario listed in `datapackage.json`.


How to create a Pathways data package
-------------------------------------

The same scenario package can be used to generate a **pathways** data package for time-resolved, system-level environmental assessment.

```python
from premise import PathwaysDataPackage
from datapackage import Package

swiss = Package("../datapackage.json")

scenario = {
    "model": "remind",
    "pathway": "SSP2-PkBudg1000",
    "external scenarios": [{"scenario": "SPS1_AP2_MD", "data": swiss}],
}

ndb = PathwaysDataPackage(
    scenarios=[scenario],
    years=[2020, 2022, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070],
    source_db="ecoinvent-3.10-cutoff",
    source_version="3.10",
    key="xxxx",
    use_absolute_efficiency=True,
    biosphere_name="ecoinvent-3.10-biosphere",
)

ndb.create_datapackage(
    name="ta-swiss-remind-SSP2-PkBudg1000-stem-SPS1_AP2_MD",
    contributors=[
        {"name": "Your name", "email": "your.email@example.org"}
    ],
)
```

This produces a pathways-compatible data package that can be used to calculate the environmental impacts of final energy provision over time while remaining consistent with both the Swiss scenario assumptions and the global background pathway.


License
-------

This scenario package is licensed under the Creative Commons Attribution 4.0 International Public License (CC BY 4.0).
See the `LICENSE` file for details.
