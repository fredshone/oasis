# OASIS for activity scheduling 

WIP for converting OASIS to activity scheduling only and estimating on NTS 2023.

Based on [OASIS](https://zenodo.org/badge/latestdoi/685457042).


## Installation guide   

local install. `uv` works well.

**NB: You need a valid CPLEX user license to use the solver, which can be obtained [here](https://www.ibm.com/academic/topic/data-science).**


## NTS 2023 experiment

The `nts2023` branch adapts OASIS to NTS 2023 diary schedules (no trips) and compares it with caveat models. See [experiments/nts2023/README.md](experiments/nts2023/README.md) for the changes made to the library, the experiments and the results.
