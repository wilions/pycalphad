# Agent Instructions — PyCalphad

> [!IMPORTANT]
> **HARD RULE — REPOSITORY BOUNDARY & OUTPUT LOCATION**:
> - Store **ONLY source code, test files, examples, tools, documentation, and package configuration** (`pycalphad/`, `docs/`, `examples/`, `tools/`, `tdb-forge/`, `pyproject.toml`, `setup.py`, `README.rst`, `AGENTS.md`, `LICENSE.txt`) inside `/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad`.
> - **DO NOT store any other files** (diagrams, `.drawio` files, PNGs, research reports, process maps, study plans, simulation output files, or benchmark artifacts) inside this folder.
> - All non-source files and task outputs **MUST** be placed directly under `/Users/pei/My Drive/Antigravity/Alloy Agents/Reports/`.

You are working in **PyCalphad**, the core CALPHAD thermodynamic computation engine, equilibrium solver, and Additive Manufacturing (AM) simulation framework.

## Strict Repository & Output Directory Rule

- **Source Code Boundary**: Keep **ONLY source code, test files, examples, tools, and package configuration** (`pycalphad/`, `docs/`, `examples/`, `tools/`, `tdb-forge/`, `pyproject.toml`, `setup.py`, `README.rst`, `AGENTS.md`) inside `/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad`.
- **Output Location**: Move and export **ALL non-source code, study plans, research reports, process maps, simulation output files, and benchmark artifacts** directly to `/Users/pei/My Drive/Antigravity/Alloy Agents/Reports`.
- Every simulation or task output MUST create its own dedicated task folder under `Reports/` formatted with the task number and name (e.g., `task_01_<name>`, `task_02_<name>`).

## Environment facts

- Run tests via `pytest`: `pytest pycalphad/tests/test_am.py`.
- Do not create temporary or scratch output files within the repository tree; use the system temp directory or `Reports/` directory.
