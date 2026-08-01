# Antigravity Mission Brief — PyCalphad Fork + MCP Seam Improvement

You are working across **two directories**:

| Directory | What it is |
|---|---|
| `Alloy Agents/PyCalphad` | A fork of upstream `pycalphad` (v0.11.3.dev) carrying a purely additive local layer: AM process simulation (`pycalphad/am`, `pycalphad/amkit`), 1D transport (`pycalphad/diffusion`), precipitation kinetics (`pycalphad/precipitation`), and a database build pipeline (`tdb-forge/`). It extends CALPHAD from *equilibrium* thermodynamics into *time-dependent* kinetics and transport. |
| `MCP/pycalphad-mcp` | 529 lines. FastMCP server over stdio; **8 tools verified live**. The **only** channel between the fork and `alloyforge`. Three of its tools are backed by the fork's local layer. |

Your mission in **this** brief is to execute **`IMPROVEMENT_PLAN.md`** (PyCalphad repo root), one
phase per conversation. It is a *separate* mission from any AlloyForge brief
(`CLOSED_LOOP_BRIEF.md`, `DATA_INTEGRITY_BRIEF.md`, `AI_BRAIN_BRIEF.md`,
`TEST_ROBUSTNESS_BRIEF.md`, `CALPHAD_OKF_BRIEF.md`) — those live in a different repo and are
complete. Do not reopen them while running this one.

## Mission protocol

1. **Read `IMPROVEMENT_PLAN.md` in full before doing anything** — especially "Out of scope" and
   the Phase 0.3 baseline step. It is the single source of truth; if this brief and the plan
   disagree, **the plan wins**.
2. **Read `CODEBASE_STUDY.md`** (PyCalphad repo root) for the analysis this plan is built on.
   §1–9 cover the fork; **§10 covers `pycalphad-mcp`**. The study is current as of 2026-07-26.
3. **Execute exactly ONE phase per conversation**, strictly in order **0 → 1 → 2 → 3 → 4**.
   Phase 0 is an enabler, not a priority: until the install is editable *and* curated databases
   are discoverable, later phases land invisibly to `alloyforge`.
4. **Plan gate:** before writing any code, present your implementation plan for the phase and
   wait for the user's approval.
5. **Track progress in the plan file:** check off `- [ ]` tasks in `IMPROVEMENT_PLAN.md` as you
   complete them. Check off only what you actually did.
6. **Report honestly.** If a phase's acceptance criteria are not met, say so with the actual
   output. Do not mark a phase done to keep momentum.

## Why this mission exists

Four problems, in the priority order the user confirmed:

1. **Trust.** `tdb-forge/pipeline/verify.py` emits a hardcoded `PASS` for invariant reactions
   without computing anything, and `tdb-forge/reports/al-sc/al-sc_report.md` — a file asserting a
   verification that never ran — is **committed to the repository**. Everything else about the
   provenance design is strong (DOI tagging, jsonschema validation, compile-time
   piecewise-continuity enforcement, build-failing supersede collision detection). This one stub
   is the weak link, and it hides well because the diffusivity check beside it is genuine.
2. **Reach.** Current work is refractory and high-entropy alloys (TiZrHfNb, FeCoCrNiMnAlTiVCu),
   but `amkit/solidification.py::find_liquidus_temperature` silently caps at 2000 K — below every
   RHEA liquidus. Worse, that wrong answer is reachable *through the MCP tool surface*, where a
   consuming agent cannot detect it (§10.7).
3. **Propagation.** `pycalphad-mcp` holds a **non-editable** copy of this fork, because
   `pyproject.toml` declares its path source without `editable = true`. Nothing you fix reaches
   the alloy design loop until that is corrected.
4. **Discoverability.** `pycalphad_list_databases` scans only `pycalphad/tests/databases`. The
   curated output `tdb-forge` exists to produce — `AM_Al_thermo.tdb` (207 KB, the merged
   product), `AM_Al_mobility.tdb`, the HEA/RHEA databases — is **invisible** to the only channel
   that consumes it. Loadable by explicit relative path, but not discoverable (§10.4).

## Ground rules

- **Fork posture: permanent private fork.** Local modules stay in-tree under the `pycalphad/`
  namespace. No upstream contribution. No package extraction.
- **`uv` is the package manager** in both repos. `uv sync`, `uv run pytest`. Compiled Cython
  extensions (`pycalphad/core/*.pyx`) are required and currently build correctly.
- **Prefix shell commands with `rtk`** per `.agents/rules/antigravity-rtk-rules.md`
  (`rtk git status`, `rtk pytest ...`) — 60–90% token savings on command output.
- **This folder is Google Drive-synced.** No scratch files in the repo tree; use the system temp
  directory.
- **Let failures fail.** Phase 1 makes verification *honest*, not *green*. If a system starts
  failing once checks are real, record it and stop. Do not tune tolerances. Do not "fix" the
  underlying thermodynamics — that is explicitly out of scope.
- **The `ONBOARDING.rst` figure of "309 passed" is unverified.** It predates the local additions.
  The 14 local tests are confirmed to *collect*; the suite has **not** been run. Phase 0.3
  establishes the real baseline in **both** repos before any code changes.

## Hard limits

- **Never modify upstream files.** Every change lands in `pycalphad/am/`, `pycalphad/amkit/`,
  `pycalphad/diffusion/`, `pycalphad/precipitation/`, `tdb-forge/`, or `MCP/pycalphad-mcp/`.

  > Off limits: `pycalphad/core/`, `pycalphad/io/`, `pycalphad/model.py`, `pycalphad/mapping/`,
  > `pycalphad/codegen/`, `pycalphad/plot/`, `pycalphad/variables.py`,
  > `pycalphad/property_framework/`.
  >
  > The local delta is currently **14,823 insertions against only 5 deletions**. That ratio is
  > what makes upstream pulls survivable. A local fix to an upstream bug is a merge liability,
  > not a win. If you believe an upstream file must change, **stop and report** instead.
  >
  > `MCP/pycalphad-mcp` is **fully in scope** — this prohibition covers only the PyCalphad tree.

- **No new runtime dependencies.** `kawin==0.5.0`, `scheil==0.3.0`, `fipy>=4.0.3`, plus
  numpy/scipy/sympy/symengine/tinydb/mcp cover everything in this plan.
- `espei` belongs in `tdb-forge/requirements.txt`, **not** `pyproject.toml` — it is pipeline-only.
  That is correct as-is; leave it.
- Never edit `.venv/`, `build/`, `*.egg-info/`, `__pycache__/`, or `uv.lock` **by hand**
  (Phase 0.1 regenerates `uv.lock` via `uv sync`, which is fine).
- **Do not write an invariant solver** (Phase 1). pycalphad already computes invariants during
  the map `verify.py` is already running. See "Reuse these seams" below.
- **Phase 4 must not change physics.** The 14 local tests must pass with identical numerical
  results. If a number moves, stop and report rather than adjusting the test.
- **Phase 1 assumption gate:** the plan infers that an invariant's temperature is
  `PhaseRegionData.ylim[0]`. This is read from the dataclass definition, *not* its producer.
  Validate against the Al-Sc eutectic at 933 K before generalizing. **Do not widen a tolerance to
  force a pass.**

## Reuse these seams (do not invent parallel mechanisms)

**In the fork:**

- `binplot(..., return_strategy=True) -> (ax, strategy)` — `pycalphad/mapping/compat_api.py:6`
  (Phase 1.1: the map is *already* computed for the phase-diagram plot; capture the strategy
  instead of discarding it)
- `BinaryStrategy.get_invariant_data(x, y) -> list[PhaseRegionData]` —
  `pycalphad/mapping/strategy/binary_strategy.py:240`
- `PhaseRegionData = StrategyData`, exposing `.phases`, `.x`, `.y`, `.xlim`, `.ylim` —
  `pycalphad/mapping/strategy/strategy_data.py:60`
- `plot_binary(strategy, ..., ax=None)` — `pycalphad/mapping/plotting.py:181` (creates its own
  axes when none supplied — why Phase 1.2's `ax=ax` is misrouted)
- `MobilityModel.interdiffusivity_matrix` / `.tracer_diffusivity` — `pycalphad/amkit/mobility.py`
  (the single TDB→diffusivity choke point; Phase 4.2 consolidates onto it, never around it)
- `DiffusionCoupleSimulation` — `pycalphad/diffusion/couple.py` (strictly supersedes `binary.py`)
- `susceptibility_from_composition` — `pycalphad/am/cracking.py` (composition → all five indices;
  Phase 3.5 wraps this rather than re-deriving)
- `registry/selection_registry.yaml` — `tdb-forge/registry/` (Phase 1.3 moves verification config
  *here*, beside the DOI entries it describes)

**In the MCP server:**

- `PyCalphadJSONEncoder` and `to_python_type` — `MCP/pycalphad-mcp/pycalphad_mcp/tools.py:77-110`
  (handles numpy scalars, NaN→`null`, ±inf, and the `\x00` padding pycalphad's fixed-width phase
  names carry — every new tool serializes through these)
- `resolve_path` — `tools.py:25` (Phase 0.2 **extends** this to search multiple roots; do not add
  a second resolver)
- `DATABASE_DIR` — `tools.py:23` (Phase 0.2 promotes this single constant to a list of roots)
- The `@mcp.tool()` + `pycalphad_am_*` pattern — `tools.py:349+` (Phase 3's new tools follow this
  shape exactly)
- `parse_conditions` / `parse_condition_value` — `tools.py:43-75` (case-insensitive keys and
  `{"start","stop","step"}` range dicts already handled)

## Definition of done (whole mission)

- [ ] `tdb-forge/reports/al-sc/al-sc_report.md` shows a **computed** invariant temperature with a
      real signed difference — not an echo of the literature value.
- [ ] Perturbing a declared invariant `T` by 50 K halts the pipeline with a **non-zero exit**.
- [ ] `find_liquidus_temperature` on `TiZrHfNb_RHEA.tdb` never silently returns `2000.0`.
- [ ] `pycalphad_list_databases` returns `AM_Al_thermo.tdb` and `TiZrHfNb_RHEA.tdb`, labelled by
      provenance tier.
- [ ] `pycalphad_load_database("TiZrHfNb_RHEA.tdb")` succeeds on the **bare filename**.
- [ ] Editing `pycalphad/precipitation/kinetics.py` changes MCP tool output **without
      reinstalling**, and survives a subsequent `uv sync`.
- [ ] Every AM-backed MCP tool has at least one test (currently **zero**).
- [ ] The 14 local tests pass; no upstream regression against the Phase 0.3 baseline.
- [ ] `git status` is clean of the untracked paths listed in Phase 2.2.

## Open question — resolve with the user before Phase 2.2 closes

`test_layout.drawio` and `pycalphad_architecture_backup.drawio` (both untracked) appear to be
scratch from the architecture-diagram work. Ask whether to commit, `.gitignore`, or delete.
**Do not decide this unilaterally.**

---

## Kickoff message to start a phase

Paste one of these to Antigravity to begin (one phase per conversation):

> Read `IMPROVEMENT_BRIEF.md`, then execute **Phase 0** of `IMPROVEMENT_PLAN.md`.
> Present your implementation plan and wait for my approval before writing code.

Swap in `Phase 1`, `Phase 2`, `Phase 3`, or `Phase 4` for subsequent conversations, in that
order. Phase 0 must complete first — it is what makes every later phase's work visible to
`alloyforge` and its verification meaningful.
