# PyCalphad Fork + MCP Seam — Improvement Plan

**Companion:** `IMPROVEMENT_BRIEF.md` — read it first (mission protocol, hard limits, seams).
**Background:** `CODEBASE_STUDY.md` — §1–9 the fork, **§10 the MCP server**. Current as of 2026-07-26.

Five phases, **executed in order, one phase per conversation**. Check off `- [ ]` tasks as you
complete them — only what you actually did.

Two working directories: `Alloy Agents/PyCalphad` (the fork) and `MCP/pycalphad-mcp` (the seam).
Both are in scope. The upstream-file prohibition applies only to the fork.

---

## Known baseline state

- Fork is upstream `pycalphad` v0.11.3.dev + two commits (`3c21fbf5`, `ecf6bc19`):
  **54 files, 14,823 insertions, 5 deletions.** Purely additive.
- All four local packages (`am`, `amkit`, `diffusion`, `precipitation`) are declared in
  `pyproject.toml` and **import cleanly**; Cython extensions are compiled. Verified.
- `pycalphad-mcp` registers **8 tools** at runtime. Verified by enumeration.
- The 14 local tests **collect** successfully. The suite has **not been run** — the
  `ONBOARDING.rst` figure of "309 passed, 2 skipped, 1 xfailed" predates the local additions and
  is unverified. **Phase 0.3 establishes the real baseline.**
- `MCP/pycalphad-mcp/tests/test_server.py` has 5 tests, all against `Al-Mg_Zhong.tdb`. **None
  touch the three AM tools.**
- Working tree is dirty: 6 modified TDBs, modified `run_pipeline.py` and
  `pycalphad_architecture.drawio`, plus untracked HEA/RHEA work (see Phase 2.2).

---

## Phase 0 — Enabler: make the seam live *and* complete

*Not a priority — a precondition. Until 0.1 and 0.2 land, later phases land invisibly to
`alloyforge`.*

### 0.1 Make the `pycalphad` dependency editable

**Root cause** (`MCP/pycalphad-mcp/pyproject.toml`):

```toml
[tool.uv]
package = false
sources = { pycalphad = { path = "../../Alloy Agents/PyCalphad" } }
```

The path source is declared **without `editable = true`**, so `uv sync` installs a *copy*.
Confirmed: `direct_url.json` reads `"editable": false`; the installed build is stamped
`…d20260716` against a working tree at `…d20260725` — same commit `gecf6bc193`, 9 days apart; and
`uv.lock` contains **zero** `editable` markers.

> ⚠ **Do NOT use `uv pip install -e`.** It appears to work, but the next `uv sync` reconciles the
> environment back to `uv.lock` and silently reverts it. Fix the declaration, not the environment.

- [x] Edit `pyproject.toml`:
      ```toml
      sources = { pycalphad = { path = "../../Alloy Agents/PyCalphad", editable = true } }
      ```
- [x] Run `uv sync` to regenerate `uv.lock`.
- [x] Confirm the Cython extensions still build under the editable install. If the build fails,
      **do not leave it stale** — document a deterministic step and flag it to the user.
- [x] Verify:
      ```bash
      cat .venv/lib/python3.13/site-packages/pycalphad-*.dist-info/direct_url.json   # "editable": true
      .venv/bin/python -c "import pycalphad; print(pycalphad.__file__)"              # resolves into the fork
      .venv/bin/python -c "import pycalphad_mcp.tools; print('ok')"                  # server still imports
      ```
- [x] Run `uv sync` a **second** time and re-check `direct_url.json` — it must remain
      `"editable": true`. This is what proves the fix is durable.

### 0.2 Make curated databases discoverable

**File:** `MCP/pycalphad-mcp/pycalphad_mcp/tools.py:23`.

```python
DATABASE_DIR = os.path.abspath(os.path.join(WORKSPACE_DIR, "Alloy Agents/PyCalphad/pycalphad/tests/databases"))
```

`pycalphad_list_databases` scans **only** that directory — 38 upstream *test* databases.
Invisible to it:

| Not discoverable | Contents |
|---|---|
| `tdb-forge/tdbs/` | `AM_Al_thermo.tdb` (207 KB — the pipeline's merged product), `AM_Al_mobility.tdb`, all `*_compiled.tdb` |
| `examples/databases/` | `FeCoCrNiMnAlTiVCu_HEA.tdb`, `TiZrHfNb_RHEA.tdb`, and 8 others |

The whole point of `tdb-forge` is producing DOI-tracked, verified, merged databases. The only
channel to the alloy design loop cannot enumerate them. Verified behaviour today:

```
resolve_path('Alloy Agents/PyCalphad/tdb-forge/tdbs/AM_Al_thermo.tdb')      → exists ✓
resolve_path('Alloy Agents/PyCalphad/examples/databases/TiZrHfNb_RHEA.tdb') → exists ✓
resolve_path('TiZrHfNb_RHEA.tdb')                                           → NOT found ✗
```

**Loadable but not discoverable** — the worst shape. A caller who already knows the full relative
path succeeds; a caller who asks "what is available?" is told the curated work does not exist.

- [x] Promote `DATABASE_DIR` to a **list of roots**: `pycalphad/tests/databases`,
      `examples/databases`, `tdb-forge/tdbs`.
- [x] Update `pycalphad_list_databases` to walk all roots and **label each entry with its
      provenance tier** — upstream test fixture / example / **tdb-forge verified output** — so a
      consuming agent can prefer curated databases over test fixtures.
- [x] **Extend `resolve_path` (`tools.py:25`)** to search the same roots, so a bare filename such
      as `TiZrHfNb_RHEA.tdb` resolves. Do not add a second resolver.
- [x] Preserve the existing `okf_generated/` subdirectory handling.

**Acceptance:**

- [x] `pycalphad_list_databases` returns `AM_Al_thermo.tdb` and `TiZrHfNb_RHEA.tdb`, each labelled.
- [x] `pycalphad_load_database("TiZrHfNb_RHEA.tdb")` succeeds on the **bare filename**.
- [x] The existing `test_list_databases` assertion (`"Al-Mg_Zhong.tdb" in databases`) still passes.

### 0.3 Capture a true test baseline — both repos

- [x] Run and record **verbatim**, including failures:
      ```bash
      cd "/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad" && uv run pytest -q 2>&1 | tail -20
      cd "/Users/pei/My Drive/Antigravity/MCP/pycalphad-mcp"      && uv run pytest -q 2>&1 | tail -20
      ```
- [x] Record both results in the commit/PR description. Do **not** assume 309.
- [x] Do **not** fix unrelated failures found here — record and move on.

> `CODEBASE_STUDY.md` needs no correction — §9 was fixed and §10 appended on 2026-07-26.

---

## Phase 1 — Verification integrity

*Highest consequence. This is why the mission exists.*

### 1.1 Replace the fake invariant check with a real one

**File:** `tdb-forge/pipeline/verify.py`, function `verify_t2_thermo_kinetic`.

**Current code (~line 100):**

```python
for inv in invariants:
    lit_T = inv['T']
    reaction_type = inv['type']
    print(f"[T2] Checking invariant transition near {lit_T} K")
    report_lines.append(f"| {reaction_type} | {lit_T} | Near {lit_T} (verified) | 0.0 | PASS |")
```

No equilibrium is computed. The literature value is echoed into the "Computed" column, the
difference is hardcoded `0.0`, `PASS` is hardcoded, and `all_passed` is never touched — so this
block **can never fail the build**.

> **Do NOT write an invariant solver.** pycalphad already computes invariants during the map
> `verify.py` is *already running* for the phase-diagram plot. The strategy object is discarded.

- [x] Capture the strategy from the existing `binplot` call (see 1.2 for the corrected call).
- [x] Extract computed invariant temperatures via `strategy.get_invariant_data(X(el), T)`.
- [x] For each declared invariant, find the nearest computed one:
      - within tolerance → `PASS`, reporting the **real signed difference**;
      - no match within tolerance → `FAIL` **and set `all_passed = False`** (halts `run_pipeline.py`).

> ### ⚠ Assumption gate — validate before generalizing
>
> This assumes `get_invariant_data(X(el), v.T)` returns **one `PhaseRegionData` per invariant**
> with effectively constant `y` (invariant reactions are isothermal), so the invariant temperature
> is `ylim[0]` (≈ `ylim[1]`). **Inferred from the dataclass definition, not from its producer.**
>
> - [x] Print the raw `get_invariant_data(...)` structure once and confirm the shape.
> - [x] Validate against the **Al-Sc eutectic at 933 K** (already declared in `run_pipeline.py`'s
>       `configs`) before trusting it for any other system.
> - [x] If the shape differs, adapt the extraction. **Do not widen the tolerance to force a pass.**

### 1.2 Fix the misrouted `ax` on the same call

**File:** `tdb-forge/pipeline/verify.py:63`.

```python
binplot(db, elements + ['VA'], phases,
        {X(elements[1]): (0, 1, 0.02), T: (T_min, T_max, 10), P: 101325}, ax=ax)
```

`ax` is **not** a recognized `binplot` parameter. Per the signature
(`pycalphad/mapping/compat_api.py:6`) it falls into `**map_kwargs` and is forwarded to
`BinaryStrategy(...)`, not to the plotter. `binplot` passes `**plot_kwargs` to `plot_binary`,
which creates its **own** axes when none is supplied (`pycalphad/mapping/plotting.py:181`).

Consequence: the `fig = plt.figure(figsize=(8, 6)); ax = fig.gca()` above it is dead code, and the
saved PNG comes from whatever figure happens to be current — `figsize` and the intended axes are
silently discarded.

- [x] Correct the call:
      ```python
      ax, strategy = binplot(
          db, elements + ['VA'], phases,
          {X(elements[1]): (0, 1, 0.02), T: (T_min, T_max, 10), P: 101325},
          plot_kwargs={'ax': ax},
          return_strategy=True,
      )
      ```
- [x] Save with `ax.get_figure().savefig(plot_path, dpi=150)` rather than `plt.savefig(...)`.

### 1.3 Move verification config into the registry, with absolute tolerances

The T2 `configs` dict is hardcoded in `tdb-forge/pipeline/run_pipeline.py` rather than beside the
entries it describes in `tdb-forge/registry/selection_registry.yaml`. Adding a system currently
means editing Python.

> **Relative tolerance is wrong here.** The existing mobility check's `rtol=1e-2` would permit 9 K
> of error on a 933 K eutectic. Invariant temperatures need an **absolute** tolerance in K.

- [x] Move verification config into `selection_registry.yaml` per entry:
      ```yaml
      - doi: "10.1007/s11669-008-9366-z"
        system: "Al-Sc"
        type: "thermodynamic"
        path: "extracted/alsc_thermo.json"
        supersede_constituents: [["AL", "SC"]]
        verification:
          T_plot_limits: [300, 1000]
          invariants:
            - type: "Eutectic"
              T: 933.0
              tol_K: 5.0
          mobility_checks: []
      ```
- [x] `run_pipeline.py` reads `entry.get('verification', {})`; **delete** the hardcoded `configs` dict.
- [x] Migrate the `Al-Mg-Si-Sc-Zr-Fe-mobility` entry's five `mobility_checks` **verbatim** — they
      are real and currently working. **Do not alter their `rtol=1e-2`.**

### 1.4 Regenerate the tainted reports

`tdb-forge/reports/al-sc/al-sc_report.md` is committed and asserts a verification that never ran.

- [x] Re-run the pipeline after 1.1–1.3 and commit the regenerated reports.

> **If a system now genuinely fails, let it fail.** A red report is the correct output. Record it;
> do not convert it to green. Correcting the underlying thermodynamics is out of scope.

**Phase 1 acceptance:**

- [x] Al-Sc eutectic reports a map-derived computed temperature with a real signed difference vs 933 K.
- [x] Perturbing the declared literature `T` by 50 K causes `run_pipeline.py` to halt non-zero.
- [x] The saved phase-diagram PNG honours `figsize=(8, 6)` and `dpi=150`.

---

## Phase 2 — Unblock refractory HEAs

### 2.1 Remove the 2000 K liquidus ceiling

**File:** `pycalphad/amkit/solidification.py`, function `find_liquidus_temperature`.

```python
T_high = 2000.0
T_low = 300.0
for _ in range(15):   # bisection
```

For TiZrHfNb — Ti 1941 K, Zr 2128 K, Hf 2506 K, Nb 2750 K (per `tdb-forge/tdbs/TiZrHfNb_RHEA.tdb`)
— solid is stable at **every** probe, so `T_low` climbs, `T_high` never moves, and the function
returns exactly `2000.0` **with no warning**. `simulate_solidification` then starts Scheil at
2010 K — mechanically below the true liquidus, inside the mushy zone rather than single-phase
liquid, which is the precondition Scheil assumes.

> **This escapes the repo.** The same path is reachable through
> `pycalphad_am_simulate_solidification` (§10.7), so `alloyforge` can receive a silently wrong
> liquidus with no way to detect it. Fixing 2.1 fixes the MCP path too — **provided 0.1 landed**,
> otherwise the fix does not propagate.

- [x] Promote `T_high` / `T_low` to keyword arguments; preserve current defaults for back-compat.
- [x] **Add bracket detection.** Before bisecting, evaluate at `T_high`. If a non-liquid phase is
      still stable there, the search never bracketed a transition — expand upward to a ceiling
      and, failing that, **raise a clear error**.
      > Silent wrong answers are the defect. Loud failure is the fix.
- [x] *Optional:* seed `T_high` from the maximum pure-element melting point in `dbf.elements`, so
      refractory systems self-configure.
- [x] Thread the new keywords through `simulate_solidification` and
      `pycalphad/am/cracking.py::susceptibility_from_composition`.

### 2.2 Version the untracked work

Untracked today — **both the artifacts and their generators**, so this work is currently losable:

| Path | Contents |
|---|---|
| `tdb-forge/scripts/` | 9 scripts: `create_rhea_tdb.py`, `create_multicomponent_hea_tdb.py`, `improve_tdb.py`, RHEA/EHEA phase-diagram, Scheil, stepped-equilibrium plotters |
| `tdb-forge/tdbs/TiZrHfNb_RHEA.tdb` | Refractory HEA database |
| `examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb` | Cantor-family HEA database |
| `examples/databases/TiZrHfNb_RHEA.tdb` | Refractory HEA database |
| `.agents/` | Antigravity RTK rules |

- [x] Commit the paths above.
- [x] Decide explicitly whether generated `.tdb` outputs belong in git, or in `.gitignore` with the
      generator as source of truth. **Do not** leave the current state where neither the output nor
      the generator is tracked.
- [x] **Ask the user** about `test_layout.drawio` and `pycalphad_architecture_backup.drawio` (look
      like scratch) before committing, ignoring, or deleting.

**Phase 2 acceptance:**

- [x] A regression test asserts `find_liquidus_temperature` on `TiZrHfNb_RHEA.tdb` either returns a
      liquidus above 2000 K or raises a clear bracket error — **never** silently returns `2000.0`.
- [x] The same is true through `pycalphad_am_simulate_solidification`.
- [x] `git status` is clean of the paths above.


---

## Phase 3 — Complete the MCP tool surface

All work in `MCP/pycalphad-mcp/pycalphad_mcp/tools.py`. Follow the existing `pycalphad_am_*`
pattern (`tools.py:349+`): `@mcp.tool()` decorator, JSON-serializable returns via the existing
helpers (`to_python_type` and `PyCalphadJSONEncoder`, `tools.py:77-110`).

### 3.1–3.3 New capability tools

- [x] `pycalphad_precipitation_kinetics` — wraps `PrecipitationKineticsSimulation`; returns time,
      volume fraction, mean radius, precipitate density; yield strength when a strength model is
      attached.
- [x] `pycalphad_diffusion_couple` — wraps `DiffusionCoupleSimulation`; returns coordinates +
      per-element composition profiles after a requested simulated duration.
- [x] `pycalphad_am_eagar_tsai_temp` — `eagar_tsai_T` is already imported at `tools.py:14` but
      never exposed. The distributed-source model complementing the exposed Rosenthal point source.

### 3.4 Expose the 2D thermal solver

- [x] `solve_thermal_profile` is imported at `tools.py:16` and never used — the second dead import.
      Expose it, returning the temperature field plus a probe time-history via
      `extract_thermal_history_probe`.

### 3.5 Complete the cracking tool

`pycalphad_am_cracking_index` returns only Clyne-Davis and Kou. `am.cracking` also provides
`calculate_rdg_index`, and `susceptibility_from_composition` additionally yields `freezing_range`
and `tfr`. The current tool also takes pre-computed *T*/*f_s* arrays, so a caller must run
solidification separately and stitch two calls together.

- [x] Add `calculate_rdg_index` to the existing array-based tool.
- [x] Add a **composition-based** variant wrapping `susceptibility_from_composition`, returning all
      five indices in one call.

### 3.6 Hygiene

- [x] Rename the `v` parameter in `pycalphad_am_rosenthal_temp` (`tools.py:424`) — it shadows
      `import pycalphad.variables as v` (`tools.py:12`). Harmless today because that function never
      touches `v.X`/`v.T`, but it is a live trap for the next edit.
- [x] Make `pycalphad_plot_phase_diagram` return a JSON error string on failure, like every other
      tool. It currently returns an `Image` on success and a bare `str` on error, so a caller
      parsing JSON is surprised on the error path.
- [x] Replace mutable `conditions: Dict[str, Any] = {}` defaults (four tools) with `None` +
      in-body default.

### 3.7 Cost bounds

No timeout or grid cap exists on `pycalphad_equilibrium`, `pycalphad_calculate`, or
`simulate_solidification`. The consumer is an autonomous agent, not a human who will notice a
runaway call.

- [x] Cap grid size, timestep count, and simulated duration on the expensive tools.
- [x] State each cap in the tool's docstring so the calling agent can reason about cost.

### 3.8 README

- [x] `MCP/pycalphad-mcp/README.md`'s Features list names 5 capabilities and omits every AM tool.
      Refresh it to match the actual surface.

**Phase 3 acceptance:**

- [x] **A test per AM-backed tool** — `pycalphad_am_simulate_solidification`,
      `pycalphad_am_cracking_index`, `pycalphad_am_rosenthal_temp`, and each new tool. Currently
      **zero** of these are tested.
- [x] The existing 5 tests still pass.
- [x] A change made to `pycalphad/precipitation/kinetics.py` is observable through the MCP tool
      **without reinstalling** — the end-to-end proof that Phase 0.1 worked.


---

## Phase 4 — Consolidate the local layer

Cleanup only. **No behaviour change** — physics must be numerically identical.

### 4.1 Remove dead compute in `pycalphad/amkit/mobility.py`

In `interdiffusivity_matrix`'s multicomponent branch, `kawin_tracer_diffusivity(...)` is fully
computed then **immediately overwritten** by `inverseMobility(...)`; the leftover comment confirms
abandoned scaffolding. This sits inside the diffusion solver's inner loop —
`DiffusionCoupleSimulation._step_implicit_numpy` calls `interdiffusivity_matrix` **once per grid
cell per timestep** — so removal roughly halves the cost of the module's hottest path.

- [x] Delete the overwritten `kawin_tracer_diffusivity` call and its now-unused import.

### 4.2 Single source of truth for Darken-Manning

The expression `((1-x_B)·M_B + x_B·M_A)·x_B·dμ_B/dx_B` exists in **three** places:
`amkit/mobility.py`, `diffusion/couple.py` (via the mobility model), and `diffusion/binary.py`
(re-derived with hardcoded default mobilities). `binary.py` is an earlier prototype superseded by
`couple.py`, which is strictly more capable.

- [x] Confirm consumers first — `pycalphad/tests/test_diffusion.py` and `MCP/pycalphad-mcp`.
- [x] Deprecate `BinaryDiffusionSimulation`: keep it as a thin shim constructing a
      `DiffusionCoupleSimulation(solver='fipy')`, emitting a `DeprecationWarning`.
- [x] **Do not delete outright in this phase.**

### 4.3 Fix `pycalphad/am/__init__.py` exports

It re-exports only `calculate_clyne_davis_index` and `calculate_kou_index`, omitting
`calculate_rdg_index` and `susceptibility_from_composition` — the latter being the module's natural
composition → indices entry point. `test_integration_pipeline.py` already reaches past the package
into `pycalphad.am.cracking` to get it.

- [x] Add both to `__init__.py`.

### 4.4 Smaller items

- [x] `pycalphad/am/heat_sources.py`, `ConicalHeatSource.__call__`: delete `volume_integral` —
      computed and never used; `coeff` re-derives the same normalisation inline.
- [x] `pycalphad/am/thermal.py`: under `strict_stability=False` the loop continues on `np.clip`-ed
      values after warning, turning a divergent solve into a plausible-looking field. Either abort
      after N consecutive warnings, or mark the returned field unconverged.
- [x] Replace bare `print()` in `diffusion/binary.py::__init__` and
      `precipitation/kinetics.py::plot_results` with `logging` — library code should not print.

**Phase 4 acceptance:**

- [x] The 14 local tests pass with **identical numerical results**. If a number moves, stop and
      report rather than adjusting the test.
- [x] A before/after timing of `test_diffusion.py::test_multicomponent_diffusion_couple` is
      recorded, evidencing the 4.1 speedup.


---

## Verification

```bash
# 1. Local layer intact (14 tests)
cd "/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad"
uv run pytest pycalphad/tests/test_am.py pycalphad/tests/test_amkit.py \
              pycalphad/tests/test_diffusion.py pycalphad/tests/test_precipitation.py \
              pycalphad/tests/test_integration_pipeline.py -v

# 2. No upstream regression — compare against the Phase 0.3 baseline, NOT against "309"
uv run pytest

# 3. Database pipeline with real invariant checks
cd tdb-forge && uv run pytest tests/test_golden.py && uv run python -m pipeline.run_pipeline

# 4. Seam live, discoverable, and extended
cd "/Users/pei/My Drive/Antigravity/MCP/pycalphad-mcp" && uv run pytest tests/
```

**Manual checks:**

- [ ] `tdb-forge/reports/al-sc/al-sc_report.md` shows a computed invariant temperature, not an echo.
- [ ] Perturbing a declared invariant `T` by 50 K halts the pipeline non-zero.
- [ ] `find_liquidus_temperature` on `TiZrHfNb_RHEA.tdb` does not return `2000.0`.
- [ ] `pycalphad_list_databases` lists `AM_Al_thermo.tdb` with its provenance tier.
- [ ] `pycalphad_load_database("TiZrHfNb_RHEA.tdb")` works on the bare filename.
- [ ] Editing `pycalphad/precipitation/kinetics.py` changes MCP tool output with no reinstall, and
      survives a subsequent `uv sync`.

---

## Out of scope

- **Any** modification to upstream fork files: `pycalphad/core/`, `io/`, `model.py`, `mapping/`,
  `codegen/`, `plot/`, `variables.py`, `property_framework/`. If you believe one must change,
  **stop and report**. (`MCP/pycalphad-mcp` is fully in scope.)
- Making a currently-failing assessment pass. Phase 1 makes verification *honest*; correcting the
  underlying thermodynamics is separate work.
- Extracting the local layer into its own package — explicitly rejected (permanent fork posture).
- Physics changes in Phase 4.
- Fixing unrelated test failures discovered in the Phase 0.3 baseline — record, don't fix.
- New runtime dependencies.
