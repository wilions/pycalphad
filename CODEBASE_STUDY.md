# PyCalphad Fork + `pycalphad-mcp` — Codebase Study

**Date:** 2026-07-25 (§1–9), 2026-07-26 (§10 + §9 correction)
**Subjects:**
- `Alloy Agents/PyCalphad` — the fork (§1–9)
- `MCP/pycalphad-mcp` — the server that exposes it to `alloyforge` (§10)

**Method:** Read-only analysis. No source code was modified; findings are reported, not
fixed. Claims marked *verified* were confirmed by execution — imports, runtime tool
enumeration, path resolution — and are shown with their output. Claims inferred from
reading are labelled as such.

---

## 1. What this repository actually is

This is **not** a bespoke codebase. It is a checkout of the upstream open-source
`pycalphad` library (branch `develop`, v0.11.3.dev) with a **small, purely additive
local delta** on top.

```
git diff --stat origin/develop..HEAD
  → 54 files changed, 14,823 insertions(+), 5 deletions(-)
```

Only **5 deletions**. Nobody has forked or altered upstream behaviour — new subsystems
have been bolted onto a stable base. Upstream documentation therefore remains valid,
and the real subject of study is the local delta.

The delta is **two commits**:

| Commit | Content |
|---|---|
| `3c21fbf5` | feat: AM, diffusion, precipitation kinetics modules + integration tests |
| `ecf6bc19` | fix: AM packages configuration; thermal limit adjustment |

Everything below the `pycalphad/{am,amkit,diffusion,precipitation}` line, plus the
entire `tdb-forge/` directory, is local work. Everything else is upstream.

### Intellectual direction, readable from `pyproject.toml` alone

The local commit adds exactly three scientific dependencies:

| Package | Pin | Brings |
|---|---|---|
| `kawin` | `==0.5.0` | KWN precipitation kinetics, mobility/diffusivity from TDB |
| `scheil` | `==0.3.0` | Scheil-Gulliver non-equilibrium solidification |
| `fipy` | `>=4.0.3` | Finite-volume PDE solver |

Upstream pycalphad computes **equilibrium** thermodynamics. All three additions are
**time-dependent**: kinetics and transport. The thesis of this fork is *extending
CALPHAD from equilibrium into process simulation for additive manufacturing.*

---

## 2. Architecture

An architecture map already exists in-repo: **`pycalphad_architecture.drawio`**
(committed in `3c21fbf5`, currently modified in the working tree). It is accurate.
Its three-layer decomposition is the right mental model:

```
┌─ 1. DATA PIPELINE (tdb-forge)  ─────────────── LOCAL ──┐
│  JSON assessment specs + SGTE unaries                  │
│      → compile.py → compiled TDB → verify → merge      │
└────────────────────────────────────────────────────────┘
                          ↓ .tdb files
┌─ 2. THERMODYNAMIC ENGINE (pycalphad) ─────── UPSTREAM ─┐
│  io/ parser → Database → Model (SymEngine AST)         │
│      → codegen/ → PhaseRecord (Cython/LLVM)            │
│      → core/ eqsolver + minimizer → xarray.Dataset     │
└────────────────────────────────────────────────────────┘
                          ↓ equilibrium results
┌─ 3. PHYSICAL SIMULATION LAYER  ─────────────── LOCAL ──┐
│  amkit.mobility │ solidification │ precipitation       │
│  diffusion.couple/binary │ am.thermal + am.cracking    │
└────────────────────────────────────────────────────────┘
```

### 2.1 Upstream spine (describe, don't audit)

Read in dependency order, not alphabetically:

| Stage | Files | Role |
|---|---|---|
| Parse | `io/grammar.py`, `io/tdb.py`, `io/database.py`, `io/cs_dat.py` | TDB/ChemSage text → `Database` (a TinyDB of parameters) |
| Model | `model.py` (1,655 lines), `models/model_mqmqa.py` | Gibbs energy as a symbolic SymEngine expression (CEF / MQMQA) |
| Compile | `codegen/phase_record_factory.py`, `codegen/sympydiff_utils.py` | Symbolic AST → compiled callables |
| Solve | `core/minimizer.pyx`, `core/eqsolver.pyx`, `core/composition_set.pyx`, `core/hyperplane.pyx` | The hot path — Cython. Lower convex hull → Newton-Raphson Gibbs minimisation |
| Orchestrate | `core/workspace.py`, `property_framework/` | Modern reactive API + `pint` unit handling |
| Trace | `mapping/` (strategies: binary / ternary / isopleth) | Zero-phase-fraction boundary tracing for phase diagrams |
| Draw | `plot/`, `plot/binary/` | Matplotlib output |

`core/*.pyx` requires compiled C extensions. **Verified working** in `.venv`
(see §5).

### 2.2 Local layer — module by module

**`pycalphad/amkit/` — analytical / service layer (482 lines)**

- `thermal.py` — closed-form melt-pool solutions: `rosenthal_T` (moving point
  source), `eagar_tsai_T` (distributed Gaussian source, evaluated by `scipy.quad`).
  `ThermalHistory` wraps a *t, T* trace with `cooling_rate(T_ref)` and
  `time_above(T)`. `solidification_conditions` returns the **G, R, cooling-rate**
  triple — the standard solidification-microstructure control variables.
- `solidification.py` — thin facade over the `scheil` package;
  `find_liquidus_temperature` (binary search on `equilibrium`) and
  `simulate_solidification(mode='scheil'|'equilibrium')`.
- `mobility.py` (240 lines, the densest local file) — `MobilityModel`, a facade
  over Kawin. Validates that MQ/MF or DQ/DF parameters exist in the TDB before
  proceeding (raises `MobilityDataError`), supports per-element user constant
  overrides, and pre-computes a 200-point `CubicSpline` of the thermodynamic
  factor `dμ_B/dx_B` for binaries. Exposes `tracer_diffusivity` and
  `interdiffusivity_matrix`.

**`pycalphad/am/` — numerical AM simulation (469 lines)**

- `heat_sources.py` — three callable source models: `GaussianHeatSource`
  (surface, W/m²), `DoubleEllipsoidalHeatSource` (Goldak, volumetric),
  `ConicalHeatSource` (keyhole-ish, volumetric).
- `thermal.py` — explicit-Euler 2D transient heat equation on a moving laser
  pass. Handles temperature-dependent ρ/cₚ/k as callables, latent heat via the
  effective-heat-capacity method over a mushy zone, adiabatic (Neumann)
  boundaries, and enforces the explicit stability limit (`strict_stability`).
- `cracking.py` — hot-tearing susceptibility indices computed from a Scheil
  curve: Clyne-Davis CSC, Kou CSI, simplified RDG, freezing range, TFR.
  `susceptibility_from_composition` is the composition → indices entry point.

**`pycalphad/diffusion/` — 1D transport (347 lines)**

- `binary.py` — `BinaryDiffusionSimulation`. FiPy FVM; pre-tabulates chemical
  potentials on a 200-point grid, interpolates `d(μ_B−μ_A)/dx_B`, applies the
  Darken-Manning interdiffusivity.
- `couple.py` — `DiffusionCoupleSimulation`. The more capable one: arbitrary
  component count, **fully implicit block-sparse** solver (`scipy.sparse`)
  assembling an *(N·Nc)²* system with full cross-component `D` coupling. Gets
  its `D` matrix from `amkit.MobilityModel`. Keeps a FiPy path for binaries.

**`pycalphad/precipitation/kinetics.py` (226 lines)**

`PrecipitationKineticsSimulation` — wrapper over Kawin's KWN `PrecipitateModel`.
Notable: `resolve_molar_volume` scrapes `V0`/`VM` out of the TDB and resolves
symbols/free variables to get a number; temperature accepts a scalar, a callable
`T(t)`, or a `(times, temps)` profile for non-isothermal heat treatments;
`add_strength_model` couples an Orowan yield-strength model on top.

### 2.3 Dependency direction inside the local layer

```
am.cracking ──────► amkit.solidification ──► scheil ──► pycalphad.equilibrium
diffusion.couple ─► amkit.mobility ────────► kawin
precipitation ────► amkit.mobility ────────► kawin
diffusion.binary ─► pycalphad.equilibrium + fipy   (does NOT use amkit)
```

`amkit` is the shared service layer; `am`, `diffusion`, and `precipitation` are
consumers. `diffusion/binary.py` is the odd one out — it re-implements the
Darken-Manning interdiffusivity that `amkit.mobility` already provides, with
hardcoded default mobilities. It reads as an earlier prototype superseded by
`couple.py`.

---

## 3. `tdb-forge/` — the database provenance pipeline

This is the most *architecturally opinionated* local work: a build system that
treats thermodynamic databases as **compiled artifacts with provenance**, rather
than hand-edited text.

```
registry/selection_registry.yaml   (DOI + system + supersede policy)
        │
        ▼
extracted/*.json  ──[jsonschema]──►  pipeline/compile.py  ──►  tdbs/*_compiled.tdb
                                                                    │
        ┌───────────────────────────────────────────────────────────┤
        ▼                    ▼                          ▼
   T1 mechanical      T2 thermo-kinetic          T3 integration
   (parses? every     (phase diagram, invariants, (collision detection vs
    phase builds a     diffusivity vs D0/Q ref)    declared supersede list)
    Model?)                     │
        └───────────────────────┴──────────► pipeline/merge.py ──► AM_Al_thermo.tdb
                                                                   AM_Al_mobility.tdb
```

Design decisions worth naming:

- **JSON is the source of truth, TDB is the build output.** `assessment.schema.json`
  (115 lines) constrains the input; `compile.py` validates against it before
  emitting anything.
- **Piecewise continuity is enforced at compile time.** `check_piecewise_continuity`
  rejects gaps *or* overlaps between temperature ranges — a class of silent TDB
  corruption that normally only shows up as a discontinuous phase diagram.
- **Supersede policy is explicit and declared.** `selection_registry.yaml` names the
  constituent arrays a new assessment is allowed to override; `verify_t3_integration`
  **fails the build** on any collision *not* on that list. This is the strongest
  integrity mechanism in the repo.
- **Every entry carries a DOI.** Provenance is a first-class field.
- **Golden tests** (`tests/test_golden.py`) round-trip `alzn_mey` and `alni_dupin`
  JSON → TDB and compare against the reference TDBs shipped in
  `pycalphad/tests/databases/`.
- `fit_gap.py` is an **ESPEI demonstrator**: refits the Al-Zn liquid interaction
  parameter from `datasets/alzn_hm_mix.json` and asserts it lands within 0.1% of
  the published Mey (1993) value of 10465.3 J/mol. A proof that the gap-filling
  path works, not a production step. (`espei` is declared in
  `tdb-forge/requirements.txt`, not in `pyproject.toml` — correct, since it is a
  pipeline-only tool.)

---

## 4. Current working state — the live frontier

The tree is dirty, and the pattern of dirt tells you what is being worked on **right
now: high-entropy alloys.**

**Modified (tracked):**
`pycalphad_architecture.drawio`, `tdb-forge/pipeline/run_pipeline.py`, and six TDBs
including both merged outputs `AM_Al_thermo.tdb` / `AM_Al_mobility.tdb`.

**Untracked (⚠ unversioned):**

| Path | Note |
|---|---|
| `tdb-forge/scripts/` | 9 scripts: RHEA/EHEA phase diagrams, Scheil plots, `create_rhea_tdb.py`, `create_multicomponent_hea_tdb.py`, `improve_tdb.py` |
| `tdb-forge/tdbs/TiZrHfNb_RHEA.tdb` | Refractory HEA database |
| `examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb` | Cantor-family HEA database |
| `examples/databases/TiZrHfNb_RHEA.tdb` | " |
| `.agents/`, `test_layout.drawio`, `pycalphad_architecture_backup.drawio` | |

The HEA/RHEA databases and the entire `scripts/` directory are **not under version
control**. Given they are generated artifacts *and* their generators are also
untracked, losing this directory would lose the current work. Flagging only — I have
not staged anything.

---

## 5. Build state — verified empirically

Commit `ecf6bc19` claims to "resolve AM packages configuration." It did. Confirmed by
execution rather than by reading the message:

```
$ .venv/bin/python -c "import pycalphad; import pycalphad.am, pycalphad.amkit,
                       pycalphad.diffusion, pycalphad.precipitation; ..."
pycalphad 0.11.3.dev10+gecf6bc193.d20260725
am/amkit/diffusion/precipitation: OK
kawin+scheil+fipy: OK, fipy 4.0.3
compiled cython extensions: OK
```

All four local packages are declared in `pyproject.toml` `packages` and import
cleanly. Cython extensions are compiled and importable.

**Tests: collected, not run.** The 14 local tests collect successfully:

```
test_am.py (4) · test_amkit.py (3) · test_diffusion.py (2)
test_precipitation.py (2) · test_integration_pipeline.py (1)   → 14 collected
```

I did **not** execute the suite (it invokes real equilibrium solves and Kawin
simulations, which are slow). Upstream `ONBOARDING.rst` documents a baseline of
"309 passed, 2 skipped, 1 xfailed" — that number predates the local additions and
has not been re-verified here. **No claim is made that tests currently pass.**

---

## 6. Findings

Read-only study — nothing was fixed. Ordered by consequence.

### 6.1 T2 invariant verification is a stub that always reports PASS ⚠ highest

`tdb-forge/pipeline/verify.py`, in `verify_t2_thermo_kinetic`:

```python
for inv in invariants:
    lit_T = inv['T']
    reaction_type = inv['type']
    print(f"[T2] Checking invariant transition near {lit_T} K")
    report_lines.append(f"| {reaction_type} | {lit_T} | Near {lit_T} (verified) | 0.0 | PASS |")
```

No equilibrium is computed. The literature temperature is echoed back into the
"Computed T" column, the difference is hardcoded to `0.0`, and the status is
hardcoded to `PASS`. `all_passed` is never touched by this block.

The consequence is worse than a missing check: `reports/al-sc/al-sc_report.md` is a
generated artifact that **states an invariant was verified when nothing was
calculated** — and that file is **committed to the repository** (added in `3c21fbf5`).
This is not merely an unimplemented stub; it is a false verification record checked
into version control, which is exactly the failure class the prior DATA_INTEGRITY work
was aimed at. The comment above it ("For simplicity, we compare...") describes an
intended implementation that was never written. The diffusivity check in the same
function *is* real (it evaluates MQ against D0/Q with a 1% tolerance) — which makes
the fake row harder to spot by eye.

### 6.2 `find_liquidus_temperature` silently caps at 2000 K — bites refractory HEAs

`pycalphad/amkit/solidification.py`:

```python
T_high = 2000.0
T_low = 300.0
for _ in range(15):   # binary search
```

If the true liquidus exceeds 2000 K, solid is stable at every probe temperature, so
`T_low` climbs and `T_high` never moves. The function returns **exactly 2000.0** with
no warning, and `simulate_solidification` then starts Scheil at 2010 K — mechanically
*below* the true liquidus, i.e. inside the mushy zone rather than in single-phase
liquid, which is the precondition Scheil assumes. (Claim is from reading the code; I
did not run a RHEA solidification to observe the numerical outcome.)

This is directly relevant to the current frontier. From `TiZrHfNb_RHEA.tdb`:

| Element | Melting point |
|---|---|
| Ti | 1941 K |
| Zr | 2128 K |
| Hf | 2506 K |
| Nb | 2750 K |

A TiZrHfNb liquidus sits well above 2000 K. **Latent, not yet biting** — no script in
`tdb-forge/scripts/` currently calls `find_liquidus_temperature` or
`simulate_solidification` on the RHEA database. It will bite the moment Scheil or
cracking analysis is pointed at refractory compositions.

Related: `T_high`/`T_low` are not parameters, and there is no post-loop check that the
search actually bracketed a transition.

### 6.3 Hardcoded absolute path in the pipeline driver

`tdb-forge/pipeline/run_pipeline.py:10`:

```python
BACKBONE_PATH = '/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad/pycalphad/tests/databases/COST507.tdb'
```

Every other path in the file is derived from `PROJECT_ROOT`. This one is absolute and
machine-specific, so the pipeline is unrunnable on any other machine or CI. It is also
in the currently-modified set.

Secondary: the T2 `configs` dict is hardcoded in `run_pipeline.py` rather than living
in `selection_registry.yaml` next to the entries it describes, so adding a system
means editing Python, not the registry.

### 6.4 Dead compute in the multicomponent interdiffusivity path

`pycalphad/amkit/mobility.py`, `interdiffusivity_matrix`, multicomponent branch:

```python
Dnkj, _, _ = kawin_tracer_diffusivity(...)   # result immediately discarded
# Note: ... Let's call the correct kawin function or compute it using our callables
from kawin.thermo.Mobility import inverseMobility
Dnkj, _, _ = inverseMobility(...)            # overwrites the above
```

The first call is fully computed then overwritten. Harmless numerically, but it is
per-cell work inside the diffusion solver's inner loop — `_step_implicit_numpy` calls
`interdiffusivity_matrix` once per grid cell per timestep — so it roughly doubles the
cost of the most expensive path in the module. The leftover comment confirms it is
abandoned scaffolding.

### 6.5 Public API omissions in `pycalphad/am/__init__.py`

```python
from .cracking import calculate_clyne_davis_index, calculate_kou_index
```

`calculate_rdg_index` and `susceptibility_from_composition` are not re-exported,
despite the latter being the module's natural entry point (composition → indices).
Callers must reach into `pycalphad.am.cracking` directly.

### 6.6 Smaller items

- `am/heat_sources.py`, `ConicalHeatSource.__call__`: `volume_integral` is computed
  and never used (`coeff` re-derives the same normalisation inline).
- `am/thermal.py`: the instability guard runs *before* `np.clip(T_new, T_ambient, …)`,
  so the first excursion is caught — but under `strict_stability=False` the loop then
  continues on clipped values, quietly converting a divergent solve into a plausible-
  looking field.
- `diffusion/binary.py`: bare `print("Pre-calculating thermodynamic functions...")` in
  `__init__`; the rest of the local layer is silent. `precipitation/kinetics.py` also
  prints on save.
- `diffusion/binary.py` duplicates the Darken-Manning expression in
  `amkit/mobility.py` and `diffusion/couple.py` — three copies of the same physics.

---

## 7. Where to look

| I want to... | Go to |
|---|---|
| Add a thermodynamic assessment | `tdb-forge/registry/selection_registry.yaml` + `tdb-forge/extracted/*.json` |
| Change what the compiler accepts | `tdb-forge/schemas/assessment.schema.json` |
| Change database merge/supersede rules | `tdb-forge/pipeline/merge.py`, `verify.py::verify_t3_integration` |
| Add a heat source model | `pycalphad/am/heat_sources.py` |
| Add a hot-cracking index | `pycalphad/am/cracking.py` (+ export it in `am/__init__.py`) |
| Change how D is obtained from a TDB | `pycalphad/amkit/mobility.py` |
| Add a solidification mode | `pycalphad/amkit/solidification.py` |
| Touch the equilibrium solver | `pycalphad/core/minimizer.pyx` — **upstream**, changes create merge debt |
| Understand the whole picture | `pycalphad_architecture.drawio` |

## 8. Common tasks

```bash
uv sync                                    # install (uv is the documented manager)
uv run pytest                              # full suite (upstream baseline: 309 passed)
uv run pytest pycalphad/tests/test_am.py   # local tests only
cd tdb-forge && python -m pipeline.run_pipeline   # DB build (see finding 6.3 first)
cd tdb-forge && pytest tests/test_golden.py       # DB round-trip regression
```

## 9. External integration seam

The sibling project `../alloyforge` does **not** import this fork. It reaches CALPHAD
through an MCP server over stdio:

```
alloyforge/l3_optimization/calphad_backend.py
  → resolve_mcp_server("pycalphad-mcp", "ALLOYFORGE_PYCALPHAD_...")
  → mcp.client.stdio  →  pycalphad-mcp server  →  pycalphad
```

So this repo is coupled to alloyforge only through the `pycalphad-mcp` server's tool
surface and the `.tdb` files `tdb-forge` produces — a process boundary, not an import.

The local layer is **partially** reachable. `pycalphad-mcp` does import this fork's
modules and exposes three AM tools; `diffusion` and `precipitation` have none. See
§10 for the full analysis of that server.

> **Correction.** An earlier revision of this section stated that *"none of the local
> `am` / `amkit` / `diffusion` / `precipitation` capability is reachable from alloyforge
> today."* That is false — three AM tools are exposed. The corrected picture is in §10.

---

# 10. `pycalphad-mcp` — the server study

**Location:** `/Users/pei/My Drive/Antigravity/MCP/pycalphad-mcp` (outside this repo)
**Size:** 529 lines total — `pycalphad_mcp/tools.py` (451), `server.py` (17), `tests/test_server.py` (55)
**Framework:** `mcp[cli]>=1.2.0`, FastMCP, stdio transport
**Verified:** 8 tools register at runtime; `import pycalphad_mcp.tools` succeeds.

This is the **only** channel between the fork and `alloyforge`. It is small, readable,
and consistently written — the issues below are about *reach* and *staleness*, not craft.

## 10.1 Architecture

```
alloyforge/l3_optimization/calphad_backend.py
   └─ resolve_mcp_server("pycalphad-mcp", ...) → stdio
        └─ server.py → mcp.run()
             └─ pycalphad_mcp/server.py   FastMCP("pycalphad"), logging → stderr
                  └─ from .tools import *      ← star-import triggers @mcp.tool() registration
                       └─ pycalphad + the fork's am/amkit
```

`server.py:9` routes logging to **stderr** deliberately, so it cannot corrupt the
JSON-RPC stream on stdout. Correct and worth preserving.

Registration happens as an import side effect (`from .tools import *`). It works — 8 tools
confirmed live — but means any import error in `tools.py` silently yields a server with
*fewer* tools rather than a crash.

## 10.2 Tool surface (8 tools, verified at runtime)

| Tool | Backed by | Notes |
|---|---|---|
| `pycalphad_list_databases` | `os.listdir(DATABASE_DIR)` | **See 10.4 — major blind spot** |
| `pycalphad_load_database` | `Database` | Returns components/elements/phases |
| `pycalphad_equilibrium` | `equilibrium` | Full grid serialization |
| `pycalphad_calculate` | `calculate` | Per-property loop |
| `pycalphad_plot_phase_diagram` | `binplot` / `ternplot` | Returns a PNG `Image` |
| `pycalphad_am_simulate_solidification` | `amkit.solidification` | **local layer** |
| `pycalphad_am_cracking_index` | `am.cracking` | **local layer** |
| `pycalphad_am_rosenthal_temp` | `amkit.thermal` | **local layer** |

Nothing from `pycalphad/diffusion/` or `pycalphad/precipitation/` is exposed.

## 10.3 What the server does well

- **Defensive serialization.** `PyCalphadJSONEncoder` (`tools.py:89`) handles numpy scalars,
  arrays, sets, bytes, NaN → `null`, and ±inf → `"Infinity"`. `to_python_type` strips the
  `\x00` padding that pycalphad's fixed-width phase-name arrays carry — a real trap, handled.
- **Input normalization.** `parse_conditions` accepts `T`/`t`, `X_MG`/`x_mg`, and range dicts
  `{"start":…, "stop":…, "step":…}`. Components and phases are upper-cased. A test pins the
  lowercase path.
- **Errors returned, not raised.** Every tool wraps in `try/except` and returns
  `{"success": false, "error": …}`, so a bad call cannot kill the session.
- **Figure hygiene.** The plot tool closes its figure in a `finally` block — no leak across calls.

## 10.4 Finding: curated databases are invisible ⚠ highest

`tools.py:23`:

```python
DATABASE_DIR = os.path.abspath(os.path.join(WORKSPACE_DIR, "Alloy Agents/PyCalphad/pycalphad/tests/databases"))
```

`pycalphad_list_databases` scans **only** that directory — 38 upstream *test* databases
(plus an `okf_generated/` subdirectory if present). Invisible to it:

| Not discoverable | Contents |
|---|---|
| `tdb-forge/tdbs/` | `AM_Al_thermo.tdb` (207 KB — the pipeline's merged product), `AM_Al_mobility.tdb`, all `*_compiled.tdb` |
| `examples/databases/` | `FeCoCrNiMnAlTiVCu_HEA.tdb`, `TiZrHfNb_RHEA.tdb`, and 8 others |

The whole point of `tdb-forge` is producing DOI-tracked, verified, merged databases. The
only channel to the alloy design loop cannot enumerate them.

**Loadable but not discoverable** — the worst shape. Verified:

```
resolve_path('Alloy Agents/PyCalphad/tdb-forge/tdbs/AM_Al_thermo.tdb')  → exists ✓
resolve_path('Alloy Agents/PyCalphad/examples/databases/TiZrHfNb_RHEA.tdb') → exists ✓
resolve_path('TiZrHfNb_RHEA.tdb')                                        → NOT found ✗
```

`resolve_path` falls back to `WORKSPACE_DIR`, so a caller who already knows the full
relative path succeeds. A caller who asks "what is available?" is told the curated work
does not exist. Fix: make `DATABASE_DIR` a *list* of roots covering `tests/databases`,
`examples/databases`, and `tdb-forge/tdbs`, and have `list_databases` label each result
with its provenance tier.

## 10.5 Finding: the install is non-editable — root cause found

`pyproject.toml`:

```toml
[tool.uv]
package = false
sources = { pycalphad = { path = "../../Alloy Agents/PyCalphad" } }
```

The path source is declared **without `editable = true`**, so `uv sync` installs a
*copy*. Confirmed by `direct_url.json` (`"editable": false`) and by the version stamps —
installed `…d20260716` against a working tree at `…d20260725`, same commit `gecf6bc193`,
9 days apart.

**Precise fix** (better than a one-off `uv pip install -e`, which the next `uv sync`
would undo):

```toml
sources = { pycalphad = { path = "../../Alloy Agents/PyCalphad", editable = true } }
```

Consequence while unfixed: every improvement to `am`/`amkit`/`diffusion`/`precipitation`
is invisible to `alloyforge` until someone manually reinstalls.

## 10.6 Finding: the AM surface has zero test coverage

`tests/test_server.py` has 5 tests, all against `Al-Mg_Zhong.tdb`, covering
`list_databases`, `load_database`, `equilibrium` (×2, one for lowercase normalization),
and `calculate`.

**No test touches `pycalphad_am_simulate_solidification`, `pycalphad_am_cracking_index`,
or `pycalphad_am_rosenthal_temp`** — i.e. the fork's entire contribution is untested at
the boundary where alloyforge consumes it. `test_list_databases` asserts a *test* database
is present, so it would also pass unchanged if 10.4 were fixed or broken.

## 10.7 Finding: the 2000 K liquidus cap reaches alloyforge

`pycalphad_am_simulate_solidification` → `amkit.simulate_solidification` →
`find_liquidus_temperature`, which hard-caps at `T_high = 2000.0` (§6.2). Called against
`TiZrHfNb_RHEA.tdb` — whose constituents melt at 1941–2750 K — it returns exactly `2000.0`
with no warning and starts Scheil below the true liquidus.

So the defect is not confined to this repo: it is **reachable as a wrong answer through
the MCP tool surface**, where a consuming agent has no way to detect it. Fixing §6.2 fixes
it here too, provided 10.5 is fixed so the change propagates.

## 10.8 Smaller findings

- **Dead imports.** `solve_thermal_profile` (`tools.py:16`) and `eagar_tsai_T`
  (`tools.py:14`) are imported and never used. Both are ready-made tools left unexposed —
  `eagar_tsai_T` in particular is the distributed-source model that complements the
  already-exposed Rosenthal point source.
- **Incomplete cracking output.** `pycalphad_am_cracking_index` returns only Clyne-Davis
  and Kou. `am.cracking` also provides `calculate_rdg_index`, and
  `susceptibility_from_composition` additionally yields `freezing_range` and `tfr` — none
  surfaced. The tool also takes pre-computed *T*/*f_s* arrays rather than a composition,
  so the caller must run solidification separately and stitch the two calls together.
- **Name shadowing.** `pycalphad_am_rosenthal_temp` names its scan-speed parameter `v`
  (`tools.py:424`), shadowing the module-level `import pycalphad.variables as v`
  (`tools.py:12`). Harmless today because that function never touches `v.X`/`v.T`, but it
  is a live trap for the next edit. The same name is reused as a comprehension variable at
  `tools.py:152` and `:287` (scoped, so benign).
- **Inconsistent return type.** `pycalphad_plot_phase_diagram` returns an `Image` on
  success but a bare `str` on failure — every other tool returns a JSON string. A caller
  parsing JSON gets a surprise on the error path.
- **Mutable default arguments.** `conditions: Dict[str, Any] = {}` on four tools. Not
  mutated, so currently benign.
- **No cost bounds.** No timeout or grid-size cap on `pycalphad_equilibrium`,
  `pycalphad_calculate`, or `simulate_solidification`. A wide grid or fine `step` can run
  for a very long time with no signal to the caller — relevant because the consumer is an
  autonomous agent, not a human who will notice.
- **Stale README.** Its Features list names 5 capabilities and omits the 3 AM tools.

## 10.9 Summary — where the seam stands

| Capability | In the fork | Exposed via MCP |
|---|---|---|
| equilibrium / calculate / phase diagrams | ✓ | ✓ |
| Scheil + equilibrium solidification | ✓ | ✓ |
| Hot-cracking indices | ✓ CSC, Kou, RDG, freezing range, TFR | ✗ partial — CSC + Kou only |
| Rosenthal point source | ✓ | ✓ |
| Eagar-Tsai distributed source | ✓ | ✗ imported, unexposed |
| 2D transient FD thermal solver | ✓ | ✗ imported, unexposed |
| Mobility / interdiffusivity | ✓ | ✗ |
| Diffusion couple (1D, multicomponent) | ✓ | ✗ |
| Precipitation kinetics (KWN) + strength | ✓ | ✗ |
| Curated `tdb-forge` databases | ✓ | ✗ loadable, not discoverable |

Roughly a third of the local layer reaches the design loop, and the databases the pipeline
exists to produce are not among what it can find.
