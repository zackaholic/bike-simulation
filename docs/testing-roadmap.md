# Testing roadmap — world-response tuning (resume here)

A reminder of the experiments we lined up. The goal is to tune **how the world
responds to a local shock** (fire is the probe) so that riding the same world
over a long time **accumulates legible history**.

**Target feel:** *meditative* — static from one ride to the next, but shifting
patterns become noticeable over ~a couple weeks of riding. Rewards careful
observation. So: calm climate-driven background **+** disturbances that leave
marks worth noticing.

## New tools to use

- **Ride-lens webview**: path overlay + ±50 m strip, version-aware ride profile
  (species density vs distance), hover→map marker, click→inspection sidebar.
  `uv run python -m bike_sim.extract.webview <world_dir>`
- **Purpose-built runs**: `--snapshot-interval` / `--summary-interval` on
  `create`/`advance` (ticks; 4 ticks = 1 yr). A "month of riding" ≈
  `advance 160 --snapshot-interval 32` → 20 snapshots @ 8 yr.
- **v2 baseline**: `worlds/baseline_seed42_v2` (10/14 species persist, ride path
  baked). Consider swapping it in for `baseline_seed42`.

## The four levers (detail in docs/decisions.md, R2026-06-21)

- **A — weather period** (`base_freq`, ~300 yr now) vs ~100 yr ecology healing →
  background drift tempo. Note: A and the ride-time mapping trade off — only the
  *ratio* (perceived change per ride) matters.
- **B — Allee threshold** (`allee_theta`, implemented, off by default) → scar
  persistence / spatial recolonization fronts.
- **C — recovery speed** (growth/dispersal scalars) → how long scars stay visible.
- **D — incumbency/priority** (**not built yet**) → path-dependence, lasting
  idiosyncratic patches. B is a prerequisite for D to produce *lasting* effects.

## Testing checklist

- [ ] **Weather-period decision**: run a month-of-riding sim at the current
      ~300 yr setting; eyeball how much biome migration shows over the window.
      Decide *subtle* (keep ~300 yr) vs *swingy* (shorten toward ~150 yr so the
      ecology lags the climate). One-line `base_freq` change.
- [ ] **Allee (B) sweep**: try `allee_theta` ∈ {0, 1, 3, 8}. Fire a scar; watch
      refill via ride-scale snapshots — does it recolonize from edges? do some
      scars persist?
- [ ] **Build incumbency (D)** (asymmetric competition toggle), then test
      path-dependence: does the same fire heal into *different* communities?
- [ ] **Fire matrix**: identical center burn, **climate frozen**, across
      {baseline, +B, +C, +D, B+D}. Measure scar persistence (yrs to re-match
      control), recolonization-front spread, community divergence.
- [ ] **Methodology (important)**: PAIRED fire-vs-control, **differenced over the
      burn mask** — global/transect metrics can't see a local fire. Observe with
      the grid-view flipbook + ride profile at ride-scale snapshots.
- [ ] **Watch the 4 excluded species** in v2 (valley_hardwood, meadow_herb,
      pioneer_forb, dry_grass): they established then got competitively
      outcompeted — do B/D let them persist via refugia / priority?
- [ ] **Rider-experience validation**: short runs with frequent snapshots;
      confirm "static ride-to-ride, shifting over weeks" actually holds at the
      chosen tuning.

## Build TODO before some of the above

- [ ] **D — incumbency** mechanism (asymmetric Lotka-Volterra) as a toggle.
- [ ] **Fire-matrix harness**: paired-diff runner over the burn mask +
      ride-scale snapshot capture (extend `scripts/test_ecology.py`).

## Outcome wanted

An empirical **"tuning → kind of change"** table. No single config is expected
to be "right" — the point is to understand the option space before committing,
in preparation for the rendering phase.
