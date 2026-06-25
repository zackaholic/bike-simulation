# Speciation — Design Exploration (in progress)

> **Status: EXPLORATORY. Nothing here is decided.** This is a working capture of an
> ongoing design conversation (Zack + Claude, 2026-06-24). It deliberately does *not*
> live in `decisions.md` yet — implementation and testing are slated for a later session.
> The point of this doc is to preserve where our thinking *converges*, where it *diverges*,
> and which tensions are still open, so we can resume without re-deriving.

---

## 1. What this mechanic is *for*

- **Revive extinction.** There is currently no extinction mechanic running. The old
  speciation/extinction/reabsorption code (`ecology.py:1058-1501`) is considered a dead
  end (see §2). Speciation is the chosen vehicle for re-introducing death to the world.
- **Maintain a stable *floor* of well-adapted species while allowing *churn*.** Not a
  forever-growing tree of life; not a static species set. A breathing community whose
  *composition* turns over while its *diversity* stays bounded.
- **Surface deep history to the rider ambiently.** The rider should perceive speciation
  *only* through morphological change in what they ride past — never through an event log
  or announcement. (This makes the phenotype genome load-bearing; see §6.)

---

## 2. Lessons from the failed old model

The previous speciation system (`_check_speciation`, `_check_extinction`,
`_check_reabsorption`) was built on an older ecology model and **never balanced** — it
produced either runaway diversity or zero events, with no stable middle. Diagnosis:

- **No working negative feedback.** Its governor was a *global* count cap
  (`saturation_factor = 1 − alive_count/100`, `ecology.py:1210`). A global knob can't
  respond to *local* crowding, so any single tuning sits on one side of the
  runaway/zero knife-edge.
- **Trigger was geographic fragmentation + barrier detection** (`ecology.py:1283-1290`),
  which was both expensive and brittle — it gated on so many conditions (300yr cooldown,
  ≥2 components, fragment size, hospitable-barrier rejection, divergence ≥0.15) that in
  practice it likely almost never fired.
- **Daughters could be created too similar to survive.** `min_genome_divergence = 0.15`
  (`ecology.py:1213`) is *below* the reabsorption threshold of 0.25 (`ecology.py:1376`)
  — so a fresh daughter could be immediately re-merged or competitively excluded. (See §5
  for why 0.25 is the magic number.)

**Takeaway:** the hard part of speciation-as-a-mechanic is **regulation**, not the
trigger or the seeding. Any working version needs a control loop that makes the species
count a stable attractor.

> **The old code has been REMOVED (2026-06-24), NOT used as a foundation.**
> `_check_speciation` / `_check_extinction` / `_check_reabsorption` (and their private
> helpers `_bfs_label` / `_connected_components_coarse` /
> `_fragments_connect_through_hospitable`, plus the `enable_speciation` /
> `enable_extinction` toggles) were designed around an ecology model that is no longer in
> use; they never balanced (runaway-or-zero) and never fired in practice. **446 lines
> deleted from `ecology.py`.** We cite them only for *lessons* (above). The mechanic below
> is a **clean-slate replacement** with **zero inherited constraint**. **The only thing that
> genuinely constrains this design is the *current* ecology model** — the per-tick
> growth/competition/mortality rule, the α competition formula, suitability normalization,
> the refugium floor, the 6-trait genome, and the data model.
>
> *Deliberately kept* (not part of the old speciation layer): distinguished-individual
> promotion (`_promote_individuals` / `enable_individuals`), and the full lineage data model
> — the `EventStore` species table (`parent_id` / `appeared_year` / `extinct_year`), its
> API (`add_species` / `mark_species_extinct` / `update_species_genome`), and `WorldQuery`'s
> timeline/lineage methods. These are live infrastructure the **new** mechanic will populate
> (extinction via `mark_species_extinct`, in-place anagenesis via `update_species_genome`).

---

## 3. Where we currently CONVERGE

The shape we both like:

1. **Trigger = bottleneck → recovery, not steady decline.** A species is pushed *low*
   during an unfavorable climate phase; the speciation happens on the *upswing*, as it
   re-expands into reopening niche space. This is post-bottleneck radiation (cf.
   post-glacial recolonization from refugia). It puts the speciation pulse exactly where
   there's opportunity, and avoids the "metronome" failure of triggering during the trough.

2. **Death is EMERGENT, not a scripted timer.** The parent dies because the better-adapted
   child outcompetes it in the now-current climate — not because a 5-year clock expired.
   This keeps death inside the sim's existing "death comes from suitability + competition"
   core, and leaves room for the occasional *climate rescue* of a parent if the cycle
   turns back before it's gone (rare, surprising — desirable).

3. **The governor: death is conditional on having birthed.** A species only becomes
   extinction-eligible *after* it has speciated. Therefore **you can never lose a species
   without having first replaced it** → the floor is structurally protected. Quiet climate
   → no bottlenecks → no speciation → no death → species persist. Cycling climate → churn.
   *This is the negative feedback the old model lacked.*

4. **Divergence magnitude is earned from the climate shift across the trough.** Parent is
   adapted to the *pre-trough* optimum; child is adapted to the *new/current* optimum. The
   size of the climate swing sets how distinct the daughter is. This ties the mechanic
   directly to the climate cycle and makes the parent's emergent death *principled* (the
   child really is fitter now).

5. **Booms and busts are emergent and correlated.** A large climate swing bottlenecks many
   species at once → many recover-and-speciate together → a **speciation boom**. The world
   is then "half full" of post-reproductive parents, all now outcompeted → a **correlated
   mass-extinction echo** as the climate settles. Small swings → small churn. The species
   count oscillates *around the floor*; the floor is the attractor. (Zack's framing,
   2026-06-24 — this is the hoped-for signature dynamic.)

6. **No speciation events.** Speciation is conveyed *ambiently* through morphology. The
   rider reads "this stand is subtly woodier / shorter-leaved than the valley behind" —
   never a notification.

7. **Phenotype / morphological genome must be re-introduced** to carry #6. (Reverses the
   refactor's deferral; see §6.)

---

## 4. The CENTRAL unresolved tension — genetic competition vs. geographic isolation

This is the hardest open problem and the one most likely to sink the mechanic.

**The bind:**
- Competition is genome-distance-based: `alpha = 1.0` for identical genomes, decaying to
  `0.4` as they diverge (`_competition_alpha`, `ecology.py:873-890`).
- Parent and child are, by construction, *similar* (child = parent + one climate-swing's
  divergence). So they compete at **near-maximal α** — the *least* coexistable pair.
- For the transient "two distinct species" coexistence window to exist at all, they need
  **separation** — otherwise whichever has marginally higher local `K·suit` excludes the
  other immediately, and the split is cosmetic.
- The current model has **no way to enforce or maintain geographic isolation.** Dispersal
  runs every tick and remixes ranges. The old model's barrier-detection approach to this
  was a disaster.

Zack's words: *"The genetic competition is the hardest to address. Geographic isolation
seems somewhat necessary but difficult to enforce."*

### Candidate resolution (Claude's proposal — NOT decided): refugia *are* the isolation

Instead of *enforcing* isolation, *derive* it from where the species survived the trough:

- During an unfavorable phase, a species survives in one or more **disjoint refugial
  pockets** (this is what the refugium floor mechanic is for — currently `0.0`/off,
  `ecology.py:402-406`, seeds a *single* best cell today).
- Pockets are geographically separated **by definition** — no enforcement needed.
- On recovery, each pocket re-expands; distinct pockets that drifted toward *different*
  local optima become distinct daughters; they make **secondary contact** somewhere in the
  middle. Whether they then coexist or one excludes the other depends on how much they
  diverged vs. the α/distance threshold (§5).

This makes **the refugium floor and speciation the same mechanism at different climate
phases.** Open sub-questions it raises:
- The refugium floor would need to support **multiple pockets**, not one cell.
- Do the pockets drift *independently during isolation* (requires per-pocket genome state —
  heavy; a species currently has *one* genome), or do we cheat and set daughter divergence
  as a function of **trough duration/depth** at the moment of recovery (cheap, captures the
  feel, less physically pure)? Leaning cheap-first.

---

## 5. The quantitative knife-edge (testable)

The competition formula gives a sharp, checkable constraint. With `BASELINE = 0.4`,
`niche_width = 0.3`:

```
alpha(d) = 0.4 + 0.6 * exp(-(d / 0.3)^2)
```

- `alpha(0.25) ≈ 0.70`, `alpha(0.30) ≈ 0.63`. So a daughter needs genome distance
  **d ≳ 0.25–0.30** to get α meaningfully below 1.0 — i.e. enough that it isn't simply
  competitively excluded by (or excluding) the parent in shared cells. (This threshold comes
  purely from the *current* competition formula. The old reabsorption constant happened to
  also be 0.25, but that's irrelevant — there is no reabsorption system here unless we
  choose to build one; see §7.)
- Climate-driven divergence lives mostly in 2 of the 6 traits (`drought_tolerance`,
  `frost_tolerance` — the ones feeding suitability). To reach total distance ~0.25 from two
  traits, each optimum must shift ~0.18–0.21. That's a **substantial but plausible** climate
  swing.

**Emergent consequence (possibly the nicest property):** the α/distance threshold acts as
an *automatic mode switch*:
- **Small climate swing** → small optimum shift → daughter distance < 0.25 → daughter is
  reabsorbed / excluded → no viable split → the lineage just **re-adapts in place
  (anagenesis)**. This may *be* the "small churn" Zack wants.
- **Large climate swing** → distance ≥ 0.25 → viable split → coexistence + emergent parent
  death → **cladogenesis + boom/bust**.

If this holds, the mechanic self-selects between in-place adaptation and true branching
based on swing magnitude — no extra knob. **Must be verified numerically**, because if a
*typical* swing produces less than ~0.25 distance, the mechanic degenerates to pure
anagenesis and never branches.

---

## 6. Phenotype / morphological genome (required, to be designed)

- Today the genome is **6 functional traits only** (`_CORE_TRAITS`, `ecology.py:53-56`):
  `drought_tolerance, frost_tolerance, growth_rate, max_height, lifespan, dispersal_range`.
  A richer two-category genome (10 functional + 7 morphological: `stem_woodiness`,
  `leaf_size`, `bark_texture`, `flower_color`, …) was *designed* in `decisions.md:347-378`
  but **deliberately deferred/stripped** in the ecology refactor.
- This mechanic **requires re-introducing the morphological genome** — it is the *only*
  channel through which the rider perceives speciation (§3.6). Not cosmetic; load-bearing.
- Design constraints to honor when we get to it:
  - Morphological traits are **zero simulation cost** — they ride along in `genome_json`
    and are never read by the tick math (preserves the refactor's performance win).
  - Morphology must **drift together with** functional divergence, so the *visible* change
    tracks the *ecological* one (a child adapted to a drier optimum should also *look*
    drier-adapted — waxier/smaller leaves, more woodiness — per the trait couplings).
  - Higher drift variance on morphology than on function, so sister species are *visibly*
    distinct even when functionally close.
- **Note:** the proposal's original trait→trait mappings need correction against the real
  model — e.g. `max_height` drives *shade competition*, not temperature; temperature is
  `frost_tolerance`. Mapping table to be redesigned alongside the genome.

---

## 7. Other open questions / where we still DIVERGE or haven't pinned

- **Per-event count accounting.** Is a speciation 1 parent → 1 child (+ parent dies later;
  net 0 churn) or 1 parent → 2 children (+ parent dies; net +1)? Resolved-ish by §3.5: over
  a *full cycle* the count mean-reverts to the floor regardless, via the boom→bust echo.
  But the *per-event* number still affects boom amplitude and compute spikes. Not locked.
- **Do we need a gene-flow / merge mechanic at all?** (Fresh design question — *not* a
  reconciliation with old code, which is disposable.) The current competition model may
  already do the job for free: two too-similar species compete at high α, so one simply
  excludes the other in shared cells — competition *itself* prevents redundant near-duplicates
  from coexisting, which is most of what merging was for. The one thing competition does
  *not* do that explicit merging did is **genome averaging** (blending two lineages' genomes
  into one via gene flow). Open question: do we want that blending for its own sake (e.g. to
  soften secondary contact into hybridization rather than exclusion), or is competitive
  exclusion the behavior we want? Default assumption for now: **no separate merge mechanic** —
  let competition handle it, and revisit only if secondary-contact dynamics feel wrong.
- **Climate rescue of parents.** Emergent death *allows* a parent to survive if the cycle
  turns back before it's outcompeted. Desirable (surprising) — but how often? Affects how
  "reliable" churn feels. Unquantified.
- **Multi-pocket refugium vs. cheap trough-duration divergence** (§4) — unresolved.
- **Where the "low → recovering" detection state lives.** Needs per-species temporal memory
  (a remembered trough + a rising-density signal) that replays bit-identically from seed.
  No schema slot exists today; RNG must stay on the `(tier, pass, tick)` substream consumed
  in `list_species` order (`rng.py:48-72`). Implementation detail, not a design blocker.

---

## 8. Compute / reproducibility notes (for the later session)

- The tick core is **O(species²)** (alpha matrix + per-cell load sum, `ecology.py:738-768`)
  plus O(species) full-grid passes and a Zarr read/write per species per tick. A speciation
  **boom transiently ~doubles** the species count → ~4× cost spikes during booms. Acceptable
  for offline advance; note it.
- The data model already supports dynamic species fully: append-only SQLite species table
  with `parent_id`/`appeared_year`/`extinct_year` (`event_store.py:384-390`), per-species
  Zarr layers created on demand, and all consumers discover species by regex-scanning
  `species_*_density`. **Nothing is hardcoded at 14.** No data-layer work needed to grow/shrink
  the species set.

---

## 9. Minimal prototype sketch (when we pick it up)

Smallest thing that tests whether the loop self-regulates — **no morphology yet**:

1. Detect **low → recovering** per species (trough memory + density rising).
2. On recovery, spawn daughter(s) with genome shifted toward the **current** local optimum;
   divergence magnitude ∝ climate shift across the trough.
3. **No explicit parent death** — let competition + mortality kill the outcompeted parent
   emergently. (Parent becomes extinction-eligible only after birthing, per §3.3.)
4. Run through `scripts/test_ecology.py`'s **equilibrium → perturb** loop with a climate
   swing, and measure: does species count hold a **band** and **breathe** with the swing
   (boom then bust, reverting to a floor), or does it run away / flatline?
5. Separately, **numerically check §5**: for representative climate swings, what daughter
   genome distance results, and is it ≥ 0.25? (Decides branch-vs-anagenesis behavior before
   we trust any run.)

---

## 10. Resume pointers

- Competition: `_competition_alpha` `ecology.py:873-890`; baseline `ecology.py:50`.
- Refugium floor: `_apply_refugium_floor` `ecology.py:798-821` (currently `floor = 0.0`).
- Old speciation/extinction/reabsorption code: **REMOVED 2026-06-24** (446 lines). See git
  history (`git log -p -- src/bike_sim/tiers/ecology.py`) if a lesson needs re-reading.
- Kept lineage API the new mechanic will use: `mark_species_extinct`,
  `update_species_genome`, `add_species` in `event_store.py`; lineage/timeline queries in
  `world_query.py:330-639`.
- Genome traits: `_CORE_TRAITS` `ecology.py:53-56`; deferred morphology `decisions.md:347-378`.
- Suitability + references: `_compute_suitability` `ecology.py:673-695`; refs `ecology.py:40-43`.
- Test harness: `scripts/test_ecology.py` (`equilibrium`, `perturb`).
