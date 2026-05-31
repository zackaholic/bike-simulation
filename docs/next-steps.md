# Next Steps — Project State and Continuation Plan

Written 2026-05-29 after completing Phases 0-8 + versioning + query expansion.

## Where the project stands

**What's built**: a complete three-tier world simulation (geology → climate-hydrology → ecology) with speciation, distinguished individuals, fire/blowdown disturbance, an orchestrator, CLI, debug visualizer, full version history, and a rich query interface. 172 tests, ~4,000 lines of source, all passing.

**What's ready to build next**: the webview observability tool. Both prerequisites are complete:
1. Version history system (copy-on-write rasters, lifecycle tracking, atomic saves)
2. Expanded query interface (7 version-aware methods the webview will consume)

**What the simulation produces today** (seed 42, 125 simulated years):
- 15 species (6 ancestors → 9 derived through geographic fragmentation)
- 753 distinguished individuals
- 35 fires + 9 blowdowns creating disturbance history
- Species distributions respond to terrain, climate, and competition
- Everything reproducible from seed

## Two parallel tracks ahead

### Track 1: Webview Observability Tool

**Why**: PNG snapshots can't answer "what's at this point?", "how did this region change?", or "what's the history of this individual?" The webview is a browser-based interactive map for inspecting world state — the tool that makes simulation refinement possible.

**Architecture**: lives in `extract/webview/`, consumes Layer B (WorldQuery) only, never Layer A directly. Flask backend, vendored Leaflet.js frontend, single-file HTML, no build tooling.

**Build sequence** (from the plan Zack developed with the architecting agent):

1. **Phase 1 — Backend skeleton**: Flask app, `/api/versions` and `/api/world/<version>/metadata` endpoints. Verify with curl.

2. **Phase 2 — Tile rendering**: `render_tile()` turning numpy arrays into PNG tiles via Pillow (not matplotlib — too slow for tiles). Filesystem cache keyed by `(version, layer, zoom, x, y)`. Zoom levels 0-4 for the 50km world.

3. **Phase 3 — Bare Leaflet map**: `L.CRS.Simple` map showing heightmap tiles. Pan and zoom. This is already more useful than PNGs.

4. **Phase 4 — Layer toggle**: base layers (heightmap, bedrock), overlays (moisture, temperature), species density dropdown. Layer switching swaps tile URLs.

5. **Phase 5 — Point inspection**: click anywhere → sidebar showing geology, climate, hydrology, vegetation, individuals, events at that point. Uses `query_point()`.

6. **Phase 6 — Individual markers**: distinguished individuals as Leaflet markers, sized by age/prominence, colored by species. Click → sidebar with full history. Uses `query_individuals_in_bbox()` and `get_individual_detail()`.

7. **Phase 7 — Version picker**: dropdown to switch versions, URL state encoding for bookmarkable views.

**Dependencies**: Flask needs to be added to pyproject.toml. Leaflet.js files vendored into `extract/webview/static/leaflet/`.

**Key decisions to make during build**:
- Tile size (256x256 is standard Leaflet)
- Zoom level mapping (zoom 0 = whole world in 1 tile → zoom 4 ≈ 12m/pixel)
- Colormap choices per layer (reuse from debug visualizer where possible)
- How to handle species density layers (there could be 15+ species; dropdown is the plan)

### Track 2: Simulation Refinement

**Why**: the simulation works but is simplified in ways that matter for the cyclist's experience. The gap analysis (`docs/gap-analysis.md`) catalogs everything; the priority gaps (ranked by impact on the rider) are:

1. **Distinguished individual lifecycle** — Currently individuals accumulate forever with no death, no aging, no snag→log→mound decay. These are the emotional anchor points; the cyclist needs to see a familiar tree die and slowly become a mossy log over rides. Implementation: track age per tick, kill individuals when density drops to zero at their location, add decay_state to EventStore (alive → snag → log → mound), visual prominence decreases over decades.

2. **Fire → seed bank germination** — Fire kills density but doesn't trigger the seed bank. The design's signature moment — "burned hillside comes back as something suppressed for a century" — doesn't happen yet. Implementation: after fire kills density in burned cells, boost establishment rate from seed bank in those cells for the next few ticks.

3. **Upward event consumption** — Fire events are recorded but climate-hydrology never reads them. Post-fire erosion pulses are a real phenomenon (no root structure → accelerated erosion). Implementation: climate-hydrology reads fire events at its next tick, applies an erosion multiplier to recently burned cells.

4. **Deeper erosion** — 5 passes of grid-based erosion produces subtle smoothing. The terrain needs visible water-carved valleys. Options: more passes (simple but slow), particle-based erosion (better results), or numba-accelerated grid erosion (same algorithm, 10-100x faster).

5. **Speciation driven by conditions** — Currently trait drift is a random walk. The design says drift should be *directional toward local conditions* (an isolated alpine population should become more cold-tolerant). Implementation: compute mean suitability in the fragment, bias trait drift toward values that would increase suitability there.

6. **Richer trait model** — 7 traits vs the design's 15-25. Adding categorical traits (woody/herbaceous, evergreen/deciduous) and linked tradeoffs would make species feel more distinct. This changes the genome structure and suitability calculation.

7. **Climate drift** — Without it, the world's climate is frozen and species distributions can only change through competition and disturbance, not through shifting baselines. Implementation: slow random walk on base temperature and precipitation parameters, making climate-hydrology's output change between ticks even without geology changes.

**Recommended order**: items 1-3 are high impact and low risk (they extend existing systems without architectural changes). Items 4-7 are larger scope. Build the webview first (Track 1), then use it to evaluate refinements (Track 2) — you'll see immediately whether a change produces the desired effect.

## Technical notes for continuation

### Test suite timing
The full test suite takes ~10 minutes. The slow tests are the multi-tick ecology and orchestrator tests (20-25 ecology ticks each). For fast iteration, skip them:
```
uv run pytest --ignore=tests/test_orchestrator.py --ignore=tests/test_ecology_advanced.py
```
This runs 142 tests in ~50 seconds. Run the full suite before committing.

### Agent workflow
Zack prefers heavy agent use with separation between test writing, implementation, and test running. This preserves main conversation context for architectural decisions. See memory file `feedback_agent_workflow.md`.

### Zack's learning goals
This is a learning/research project. Explain what we're building at each step and how it connects to the architecture. Don't assume he holds all design doc knowledge in his head. See memory file `user_learning_goals.md`.

### Key files
- `CLAUDE.md` — project instructions, architectural commitments, current status
- `docs/architecture.md` — three-tier design, data layers, reproducibility
- `docs/simulation-design.md` — what each tier does in detail
- `docs/gap-analysis.md` — design spec vs implementation comparison
- `docs/decisions.md` — decision log with rationale
- `docs/roadmap.md` — original phase-by-phase build plan

### Commit history
```
005cb36 Phase 0: repo skeleton, World data model, and seeded RNG infrastructure
54993fd Phase 1: canonical state I/O — RasterStore, EventStore, World directory
224eeeb Phase 2: query interface — WorldQuery as Layer B abstraction
a34a7b8 Phase 3: debug visualizer — top-down PNG renderer with synthetic data
8a46313 Phase 4: geology stub — noise heightmap, Voronoi bedrock, soil derivation
77596f0 Phase 5: climate-hydrology tier — climate, erosion, rivers, derived cache
8ae3042 Phase 6: basic ecology — species, suitability, competition, dispersal, seed bank
8b3984b Phase 7: advanced ecology — individuals, speciation, disturbance
97b8d0e Phase 8: orchestrator and CLI — scheduling tiers, ride advancement
6f2aa16 Update docs: current status through Phase 8, decision log
17887fc Add gap analysis: design spec vs implementation through Phase 8
031fda1 Add versioning system: version history, lifecycle tracking, atomic saves
2c85cbf Expand query interface with version-aware methods for webview
```

### Dependencies
```
numpy>=2.0, zarr>=3.0, matplotlib>=3.8
dev: pytest>=8.0, ruff>=0.8
not yet added: flask (needed for webview)
```

### Performance profile
- Geology tick: ~1s
- Climate-hydrology tick: ~5-10s (flow accumulation is the bottleneck — Python loop over 1M cells)
- Ecology tick: ~2-5s (increases with species count; 15 species × 1M cells)
- 25 ecology ticks: ~2-3 minutes
- Flow accumulation + distance transform: candidates for numba acceleration
- Fire spread CA: pure Python BFS, could be vectorized
