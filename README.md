# Bike Simulation

A world simulation traversed by pedaling. The simulation generates a coherent, evolving world overnight; in the morning, the rider mounts a bike trainer with a tachometer and pedals through a fresh 25-30 km route at cycling pace. The world is meant to reward slow, sustained attention: the kind of place where "why is that boulder there?" is a meaningful question with an answer rooted in process.

## Emotional core

Exploration, discovery, and novelty — but specifically the *meditative* kind. The world should feel like the current frame of a long history, not a procedurally placed scene. Subtle changes between rides (a mother tree finally falling, a fire scar from last week's "lightning," a new species spreading into a niche it just adapted to) should matter more than dramatic ones. The rider should become a naturalist of a world that genuinely existed before they arrived.

Inspirations: Dwarf Fortress (deep procedural history producing emergent meaning), Proteus (sensory and emotional coherence over systematic justification), Caves of Qud (subverting cause-and-effect in service of mythic feel).

## Core architectural commitments

**The world is simulated, not generated.** Generation and overnight advancement are the same operation at different scales. A new world is spun up by simulating 200M+ years of history and sampling the current frame.

**Three simulation tiers with decoupled timescales.**
1. **Geology**: bedrock, faults, soil parent material. Ticks every 10K-100K years. Largely static during normal play.
2. **Climate-hydrology**: climate envelope, rivers (as a graph), watersheds, erosion, soil moisture. Ticks every 10-100 years. Contains a slow climate envelope and fast weather realizations sampled on demand.
3. **Ecology**: species populations, distinguished individuals, disturbances. Ticks every season. Where the cyclist mostly lives.

**Tiers communicate via events, not direct calls.** Slow tiers emit state changes downward; fast tiers emit events (fire happened here, beaver dam placed) upward, batched and applied at the slow tier's next tick.

**Rendering is fully decoupled from simulation.** The simulator produces canonical state. A query interface answers spatial/temporal questions over that state. Renderer-specific extractors derive output formats (e.g., Godot chunks-and-manifest) from the query interface. The renderer never reads simulation state directly.

**The world is reproducible from seed.** Deterministic RNG, seeded per-tier per-tick. Replay from seed reproduces bit-identical state. This is for debugging and operational comfort, not a runtime feature.

**The rider is an observer, not an actor.** No back-channel from rendering to simulation. The only feedback loop is: ride logs trigger overnight advancement. Heartrate and pedaling cadence affect render-time visual effects only; they do not modify world state.

## What's special about this project

The overnight compute budget is effectively unlimited (mini PC sitting idle for 22 hours a day). This permits *physically motivated* simulation passes — real hydraulic erosion, real ecological succession, real evolutionary radiation of species — at scales most games can't afford. The architectural goal is to spend that compute on processes that leave *visible traces* a cyclist would notice.
