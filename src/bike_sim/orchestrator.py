"""Orchestrator — bridge between ride logs and the three-tier simulation.

The Orchestrator coordinates geology, climate-hydrology, and ecology tiers
for a given time advance.  Instead of manually ticking each tier, callers
say "advance 50 years" (or "advance a 30-minute ride") and the orchestrator
figures out the right number of ticks at each tier's timescale.

Tick ordering follows the architecture: ecology ticks first (fastest),
climate-hydrology when enough ecology years have accumulated, and geology
only when climate-hydrology years cross the geology threshold (very rare
during normal play).
"""

from __future__ import annotations

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import EcologyTier
from bike_sim.tiers.geology import GeologyTier
from bike_sim.world import World


class Orchestrator:
    """Schedule ticks across all three simulation tiers."""

    RIDE_YEARS_PER_MINUTE: float = 1.0
    RIDE_MAX_YEARS: float = 50.0

    def __init__(self, world: World) -> None:
        self._world = world

    def create_world(self) -> None:
        """Initialize world by ticking geology and climate-hydrology once.

        Ecology is *not* ticked here — it starts from zero when the first
        ``advance()`` call runs.  Creates version 0 to snapshot the initial
        world state.
        """
        next_version = self._world.current_version + 1
        self._world.rasters.set_version(next_version)

        GeologyTier(self._world).tick()
        ClimateHydrologyTier(self._world).tick()

        self._world.commit_version(trigger="create_world")

    def advance(self, years: float) -> dict:
        """Advance simulation by the given number of years.

        Tick ordering per the architecture:
          1. Ecology ticks (multiple, at 5 years each)
          2. Climate-hydrology ticks (if threshold reached)
          3. Geology ticks (if threshold reached — very rare)

        Returns a summary dict with tick counts.
        """
        next_version = self._world.current_version + 1
        self._world.rasters.set_version(next_version)

        eco = EcologyTier(self._world)

        eco_ticks = int(years / EcologyTier.YEARS_PER_TICK)

        climate_ticks = 0
        geology_ticks = 0

        eco_clock = self._world.tier_clocks["ecology"]
        climate_clock = self._world.tier_clocks["climate_hydrology"]
        geology_clock = self._world.tier_clocks["geology"]

        for _ in range(eco_ticks):
            eco.tick()

            # Climate-hydrology should tick once per 1000 ecology-years.
            # The initial tick from create_world() doesn't count — it
            # bootstraps the climate state before ecology starts.
            expected_climate = int(eco_clock.simulated_year // ClimateHydrologyTier.YEARS_PER_TICK)
            # Ticks beyond the initial bootstrap tick.
            actual_extra_climate = climate_clock.tick_number - 1
            if expected_climate > actual_extra_climate:
                ClimateHydrologyTier(self._world).tick()
                climate_ticks += 1

                # Similarly for geology relative to climate years.
                expected_geology = int(climate_clock.simulated_year // GeologyTier.YEARS_PER_TICK)
                actual_extra_geology = geology_clock.tick_number - 1
                if expected_geology > actual_extra_geology:
                    GeologyTier(self._world).tick()
                    geology_ticks += 1

        self._world.commit_version(trigger=f"advance {years} years")

        return {
            "years_advanced": years,
            "ecology_ticks": eco_ticks,
            "climate_hydrology_ticks": climate_ticks,
            "geology_ticks": geology_ticks,
        }

    def advance_ride(self, ride_duration_minutes: float) -> dict:
        """Convert ride duration to sim years and advance.

        Uses ``RIDE_YEARS_PER_MINUTE`` (1.0) to convert, capped at
        ``RIDE_MAX_YEARS`` (50.0) so a long ride doesn't skip too much
        world history.
        """
        years = min(
            ride_duration_minutes * self.RIDE_YEARS_PER_MINUTE,
            self.RIDE_MAX_YEARS,
        )
        return self.advance(years)

    def introduce_fire(self, x: float, y: float) -> None:
        """Manually trigger a fire event at the given world coordinates."""
        current_year = self._world.tier_clocks["ecology"].simulated_year
        self._world.events.add_event(
            "fire",
            x,
            y,
            current_year,
            radius=500.0,
            data={"source": "manual"},
        )

    def status(self) -> dict:
        """Return current world status summary.

        Includes seed, tier clock positions, species count, and individual
        count (searched across the full world extent).
        """
        species = self._world.events.list_species()
        individuals = self._world.events.find_individuals_near(25000, 25000, 50000)
        return {
            "seed": self._world.seed,
            "simulated_year": self._world.tier_clocks["ecology"].simulated_year,
            "geology_tick": self._world.tier_clocks["geology"].tick_number,
            "climate_hydrology_tick": self._world.tier_clocks["climate_hydrology"].tick_number,
            "ecology_tick": self._world.tier_clocks["ecology"].tick_number,
            "species_count": len(species),
            "individual_count": len(individuals),
            "current_version": self._world.current_version,
            "total_versions": len(self._world.list_versions()),
        }
