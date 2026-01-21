import logging
import pytest

from src.engine.models.map.sector import Sector
from src.engine.models.map.sector_type import SectorType
from src.engine.models.map.sector_state import SectorState
from src.engine.models.core.location import Location
from src.engine.models.agents.fire_brigade import FireBrigade
from src.engine.models.agents.agent_state import AGENT_STATE


class MapStub:
    def __init__(self, sector):
        self._sector = sector

    def find_sector(self, location: Location):
        # Always return the provided sector for testing
        return self._sector


def test_fire_brigade_sets_executing_sector_when_missing(caplog):
    caplog.set_level(logging.DEBUG)

    sector = Sector(
        sector_id=10,
        row=0,
        column=0,
        sector_type=SectorType.FOREST,
        initial_state=SectorState.NORMAL,
        fire_level=10.0,
    )

    base_loc = Location(0.0, 0.0)
    init_loc = Location(0.0, 0.0)

    fb = FireBrigade("fb1", None, base_loc, init_loc, initial_state=AGENT_STATE.EXTINGUISHING)

    assert fb._state.value == "executing"
    assert fb._executing_sector is None

    map_stub = MapStub(sector)

    res = fb.update_physics(delta=1.0, map_ref=map_stub)

    # After update_physics, the executing sector should be set
    assert fb._executing_sector is sector
    # And no warning about missing executing_sector should be present
    assert not any("No stored executing_sector" in record.getMessage() and record.levelno >= logging.WARNING for record in caplog.records)

    # Simulate executing tick to finish part of task
    fb.execute_task(1.0, sector)
    # Clean up
    fb._executing_sector = None
