from datetime import datetime
from src.engine.models.agents.fire_brigade import FireBrigade
from src.engine.models.agents.forester_patrol import ForesterPatrol
from src.engine.models.core.location import Location
from src.engine.agent_manager.agent_manager import AgentManager


class FakeMap:
    def __init__(self, brigades, patrols):
        self.fire_brigades = brigades
        self.forester_patrols = patrols

    def find_sector(self, location):
        class S:
            sector_id = 0
        return S()


def test_agent_ids_are_integers_when_numeric_strings():
    # Create agents with numeric string IDs
    fb = FireBrigade(fire_brigade_id="0", timestamp=datetime.now(), base_location=Location(0, 0), initial_location=Location(0, 0))
    fp = ForesterPatrol(forester_patrol_id="1", timestamp=datetime.now(), base_location=Location(0, 0), initial_location=Location(0, 0))

    fake_map = FakeMap([fb], [fp])
    manager = AgentManager(fake_map, message_store=None, engine=None)

    states = manager.get_agent_states()
    assert isinstance(states, list) and len(states) == 2

    # find entries by type
    fb_state = next(s for s in states if s.get("type") == "fire_brigade")
    fp_state = next(s for s in states if s.get("type") == "forester")

    # IDs should be ints when numeric
    assert isinstance(fb_state.get("fireBrigadeId"), int), f"Expected int, got {type(fb_state.get('fireBrigadeId'))}"
    assert fb_state.get("fireBrigadeId") == 0

    assert isinstance(fp_state.get("foresterPatrolId"), int), f"Expected int, got {type(fp_state.get('foresterPatrolId'))}"
    assert fp_state.get("foresterPatrolId") == 1
