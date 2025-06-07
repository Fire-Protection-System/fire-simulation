import json
import random
import copy
import logging

from datetime import datetime
from typing import List, Tuple, Optional

from simulation.sectors.fire_state import FireState
from simulation.forest_map import ForestMap
from simulation.fire_brigades.fire_brigade import FireBrigade
from simulation.agent_state import AGENT_STATE
from simulation.sectors.sector import Sector
from simulation.fire_brigades.fire_brigade_state import FIREBRIGADE_STATE

from recomendation.mcts_search import mcts_search
from recomendation.fire_simulation_node import FireSimulationNode

logger = logging.getLogger(__name__)

CONFIG_PATH = "./configs/my-file.json"

NUMBER_OF_SIMULATED_FIRES = 0
NUMBER_OF_MAX_SEARCH_STEPS = 5
MAX_SIMULATION_TIME = 5

import traceback

def load_forest_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def print_sector_states(sectors: list[Sector], title: str = "Sector States"):
    print(f"\n=== {title} ===")
    for s in sorted(sectors, key=lambda s: s.sector_id):
        brigades = getattr(s, "_number_of_fire_brigades", 0)
        print(
            f"Sector {s.sector_id:2d} | "
            f"Fire: {s.fire_level:5.1f} | "
            f"Burn: {s.burn_level:5.1f} | "
            f"Brigades: {brigades:2d} | "
            f"Extinguished: {s.extinguish_level:6.1f} | "
            f"State: {s.fire_state.name}"
        )

def predict(forest_map: ForestMap) -> List[Tuple[int, int]]:
    try:
        """
        Run MCTS to predict optimal actions for fire brigades
        Returns list of (agent_id, sector_id) tuples
        """
        sectors = [s for row in forest_map.sectors for s in row]
        agents = [a.clone() for a in forest_map.fireBrigades]
        
        active_fires = sum(1 for s in sectors if s.fire_state == FireState.ACTIVE)
        logger.info(f"MCTS starting with {active_fires} active fires and {len(agents)} fire brigades")
        
        root = FireSimulationNode(
            sectors=[s.clone() for s in sectors],
            agents=agents,
            time_step=0,
            max_steps=NUMBER_OF_MAX_SEARCH_STEPS,
            forest_map=forest_map.clone(),
            max_brigades=len(forest_map.fireBrigades)
        )
        
        logger.info(f"Starting MCTS search with time limit {MAX_SIMULATION_TIME} seconds...")
        best_actions, best_score = mcts_search(root, time_limit=MAX_SIMULATION_TIME, return_score=True)
        logger.info(f"Found optimal actions: {best_actions}")
        
        if best_actions:
            for agent_id, sector_id in best_actions:
                sector = next((s for s in sectors if s.sector_id == sector_id), None)
                if sector:
                    fire_level = sector.fire_level
                    burn_level = sector.burn_level
                    logger.info(f"Agent {agent_id} → Sector {sector_id} (Fire: {fire_level}, Burn: {burn_level})")
        else:
            logger.warning("MCTS found no recommended actions")
    
    except Exception as e:
        logger.error(f"Error in MCTS prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
    return best_actions

def run_mcts_test():
    forest_map = ForestMap.from_conf(load_forest_config(CONFIG_PATH))

    for _ in range(NUMBER_OF_SIMULATED_FIRES):
        forest_map.start_new_fire()

    sectors = [s.clone() for row in forest_map.sectors for s in row]
    agents = [s.clone() for s in forest_map.fireBrigades]
    print_sector_states(sectors, "Before any action")
    
    root = FireSimulationNode(
        sectors=[s.clone() for s in sectors],
        agents=[a.clone() for a in agents],
        time_step=0,
        max_steps=NUMBER_OF_MAX_SEARCH_STEPS,
        forest_map=forest_map.clone(),
        max_brigades=len(agents)
    )

    no_action_sectors = root.simulate_steps_without_action(root.max_steps)
    print_sector_states(no_action_sectors, "After doing NOTHING")
    action = mcts_search(root, time_limit=MAX_SIMULATION_TIME)

    if not action:
        print("\nMCTS did not find a viable action.")
        return

    print("\nMCTS recommends dispatching:")
    for agent_idx, sector_id in action:
        print(f"  • Agent #{agent_idx} → Sector {sector_id}")

    with_action_sectors = root.simulate_steps_with_action(NUMBER_OF_MAX_SEARCH_STEPS, action)
    print_sector_states(with_action_sectors, "After recomendations")

    print("")
    
if __name__ == "__main__":
    run_mcts_test()
