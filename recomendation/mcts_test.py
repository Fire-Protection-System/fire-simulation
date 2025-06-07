import json
import random
import numpy as np
import copy
import logging
import sys

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

from reward_strategies import *

logger = logging.getLogger(__name__)
logger.disabled = True
logger.setLevel(logging.CRITICAL)

CONFIG_PATH = "./configs/forest_4x4_conf_20250607_164147.json"

NUMBER_OF_SIMULATED_FIRES = 4
NUMBER_OF_MAX_SEARCH_STEPS = 5
MAX_SIMULATION_TIME = 3

NO_EPOCHS = 10
NO_TESTS = 5

random.seed(42)
np.random.seed(42)

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
    action, best_score = mcts_search(root, time_limit=MAX_SIMULATION_TIME, return_score=True)

    print("\n=== BEST SCORE ====\n", best_score, "\n")

    if not action:
        print("\nMCTS did not find a viable action.")
        return

    print("\nMCTS recommends dispatching:")
    for agent_idx, sector_id in action:
        print(f"  • Agent #{agent_idx} → Sector {sector_id}")

    with_action_sectors = root.simulate_steps_with_action(NUMBER_OF_MAX_SEARCH_STEPS, action)
    print_sector_states(with_action_sectors, "After recomendations")
    print("")


def data_collector():    
    result_agg = {}
    
    for test_id in range(1, NO_TESTS):
        MAX_SIMULATION_TIME = test_id  
        result_agg[test_id] = {}
        
        print(f"Running test {test_id + 1}/{NO_TESTS} (time limit: {MAX_SIMULATION_TIME})")
        
        for strategy_name, strategy_config in REWARD_CONFIGS.items():
            print(f"  Testing strategy: {strategy_name}")
            
            strategy_results = {
                "avg_burned_sectors": 0,
                "avg_fire_level": 0,
                "strategy_name": strategy_name,
                "time_limit": MAX_SIMULATION_TIME,
                "epochs_completed": 0
            }
            
            for epoch in range(NO_EPOCHS):
                try:
                    forest_map = ForestMap.from_conf(load_forest_config(CONFIG_PATH))
                    
                    for _ in range(NUMBER_OF_SIMULATED_FIRES):
                        forest_map.start_new_fire()
                    
                    sectors = [s.clone() for row in forest_map.sectors for s in row]
                    agents = [s.clone() for s in forest_map.fireBrigades]
                    
                    root = FireSimulationNode(
                        sectors=[s.clone() for s in sectors],
                        agents=[a.clone() for a in agents],
                        time_step=0,
                        max_steps=NUMBER_OF_MAX_SEARCH_STEPS,
                        forest_map=forest_map.clone(),
                        max_brigades=len(agents),
                        reward_strategy=strategy_config
                    )
                    
                    no_action_sectors = root.simulate_steps_without_action(root.max_steps)
                    
                    action, best_score = mcts_search(
                        root, 
                        time_limit=MAX_SIMULATION_TIME, 
                        return_score=True
                    )
                    
                    epoch_burned_sectors = 0
                    epoch_fire_level = 0
                    simulation_runs = 100
                    
                    for run in range(simulation_runs):
                        with_action_sectors = root.simulate_steps_with_action(
                            NUMBER_OF_MAX_SEARCH_STEPS, 
                            action
                        )
                        
                        epoch_burned_sectors += sum(sector.burn_level for sector in with_action_sectors)
                        epoch_fire_level += sum(sector.fire_level for sector in with_action_sectors)
                    
                    strategy_results["avg_burned_sectors"] += (epoch_burned_sectors / simulation_runs) / NO_EPOCHS
                    strategy_results["avg_fire_level"] += (epoch_fire_level / simulation_runs) / NO_EPOCHS
                    strategy_results["epochs_completed"] += 1
                    
                except Exception as e:
                    print(f"    Error in epoch {epoch}: {e}")
                    continue
            
            result_agg[test_id][strategy_name] = strategy_results
    return result_agg


def print_results_summary(result_agg):
    """
    Print a clear summary of the collected results.
    """
    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    print("="*60)
    
    for test_id, strategies in result_agg.items():
        print(f"\nTest {test_id + 1} (Time Limit: {test_id}):")
        print("-" * 40)
        
        for strategy_name, results in strategies.items():
            print(f"  Strategy: {strategy_name}")
            print(f"    Avg Burned Sectors: {results['avg_burned_sectors']:.3f}")
            print(f"    Avg Fire Level: {results['avg_fire_level']:.3f}")
            print(f"    Epochs Completed: {results['epochs_completed']}/{NO_EPOCHS}")
            print()

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    logging.root.setLevel(logging.CRITICAL)

    simulation_logger = logging.getLogger("simulation")
    simulation_logger.setLevel(logging.CRITICAL)
    simulation_logger.propagate = False
    simulation_logger.disabled = True 

    results = data_collector()
    print_results_summary(results)