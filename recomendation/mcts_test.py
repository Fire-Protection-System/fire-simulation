import random
import json

from simulation.sectors.fire_state import FireState
from simulation.forest_map import ForestMap
from mcts_search import mcts_search
from fire_simulation_node import FireSimulationNode

def load_forest_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def print_sector_states(sectors, title="Sector States"):
    print(f"\n{title}:")
    for s in sectors:
        print(f"Sector {s.sector_id} | Fire: {s.fire_level:.1f} | Burn: {s.burn_level:.1f} | "
              f"Extinguish: {getattr(s, '_number_of_fire_brigades', 0)} | State: {s.fire_state.name}")

def run_mcts_test():
    configuration = load_forest_config("forest_4x4_conf_20250415_173345.json")
   
    forest_map = ForestMap.from_conf(configuration)
    
    forest_map.start_new_fire()
    forest_map.start_new_fire()
    forest_map.start_new_fire()

    sectors = [sector for row in forest_map.sectors for sector in row]

    initial_node = FireSimulationNode(sectors=sectors, time_step=0, max_steps=2, map=forest_map, max_brigades = 5)
    print_sector_states(sectors, "Sector States Before Action")

    no_action_sectors = initial_node.simulate_steps_without_action(initial_node.max_steps);
    print_sector_states(no_action_sectors, "Sector States Without Action")

    action = mcts_search(initial_node, time_limit=5)

    if action:
        print("\nMCTS recommends:")
        for sector_id, brigades in action:
            print(f"  - Send {brigades} fire brigade(s) to sector {sector_id}")
    else:
        print("\nMCTS did not find a viable action.")
        return

    for sector in sectors:
        for sector_id, brigades in action:
            if sector.sector_id == sector_id:
                sector._number_of_fire_brigades = brigades

    for sector in sectors:
        sector.update_sector()

    print_sector_states(sectors, "Sector States After Action")


if __name__ == "__main__":
    run_mcts_test()
