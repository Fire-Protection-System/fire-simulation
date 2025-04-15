import copy
import random
from itertools import product
from typing import List, Tuple, Optional

from node import Node
from simulation.sectors.sector import Sector
from simulation.sectors.fire_state import FireState
from simulation.fire_spread.coef_generator import calculate_beta
from simulation.fire_spread.wind import Wind
from simulation.forest_map import ForestMap
from simulation.sectors.geographic_direction import GeographicDirection

class FireSimulationNode(Node):
    def __init__(
        self,
        sectors: List[Sector],
        time_step: int,
        max_steps: int = 10,
        action: Optional[List[Tuple[int, int]]] = None,
        max_brigades: int = 2,
        map: Optional[ForestMap] = None,
        wind: Optional[Wind] = None
    ):
        self.sectors = sectors
        self.time_step = time_step
        self.max_steps = max_steps
        self.action = action
        self.max_brigades = max_brigades
        self.map = map
        self.wind = wind or Wind()

    def find_children(self):
        active_sectors = [sector for sector in self.sectors if sector.fire_state == FireState.ACTIVE]
        children = set()

        if not active_sectors:
            return children

        ranges = [range(0, self.max_brigades + 1) for _ in active_sectors]

        for distribution in product(*ranges):
            if sum(distribution) == self.max_brigades and any(b > 0 for b in distribution):
                assignments = [
                    (sector.sector_id, brigades)
                    for sector, brigades in zip(active_sectors, distribution) if brigades > 0
                ]
                new_sectors, new_map = self._simulate_action(assignments)

                child = FireSimulationNode(
                    new_sectors,
                    self.time_step + 1,
                    self.max_steps,
                    action=assignments,
                    max_brigades=self.max_brigades,
                    map=new_map, 
                    wind=copy.deepcopy(self.wind)
                )

                children.add(child)

        return children

    def find_random_child(self):
        active = [s for s in self.sectors if s.fire_state == FireState.ACTIVE]
        if not active:
            return None

        distribution = [0] * len(active)
        for _ in range(self.max_brigades):
            distribution[random.randint(0, len(active) - 1)] += 1

        assignments = [
            (active[i].sector_id, distribution[i])
            for i in range(len(active)) if distribution[i] > 0
        ]

        new_sectors, new_map = self._simulate_action(assignments)

        return FireSimulationNode(
            new_sectors,
            self.time_step + 1,
            self.max_steps,
            action=assignments,
            max_brigades=self.max_brigades,
            map=new_map,
            wind=copy.deepcopy(self.wind)
        )

    def is_terminal(self):
        return self.time_step >= self.max_steps or all(s.fire_state == FireState.INACTIVE for s in self.sectors)

    def reward(self):
        total_burn = sum(s.burn_level for s in self.sectors)
        return -total_burn

    def __hash__(self):
        return hash((self.time_step, tuple((s.sector_id, s.fire_level, s.burn_level) for s in self.sectors)))

    def __eq__(self, other):
        return isinstance(other, FireSimulationNode) and self.__hash__() == other.__hash__()

    def _simulate_action(self, assignments: List[Tuple[int, int]]) -> Tuple[List[Sector], ForestMap]:
        new_sectors = [s.clone() for s in self.sectors]
        new_map = self.map.clone()
        new_map.update_sectors(new_sectors)

        for s in new_sectors:
            for sector_id, brigades in assignments:
                if s.sector_id == sector_id:
                    s._number_of_fire_brigades = brigades

        for s in new_sectors:
            s.update_sector()

        self.simulate_fire_spread(new_sectors, new_map)

        return new_sectors, new_map

    def simulate_fire_spread(self, sectors: List[Sector], map: ForestMap):
        sectors_on_fire = [s for s in sectors if s.fire_state == FireState.ACTIVE]
        for sector in sectors_on_fire:
            self.update_fire_spread(sector, sectors, map)

        self.wind.update_wind()

    def update_fire_spread(self, active_sector: Sector, all_sectors: List[Sector], map: ForestMap):
        neighbors = map.get_adjacent_sectors(active_sector)

        for neighbor, direction in neighbors:
            if neighbor.fire_state == FireState.INACTIVE:
                prob = calculate_beta(self.wind, neighbor.sector_type, direction)
                if random.random() < prob:
                    neighbor.start_fire()

    def get_adjacent_sectors(self, sector: Sector) -> List[Tuple[Sector, GeographicDirection]]:
        if not self.map:
            return []
        return self.map.get_adjacent_sectors(sector)

    def simulate_steps_without_action(self, steps: int) -> List[Sector]:
        new_sectors = [s.clone() for s in self.sectors]
        
        new_map = self.map.clone()
        new_map.update_sectors(new_sectors)

        for step in range(steps):
            for sector in new_sectors:
                sector.update_sector()

            self.simulate_fire_spread(new_sectors, new_map)

        return new_sectors