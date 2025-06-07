import collections
import copy
import heapq
import random
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from recomendation.reward_calculator import RewardCalculator
from recomendation.node import Node

from simulation.agent import Agent
from simulation.agent_state import AGENT_STATE
from simulation.fire_brigades.fire_brigade import FireBrigade
from simulation.fire_spread.coef_generator import calculate_beta
from simulation.fire_spread.wind import Wind
from simulation.forest_map import ForestMap
from simulation.forester_patrols.forester_patrol import ForesterPatrol
from simulation.sectors.fire_state import FireState
from simulation.sectors.geographic_direction import GeographicDirection
from simulation.sectors.sector import Sector

class Position(NamedTuple):
    latitude: float
    longitude: float

@dataclass
class CalculationContext:
    active_sectors: List
    num_agents: int
    sector_threat: Dict[int, float]
    required_brigades: Dict[int, int]
    agent_locations: Dict[int, Tuple[float, float]]
    sector_locations: Dict[int, Tuple[float, float]]
    agent_to_sector_distances: Dict[Tuple[int, int], float]
    top_sectors: List
    agent_scores: List[Tuple[int, float]]

class OptimizationConfig:
    MIN_DISTANCE_THRESHOLD = 1
    DISTANCE_PENALTY_FACTOR = 1
    MAX_RANDOM_ATTEMPTS = 10
    MAX_TOTAL_CHILDREN = 30
    AGENT_SECTOR_MULTIPLIER = 2
    TOP_SECTOR_PERCENTAGE = 0.1
    MAX_COORDINATED_SECTORS = 3

class FireSimulationNode(Node):
    __slots__ = ('sectors', 'sector_states', 'agents', 'agent_states', 'time_step', 
                'max_steps', 'action', 'max_brigades', 'map', 'wind', '_children_cache',
                '_sector_lookup', '_active_sector_ids')
    
    def __init__(
        self,
        sectors: List[Sector],
        agents: List[Agent],
        time_step: int,
        max_steps: int = 10,
        action: Optional[List[Tuple[int, int]]] = None,
        max_brigades: int = 2,
        forest_map: Optional[ForestMap] = None,
        wind: Optional[Wind] = None,

        sector_states=None,
        agent_states=None,
        _sector_lookup=None,
        _active_sector_ids=None,
        reward_strategy = ""
    ):
        self.sectors = sectors
        self.reward_strategy = reward_strategy
        
        """Caching every state/object possible to optimize computations."""
        if sector_states is None:
            self.sector_states = frozenset((
                s.sector_id,
                s.fire_state, 
                s.fire_level, 
                s.burn_level, 
                s._number_of_fire_brigades
            ) for s in sectors)
        else:
            self.sector_states = sector_states
                    
        if agent_states is None:
            self.agent_states = tuple((
                a.state, 
                a.location.latitude, 
                a.location.longitude,
                getattr(a.destination, 'latitude', None),
                getattr(a.destination, 'longitude', None)
            ) for a in agents)
        else:
            self.agent_states = agent_states
        
        if _sector_lookup is None:
            self._sector_lookup = {
                s.sector_id: s for s in sectors
            }
        else:
            self._sector_lookup = _sector_lookup
            
        if _active_sector_ids is None:
            self._active_sector_ids = frozenset(s.sector_id for s in sectors if s.fire_state == FireState.ACTIVE)
        else:
            self._active_sector_ids = _active_sector_ids
        
        self.time_step = time_step
        self.max_steps = max_steps
        self.action = tuple(action) if action else None
        self.max_brigades = max_brigades
        self.map = forest_map
        self.wind = wind or Wind()
        self._children_cache = None
        self.agents = agents
        
    def __hash__(self):
        return hash((self.time_step, self.sector_states, self.agent_states))

    def __eq__(self, other):
        return (isinstance(other, FireSimulationNode) and 
                self.time_step == other.time_step and
                self.sector_states == other.sector_states and
                self.agent_states == other.agent_states)

    def _calculate_all_distances(self, agent_locations: Dict, sector_locations: Dict, 
                           sector_ids: Set[int], num_agents: int) -> Dict[Tuple[int, int], float]:
        """ 
            Obliczanie dystansów aktor-sektor
        """
        return {
            (i, sid): np.hypot(
                agent_locations[i][0] - sector_locations[sid].latitude,
                agent_locations[i][1] - sector_locations[sid].longitude
            )
            for i in range(num_agents)
            for sid in sector_ids
        }

    def _calculate_top_k(self, num_sectors: int, num_agents: int) -> int:
        """
            Opytmalizacja: Top k sektorów pod względem scoringu do rozważenia
        """
        return max(1, min(
            num_sectors, 
            num_agents * OptimizationConfig.AGENT_SECTOR_MULTIPLIER,
            int(len(self.sectors) * OptimizationConfig.TOP_SECTOR_PERCENTAGE)
        ))

    def _calculate_agent_scores(self, num_agents: int, top_sectors: List, 
                          sector_threat: Dict, distances: Dict) -> List[Tuple[int, float]]:
        """
            Scoring sektorów pod względem zagrożenia.
            Scoring = zagrożenie sektora / dystans danego agenta. 
        """
        agent_scores = []
        for i in range(num_agents):
            score = sum(
                sector_threat[s.sector_id] / max(OptimizationConfig.MIN_DISTANCE_THRESHOLD, 
                                            distances[(i, s.sector_id)])
                for s in top_sectors
            )
            agent_scores.append((i, score))
        
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores

    def _generate_greedy_assignments(self, context: CalculationContext) -> Set:
        """
            Zachłanna strategia szukania rozwiązania.
            Najbardziej zagrożone, jednocześnie najbliższe sektory.
        """
        children = set()
        assigned_sectors = set()
        threat_actions = []
        sector_brigade_counts = collections.defaultdict(int)
        
        for agent_id, _ in context.agent_scores:
            best_assignment = self._find_best_sector_for_agent(
                agent_id, context, assigned_sectors
            )
            
            if best_assignment:
                sector_id = best_assignment[0]
                if sector_brigade_counts[sector_id] < context.required_brigades[sector_id]:
                    threat_actions.append((agent_id, sector_id))
                    assigned_sectors.add(sector_id)
                    sector_brigade_counts[sector_id] += 1
        
        if threat_actions:
            children.add(self._apply_step(threat_actions))
        
        return children

    def _find_best_sector_for_agent(self, agent_id: int, context: CalculationContext, 
                               assigned_sectors: Set[int]) -> Optional[Tuple[int, float]]:
        """
            Znajdowanie najlepszego sektora dla danego agenta
        """ 
        best_options = [
            (sid, context.sector_threat[sid] / (1 + context.agent_to_sector_distances[(agent_id, sid)] * OptimizationConfig.DISTANCE_PENALTY_FACTOR))
            for sid in context.sector_threat
            if sid not in assigned_sectors
        ]
        
        if best_options:
            return max(best_options, key=lambda x: x[1])
        return None

    def _generate_coordinated_assignments(self, context: CalculationContext) -> Set:
        """
            Podział wszystkich brygad w większe grupy. 
        """
        children = set()
        
        if context.num_agents <= 1:
            return children
        
        max_sectors_to_process = min(OptimizationConfig.MAX_COORDINATED_SECTORS, 
                                    len(context.top_sectors))
        
        for sector in context.top_sectors[:max_sectors_to_process]:
            sector_id = sector.sector_id
            required = context.required_brigades[sector_id]
            
            if required == 0:
                continue
            
            assignment = self._create_coordinated_assignment(
                sector_id, required, context
            )
            
            if assignment:
                children.add(self._apply_step(assignment))
        
        return children

    def _create_coordinated_assignment(self, sector_id: int, required_brigades: int, 
                                 context: CalculationContext) -> List[Tuple[int, int]]:
        """ Rozdysponowanie wielu brygad do danego sektora """
        closest_agents = sorted(
            [(i, context.agent_to_sector_distances[(i, sector_id)]) 
            for i in range(context.num_agents)],
            key=lambda x: x[1]
        )[:min(required_brigades, context.num_agents)]
        
        return [(agent_id, sector_id) for agent_id, _ in closest_agents]

    def _generate_probabilistic_assignments(self, context: CalculationContext) -> Set:
        """ Random selection"""
        children = set()
        seen_assignments = set()
        max_random = min(OptimizationConfig.MAX_RANDOM_ATTEMPTS, 
                        OptimizationConfig.MAX_TOTAL_CHILDREN - len(children))
        
        agent_sector_probs = self._calculate_assignment_probabilities(context)
        
        for _ in range(max_random):
            assignment = self._generate_random_assignment(
                context, agent_sector_probs
            )
            
            if assignment and assignment not in seen_assignments:
                seen_assignments.add(assignment)
                children.add(self._apply_step(list(assignment)))
        
        return children

    def _calculate_assignment_probabilities(self, context: CalculationContext) -> Dict:
        agent_sector_probs = {}
        sector_ids = list(context.sector_threat.keys())
        
        for i in range(context.num_agents):
            probs = np.array([
                context.sector_threat[sid] / (1 + context.agent_to_sector_distances[(i, sid)])
                for sid in sector_ids
            ])
            
            if probs.sum() > 0:
                probs = probs / probs.sum()
                agent_sector_probs[i] = (sector_ids, probs)
        
        return agent_sector_probs

    def _generate_random_assignment(self, context: CalculationContext, 
                                agent_probs: Dict) -> Optional[Tuple]:
        actions = []
        temp_brigade_counts = collections.defaultdict(int)
        
        for i in range(context.num_agents):
            if i in agent_probs:
                sector_ids, probs = agent_probs[i]
                chosen = np.random.choice(sector_ids, p=probs)
                
                if temp_brigade_counts[chosen] < context.required_brigades[chosen]:
                    actions.append((i, chosen))
                    temp_brigade_counts[chosen] += 1
        
        return tuple(sorted(actions)) if actions else None


    def _prepare_calculation_context(self) -> CalculationContext:
        """
            Precomputing
        """
        active_sectors = [self._sector_lookup[sid] for sid in self._active_sector_ids]
        num_agents = len(self.agents)
        
        if hasattr(self, '_cached_context') and self._cached_context['timestamp'] > time.time() - 0.1:
            return self._cached_context['data']

        sector_threat = {
            s.sector_id: s.fire_level * (1 + s.burn_level)
            for s in active_sectors
        }

        required_brigades = {
            s.sector_id: s.required_fire_brigades()
            for s in active_sectors
        }
        
        agent_locations = {
            i: (self.agents[i].location.latitude, self.agents[i].location.longitude)
            for i in range(num_agents)
        }

        sector_locations = {
            s.sector_id: self.map.get_sector_location(s)
            for s in active_sectors
        }
        
        agent_to_sector_distances = self._calculate_all_distances(
            agent_locations, sector_locations, sector_threat.keys(), num_agents
        )
        
        top_k = self._calculate_top_k(len(active_sectors), num_agents)
        top_sectors = heapq.nlargest(top_k, active_sectors, 
                                    key=lambda s: sector_threat[s.sector_id])
        
        agent_scores = self._calculate_agent_scores(
            num_agents, top_sectors, sector_threat, agent_to_sector_distances
        )
        
        return CalculationContext(
            active_sectors=active_sectors,
            num_agents=num_agents,
            sector_threat=sector_threat,
            required_brigades=required_brigades,
            agent_locations=agent_locations,
            sector_locations=sector_locations,
            agent_to_sector_distances=agent_to_sector_distances,
            top_sectors=top_sectors,
            agent_scores=agent_scores
        )
    
    def _calculate_distance(self, loc1, loc2):
        """Calculate distance between two locations using numpy for speed"""
        lat_diff = loc1.latitude - loc2.latitude
        lon_diff = loc1.longitude - loc2.longitude
        return np.sqrt(lat_diff**2 + lon_diff**2)

    def find_children(self):
        if self._children_cache is not None:
            return self._children_cache

        if not self._active_sector_ids:
            self._children_cache = set()
            return self._children_cache

        context = self._prepare_calculation_context()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._generate_greedy_assignments, context),
                executor.submit(self._generate_coordinated_assignments, context),
                executor.submit(self._generate_probabilistic_assignments, context),
            ]
            children = set()
            for future in futures:
                children.update(future.result())

        self._children_cache = children
        return children

    def find_random_child(self):
        """Find a random child using cached active sectors"""
        if not self._active_sector_ids:
            return None
            
        active_sectors = [self._sector_lookup[sid] for sid in self._active_sector_ids]
        num_agents = len(self.agents)
        actions = []
        
        if len(active_sectors) >= num_agents:
            chosen_sectors = random.sample(active_sectors, num_agents)
            for i, sector in enumerate(chosen_sectors):
                actions.append((i, sector.sector_id))
        else:
            for i in range(num_agents):
                actions.append((i, random.choice(active_sectors).sector_id))
        
        return self._apply_step(actions)

    def _apply_step(self, actions: List[Tuple[int, int]]):
        """Apply actions and create new state with optimized object creation"""
        sectors_to_modify = set()
        for _, sector_id in actions:
            sectors_to_modify.add(sector_id)
            
        new_sectors_map = {}
        new_sectors = []

        for s in self.sectors:
            cloned = s.clone()
            new_sectors_map[s.sector_id] = cloned
            new_sectors.append(cloned)
        
        new_map = self.map.clone()
        new_map.update_sectors(new_sectors)
        
        new_agents = []
        for i, agent in enumerate(self.agents):
            agent_actions = [a for a in actions if a[0] == i] 
            if agent_actions:
                new_agent = agent.clone()
                new_agents.append(new_agent)
            else:
                new_agents.append(agent)
        
        for agent_idx, sector_id in actions:
            agent = new_agents[agent_idx]
            dest_sector = new_map.get_sector(sector_id)
            dest_position = new_map.get_sector_location(dest_sector)
            agent.set_state_travelling(dest_position)
        
        for agent in new_agents:
            if agent.state == AGENT_STATE.TRAVELLING:
                if self._update_position(agent):
                    agent._location = agent.destination
                
                    if agent.destination == agent.base_location:
                        agent.set_state_available()
                    else:
                        sec = new_map.find_sector(agent.location)
                        sec._number_of_fire_brigades += 1
                        agent.set_state_executing()

        updated_sector_ids = set()
        for agent in new_agents:
            if agent.state == AGENT_STATE.EXECUTING:
                sec = new_map.find_sector(agent.location)

                if sec.sector_id not in updated_sector_ids:
                    if sec is not new_sectors_map[sec.sector_id]:
                        sec = sec.clone()
                        new_sectors_map[sec.sector_id] = sec
                    updated_sector_ids.add(sec.sector_id)

                if isinstance(agent, FireBrigade):
                    sec.update_extinguish_level()
                    
                if hasattr(agent, "is_task_finished") and agent.is_task_finished(sec):
                    agent.set_state_available()

        new_active_sector_ids = set()
        for sector in new_sectors:
            sector.update_sector()
            if sector.fire_state == FireState.ACTIVE:
                new_active_sector_ids.add(sector.sector_id)
        
        self._spread_fire(new_sectors, new_map, copy.copy(self.wind), new_active_sector_ids)

        new_sector_states = frozenset((s.sector_id, s.fire_state, s.fire_level, s.burn_level, 
                                      s._number_of_fire_brigades) for s in new_sectors)
                                      
        new_agent_states = tuple((a.state, a.location.latitude, a.location.longitude,
                                getattr(a.destination, 'latitude', None),
                                getattr(a.destination, 'longitude', None)) for a in new_agents)
                                
        new_sector_lookup = {s.sector_id: s for s in new_sectors}
        new_active_sector_ids = frozenset(s.sector_id for s in new_sectors if s.fire_state == FireState.ACTIVE)

        return FireSimulationNode(
            sectors=new_sectors,
            agents=new_agents,
            time_step=self.time_step + 1,
            max_steps=self.max_steps,
            action=actions,
            max_brigades=self.max_brigades,
            forest_map=new_map,
            wind=copy.copy(self.wind),
            sector_states=new_sector_states,
            agent_states=new_agent_states,
            _sector_lookup=new_sector_lookup,
            _active_sector_ids=new_active_sector_ids
        )

    def calculate_reward(self, state):
        calculator = RewardCalculator(self.reward_strategy)
        return calculator.calculate_reward(state)

    def is_terminal(self):
        """Optimized terminal state check using cached active sectors"""
        return self.time_step >= self.max_steps or not self._active_sector_ids

    def reward(self):
        """Calculate reward"""
        return self.calculate_reward(self)

    def _update_position(self, agent: Agent) -> bool:
        """Update agent position with optimized movement calculation"""
        delta = 0.005

        dest_lat, dest_lon = agent.destination.latitude, agent.destination.longitude
        curr_lat, curr_lon = agent.location.latitude, agent.location.longitude
        
        lat_diff = dest_lat - curr_lat
        lon_diff = dest_lon - curr_lon
        
        agent._location.latitude += min(delta, lat_diff) if lat_diff > 0 else max(-delta, lat_diff)
        agent._location.longitude += min(delta, lon_diff) if lon_diff > 0 else max(-delta, lon_diff)

        return (abs(lat_diff) <= 0.001 and abs(lon_diff) <= 0.001)

    def _spread_fire(self, sectors: List[Sector], forest_map: ForestMap, wind: Wind, active_sector_ids=None):
        """Optimized fire spread simulation using cached data structures"""

        if active_sector_ids is None:
            active_sectors = [s for s in sectors if s.fire_state == FireState.ACTIVE]
        else:
            active_sectors = [s for s in sectors if s.sector_id in active_sector_ids]
        
        sectors_to_activate = []
        sector_lookup = {s.sector_id: s for s in sectors}
        
        adjacency_map = {}
        for sector in active_sectors:
            adjacency_map[sector.sector_id] = forest_map.get_adjacent_sectors(sector)

        for sector in active_sectors:
            for neighbor, direction in adjacency_map[sector.sector_id]:
                if neighbor.fire_state == FireState.INACTIVE:
                    prob = calculate_beta(wind, neighbor.sector_type, direction)
                    if random.random() < prob:
                        sectors_to_activate.append(neighbor)
        
        for sector in sectors_to_activate:
            sector.start_fire()
            
        wind.update_wind()

    def simulate_steps_without_action(self, steps: int) -> List[Sector]:
        """Simulate future steps without agent actions - optimized version"""
        active_sectors = {s.sector_id for s in self.sectors if s.fire_state == FireState.ACTIVE}
        
        clones_map = {}
        clones = []
        
        sectors_to_clone = set(active_sectors)
        for s in self.sectors:
            if s.fire_state == FireState.ACTIVE:
                for neighbor, _ in self.map.get_adjacent_sectors(s):
                    sectors_to_clone.add(neighbor.sector_id)
        
        for s in self.sectors:
            if s.sector_id in sectors_to_clone:
                clone = s.clone()
                clones_map[s.sector_id] = clone
                clones.append(clone)
            else:
                clones_map[s.sector_id] = s
                clones.append(s)
        
        fmap = self.map.clone()
        fmap.update_sectors(clones)
        w = copy.copy(self.wind)
        
        active_sector_ids = frozenset(s.sector_id for s in clones if s.fire_state == FireState.ACTIVE)

        for _ in range(steps):
            sectors_to_update = set()
            
            for s_id in active_sector_ids:
                s = clones_map[s_id]
                if s is self._sector_lookup.get(s_id):
                    s = s.clone()
                    clones_map[s_id] = s
                    idx = clones.index(self._sector_lookup[s_id])
                    clones[idx] = s
                s.update_sector()
                
                for neighbor, _ in fmap.get_adjacent_sectors(s):
                    sectors_to_update.add(neighbor.sector_id)
                
            self._spread_fire(clones, fmap, w, active_sector_ids)
            
            active_sector_ids = frozenset(s.sector_id for s in clones if s.fire_state == FireState.ACTIVE)

        return clones

    def simulate_steps_with_action(self, steps: int, actions: List[Tuple[int, int]]) -> Tuple[List[Sector], List[Agent]]:
        active_sectors = {s.sector_id for s in self.sectors if s.fire_state == FireState.ACTIVE}
        
        new_sectors = [s.clone() for s in self.sectors]
        new_sectors_map = {s.sector_id: s for s in new_sectors}

        new_map = self.map.clone()
        new_map.update_sectors(new_sectors)

        new_agents = []
        for i, agent in enumerate(self.agents):
            if any(a[0] == i for a in actions):
                new_agents.append(agent.clone())
            else:
                new_agents.append(agent)
                
        for agent_idx, sector_id in actions:
            agent = new_agents[agent_idx]
            dest_sector = new_map.get_sector(sector_id)
            dest_position = new_map.get_sector_location(dest_sector)
            agent.set_state_travelling(dest_position)

        for _ in range(steps):
            updated_sector_ids = set()

            for agent in new_agents:
                if agent.state == AGENT_STATE.TRAVELLING:
                    if self._update_position(agent):
                        agent._location = agent.destination

                        if agent.destination == agent.base_location:
                            agent.set_state_available()
                        else:
                            sec = new_map.find_sector(agent.location)
                            sec._number_of_fire_brigades += 1
                            agent.set_state_executing()

                if agent.state == AGENT_STATE.EXECUTING:
                    sec = new_map.find_sector(agent.location)
                    if sec and sec.sector_id not in updated_sector_ids:
                        if sec is not new_sectors_map[sec.sector_id]:
                            sec = sec.clone()
                            new_sectors_map[sec.sector_id] = sec
                            new_map.update_sectors([sec]) 
                        updated_sector_ids.add(sec.sector_id)

                    sec.update_extinguish_level()

                    if agent.is_task_finished(sec):
                        agent.set_state_available()

            new_active_sector_ids = set()
            for sector in new_sectors:
                sector.update_sector()
                if sector.fire_state == FireState.ACTIVE:
                    new_active_sector_ids.add(sector.sector_id)

            self._spread_fire(new_sectors, new_map, copy.copy(self.wind), new_active_sector_ids)

        # print()
        # for a in new_agents:
        #     print(f"Agent {a.fire_brigade_id} state: {a.state.name}")
    
        return new_sectors