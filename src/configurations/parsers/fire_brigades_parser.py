
from datetime import datetime
from typing import List, Dict, Any
import logging

from src.engine.models.core.location import Location
from src.engine.models.agents.fire_brigade import FireBrigade
from src.engine.models.agents.fire_brigade_state import FIREBRIGADE_STATE

logger = logging.getLogger(__name__)

def parse_fire_brigades(data: List[Dict[str, Any]]) -> List[FireBrigade]:
    '''
    Parsing fire brigades data from JSON input from selected configuration
    
    :param data: json input containing fire brigades information
    :type data: List[Dict[str, Any]]
    :return: list of FireBrigade objects parsed from the input data
    :rtype: List[FireBrigade]
    '''
    
    brigades = []
    
    fire_brigades_data = data.get('fireBrigades', [])
    if not isinstance(fire_brigades_data, list):
        logger.error("'fireBrigades' is missing or is not a list")
        return brigades
    
    for item in fire_brigades_data:
        try:
            fire_brigade_id = item["fireBrigadeId"]
            timestamp = datetime.fromisoformat(item["timestamp"]) 
            state = FIREBRIGADE_STATE[item["state"]]
            
            base_location = Location(
                longitude = float(item["baseLocation"]["longitude"]),
                latitude  = float(item["baseLocation"]["latitude"])
            )
            current_location = Location(
                longitude = float(item["currentLocation"]["longitude"]),
                latitude  = float(item["currentLocation"]["latitude"])
            )

            brigades.append(FireBrigade(
                fire_brigade_id=fire_brigade_id,
                timestamp=timestamp,
                initial_state=state,
                base_location=base_location,
                initial_location=current_location
            ))
       
        except KeyError as e:
            logger.error(f"ERROR: Fire Brigade Parser Missing key in data: {e}")
        
        except ValueError as e:
            logger.error(f"ERROR: Fire Brigade Parser Value error: {e}")
        
        except TypeError as e:
            logger.error(f"ERROR: Fire Brigade Parser Type error: {e}")
        
    return brigades
