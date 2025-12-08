from enum import Enum
from src.engine.models.sensors.sensor_type import SensorType

class TopicRegistry(str, Enum):
    
    TEMPERATURE_HUMIDITY = "simulation.telemetry.sensors.temp_humidity"
    WIND_SPEED           = "simulation.telemetry.sensors.wind_speed"
    WIND_DIRECTION       = "simulation.telemetry.sensors.wind_direction"
    LITTER_MOISTURE      = "simulation.telemetry.sensors.litter_moisture"
    CO2                  = "simulation.telemetry.sensors.co2"
    PM2_5                = "simulation.telemetry.sensors.pm2_5"
    CAMERA               = "simulation.telemetry.sensors.camera"
    SECTOR_STATE         = "simulation.telemetry.map.sector_state"
    FORESTER_STATE       = "simulation.telemetry.agents.forester"
    FIRE_BRIGADE_STATE   = "simulation.telemetry.agents.fire_brigade"
    EVENTS               = "simulation.events"
    FORESTER_ACTIONS     = "simulation.control.forester_actions"
    FIRE_BRIGADE_ACTIONS = "simulation.control.fire_brigade_actions"
    RECOMMENDATIONS      = "simulation.recommendations"

SENSOR_TO_TOPIC = {
    SensorType.TEMPERATURE_AND_AIR_HUMIDITY: TopicRegistry.TEMPERATURE_HUMIDITY,
    SensorType.WIND_SPEED:                   TopicRegistry.WIND_SPEED,
    SensorType.WIND_DIRECTION:               TopicRegistry.WIND_DIRECTION,
    SensorType.LITTER_MOISTURE:              TopicRegistry.LITTER_MOISTURE,
    SensorType.CO2:                          TopicRegistry.CO2,
    SensorType.PM2_5:                        TopicRegistry.PM2_5,
    SensorType.CAMERA:                       TopicRegistry.CAMERA,
}


def get_topic_for_sensor(sensor_type: SensorType) -> str:
    topic = SENSOR_TO_TOPIC.get(sensor_type)
    if topic is None:
        raise ValueError(f"No topic mapping for sensor type: {sensor_type}")
    return topic.value
