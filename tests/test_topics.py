"""
Tests for messaging topics module.
"""
import pytest
from src.messaging.topics import get_topic_for_sensor, TopicRegistry, SENSOR_TO_TOPIC
from src.engine.models.sensors.sensor_type import SensorType


def test_sensor_topic_mapping():
    topic = get_topic_for_sensor(SensorType.TEMPERATURE_AND_AIR_HUMIDITY)
    assert topic == TopicRegistry.TEMPERATURE_HUMIDITY.value
    assert "simulation.telemetry.sensors" in topic


def test_all_sensor_types_mapped():
    for sensor_type in SensorType:
        topic = get_topic_for_sensor(sensor_type)
        assert topic.startswith("simulation.telemetry.sensors.")


def test_unknown_sensor_raises_error():
    class FakeSensor:
        pass
    
    fake = FakeSensor()
    with pytest.raises(ValueError, match="No topic mapping"):
        get_topic_for_sensor(fake)


def test_all_topics_have_simulation_prefix():
    for topic in TopicRegistry:
        assert topic.value.startswith("simulation."), f"Invalid topic: {topic.value}"


def test_control_topics_exist():
    assert TopicRegistry.FORESTER_ACTIONS.value == "simulation.control.forester_actions"
    assert TopicRegistry.FIRE_BRIGADE_ACTIONS.value == "simulation.control.fire_brigade_actions"


def test_telemetry_topics_exist():
    assert TopicRegistry.SECTOR_STATE.value == "simulation.telemetry.map.sector_state"
    assert TopicRegistry.EVENTS.value == "simulation.events"
