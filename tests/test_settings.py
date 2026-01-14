"""
Tests for settings module.
"""
import os
import pytest
from settings.communucation_settings import get_settings, DEFAULT_EXCHANGE, DEFAULT_RABBITMQ_HOST


def test_get_settings_defaults():
    settings = get_settings()
    assert settings.rabbitmq_host == DEFAULT_RABBITMQ_HOST
    assert settings.exchange_name == DEFAULT_EXCHANGE
    assert settings.tick_interval == 5.0


def test_get_settings_with_env_override(monkeypatch):
    monkeypatch.setenv("RABBITMQ_HOST", "custom-host")
    monkeypatch.setenv("FIRE_SIMULATION_EXCHANGE_NAME", "custom-exchange")
    monkeypatch.setenv("TICK_INTERVAL", "10.0")
    
    settings = get_settings()
    assert settings.rabbitmq_host == "custom-host"
    assert settings.exchange_name == "custom-exchange"
    assert settings.tick_interval == 10.0


def test_settings_type_validation():
    settings = get_settings()
    assert isinstance(settings.rabbitmq_port, int)
    assert isinstance(settings.tick_interval, float)
    assert isinstance(settings.exchange_name, str)
