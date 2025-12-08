"""
Tests for SimpleSimulationEngine.
"""
import pytest
import asyncio
from src.engine.simple_simulation_engine import SimpleSimulationEngine


@pytest.fixture
def engine():
    return SimpleSimulationEngine()


@pytest.fixture
def sample_config():
    return {
        "map": {
            "rows": 3,
            "cols": 3
        },
        "agents": []
    }


def test_engine_initialization(engine):
    assert engine.config is None
    assert engine._map is None
    assert not engine.is_running()
    assert engine._tick_count == 0


@pytest.mark.asyncio
async def test_load_config(engine, sample_config):
    await engine.load_config(sample_config)
    assert engine.config is not None
    assert engine._map is not None
    assert len(engine.all_sectors) == 9  # 3x3
    assert engine._tick_count == 0


@pytest.mark.asyncio
async def test_start_without_config_raises_error(engine):
    with pytest.raises(ValueError, match="Configuration not loaded"):
        await engine.start()


@pytest.mark.asyncio
async def test_start_and_stop(engine, sample_config):
    await engine.load_config(sample_config)
    await engine.start()
    assert engine.is_running()
    
    await engine.stop()
    assert not engine.is_running()


@pytest.mark.asyncio
async def test_snapshot(engine, sample_config):
    await engine.load_config(sample_config)
    snapshot = engine.snapshot()
    
    assert "tick" in snapshot
    assert "running" in snapshot
    assert "fire_count" in snapshot
    assert "total_sectors" in snapshot
    assert snapshot["config_loaded"] is True


@pytest.mark.asyncio
async def test_step_returns_correct_structure(engine, sample_config):
    await engine.load_config(sample_config)
    await engine.start()
    
    result = await engine.step(1)
    
    assert "tick" in result
    assert "sensor_messages" in result
    assert "sector_states" in result
    assert "agent_states" in result
    assert "events" in result
    assert result["tick"] == 1


@pytest.mark.asyncio
async def test_multiple_steps_increment_tick(engine, sample_config):
    await engine.load_config(sample_config)
    await engine.start()
    
    result1 = await engine.step(1)
    assert result1["tick"] == 1
    
    result2 = await engine.step(2)
    assert result2["tick"] == 3  # 1 + 2


@pytest.mark.asyncio
async def test_pause_not_implemented(engine, sample_config):
    await engine.load_config(sample_config)
    
    with pytest.raises(NotImplementedError):
        await engine.pause()
