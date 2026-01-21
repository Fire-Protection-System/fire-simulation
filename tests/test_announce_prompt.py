import unittest
from unittest.mock import MagicMock
from datetime import datetime
from src.engine.models.agents.fire_brigade import FireBrigade
from src.engine.models.core.location import Location
from src.engine.models.agents.agent import AgentTask

class DummyLLM:
    def __init__(self, responses):
        self._responses = responses
        self._calls = 0
    def complete(self, user_prompt, system_prompt):
        r = self._responses[self._calls] if self._calls < len(self._responses) else self._responses[-1]
        self._calls += 1
        return r

class TestAnnouncePrompt(unittest.TestCase):
    def setUp(self):
        self.fb = FireBrigade(fire_brigade_id='1', timestamp=datetime.now(), base_location=Location(0,0), initial_location=Location(0,0))
        self.comm = MagicMock()
        self.fb.set_communication(self.comm)

    def test_first_line_used(self):
        llm = DummyLLM(["Line1\nExtra line\nAnother"])
        self.fb.set_llm_client(llm)
        task = AgentTask(task_type='move_to', location=Location(1,1), target_sector_id=8)
        self.fb._announce_order_to_llm(task)
        self.comm.announce_to_llm_chat.assert_called()
        args = self.comm.announce_to_llm_chat.call_args[0][0]
        self.assertIn('description', args)
        self.assertTrue(args['description'].startswith('Line1'))

    def test_truncates_long(self):
        long_line = ' '.join([f'word{i}' for i in range(40)])
        llm = DummyLLM([long_line])
        self.fb.set_llm_client(llm)
        task = AgentTask(task_type='move_to', location=Location(1,1), target_sector_id=8)
        self.fb._announce_order_to_llm(task)
        self.comm.announce_to_llm_chat.assert_called()
        args = self.comm.announce_to_llm_chat.call_args[0][0]
        first = args['description'].split()[0]
        self.assertTrue(len(args['description'].split()) <= 25)

    def test_retry_on_empty(self):
        llm = DummyLLM(["", "Retry line"])
        self.fb.set_llm_client(llm)
        task = AgentTask(task_type='move_to', location=Location(1,1), target_sector_id=8)
        self.fb._announce_order_to_llm(task)
        self.comm.announce_to_llm_chat.assert_called()
        args = self.comm.announce_to_llm_chat.call_args[0][0]
        self.assertTrue(args['description'].startswith('Retry line'))

if __name__ == '__main__':
    unittest.main()
