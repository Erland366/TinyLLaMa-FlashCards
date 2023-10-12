import pytest

from inference import TinyLLaMaInference
from tinyllama_breakdown.utils import read_yaml_file


class TestInference:
    @pytest.fixture(scope="function", autouse=True)
    def set_up(self):
        self.cfg = read_yaml_file("config/inference_awq.yaml")

    def test_check_config(self):
        # self.set_up()
        TinyLLaMaInference.check_config(self.cfg)
        assert self.cfg is not None
