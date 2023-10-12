from tinyllama_breakdown.utils import read_yaml_file


class TestConfig:
    def test_load_config(self):
        cfg = read_yaml_file("config/finetune.yaml")
        print(cfg)
