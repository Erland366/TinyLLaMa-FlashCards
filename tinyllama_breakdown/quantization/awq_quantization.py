from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


class QuantizeAWQ:
    def __init__(self, cfg) -> None:
        self.model_id = cfg.main.model_name_or_path
        self.cfg = cfg
        self.model, self.tokenizer = self.initialize_model(cfg)

    def initialize_model(self, cfg) -> None:
        try:
            model = AutoAWQForCausalLM.from_pretrained(**cfg.main)
            tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
        except TypeError:
            raise TypeError(
                "Arguments not found for AWQ model. Are you sure you are using AWQ config?"  # noqa: E501
            )

        return model, tokenizer

    def quantize(self, verbose: bool = False) -> None:
        quant_config = self.cfg.quant_config
        quant_path = self.cfg.extra.quant_path

        self.model.quantize(
            self.tokenizer,
            quant_config=quant_config,
            calib_data=self.cfg.extra.calib_data,
            text_column=self.cfg.extra.text_column,
        )

        self.model.save_quantized(quant_path)
        self.tokenizer.save_pretrained(quant_path)

        print(f"Model is quantized and saved at '{quant_path}'.")
