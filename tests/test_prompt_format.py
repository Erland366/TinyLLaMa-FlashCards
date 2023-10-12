import pytest

from tinyllama_breakdown.templates.prompt_format import (
    ALPACA_ID_PROMPT_DICT,
    ALPACA_PROMPT_DICT,
    CHATML_ID_PROMPT_DICT,
)


class TestFormat:
    @pytest.fixture(autouse=True, scope="function")
    def set_up(self):
        self.input_format = "prompt_input"
        self.instruction_format = "prompt_instruction"
        self.result_format = "prompt_result"

    def test_alpaca_with_input(self):
        result = ALPACA_PROMPT_DICT["prompt_input"].format(
            instruction=self.instruction_format, user=self.input_format, assistant=""
        )
        print(result)

    def test_alpaca_id_with_input(self):
        result = ALPACA_ID_PROMPT_DICT["prompt_input"].format(
            instruction=self.instruction_format, user=self.input_format, assistant=""
        )
        print(result)

    def test_chatml_id_with_input(self):
        result = CHATML_ID_PROMPT_DICT["prompt_input"].format(
            instruction=self.instruction_format, user=self.input_format, assistant=""
        )
        print(result)

    def test_alpaca_no_input(self):
        result = ALPACA_PROMPT_DICT["prompt_no_input"].format(
            instruction=self.instruction_format, user=self.input_format, assistant=""
        )
        print(result)

    def test_alpaca_id_no_input(self):
        result = ALPACA_ID_PROMPT_DICT["prompt_no_input"].format(
            instruction=self.instruction_format, user=self.input_format, assistant=""
        )
        print(result)

    def test_chatml_id_no_input(self):
        result = CHATML_ID_PROMPT_DICT["prompt_no_input"].format(
            instruction=self.instruction_format, user=self.input_format, assistant=""
        )
        print(result)
