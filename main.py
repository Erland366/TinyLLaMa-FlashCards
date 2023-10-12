import logging

from dotenv import load_dotenv

from tinyllama_breakdown.tinyllama.finetune import finetune_tinyllama
from tinyllama_breakdown.utils import (
    check_user_token,
    load_user_token,
    read_yaml_file,
)

LOG = logging.getLogger("tinyllama_breakdown")

load_dotenv()


def main() -> None:
    training_config = read_yaml_file("config/finetune.yaml")

    LOG.info("Load user HuggingFace token")
    load_user_token()

    LOG.info("Check if user is logged in")
    check_user_token()

    LOG.info("Start finetuning")
    finetune_tinyllama(training_config)


if __name__ == "__main__":
    main()
