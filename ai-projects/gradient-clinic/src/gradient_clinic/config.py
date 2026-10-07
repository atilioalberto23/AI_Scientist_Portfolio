from pathlib import Path

import yaml


def load_config(path):
    """
    Load a Gradient Clinic experiment configuration.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML dictionary."
        )

    return config