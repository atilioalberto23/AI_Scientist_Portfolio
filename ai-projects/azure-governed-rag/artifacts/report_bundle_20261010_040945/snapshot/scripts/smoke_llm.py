from pathlib import Path
import sys

import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = LAB_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from azure_governed_rag.llm import (
    build_client,
    run_inference,
)


def load_config():

    path = LAB_ROOT / "config.yaml"

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def main():

    config = load_config()

    azure = config["azure"]

    print(
        "\n"
        "====================================\n"
        " AZURE GOVERNED RAG · SMOKE TEST\n"
        "====================================\n"
    )

    client = build_client(
        azure["openai_base_url"]
    )

    output = run_inference(
        client=client,
        deployment=azure["deployment"],
        prompt=(
            "Return exactly the string "
            "AZURE_SLICE_OK"
        ),
    )

    print("Response:")
    print(output)

    if output.strip() != "AZURE_SLICE_OK":

        raise RuntimeError(
            "Unexpected model response."
        )

    print(
        "\n✓ Identity → deployment → "
        "inference path verified.\n"
    )


if __name__ == "__main__":
    main()