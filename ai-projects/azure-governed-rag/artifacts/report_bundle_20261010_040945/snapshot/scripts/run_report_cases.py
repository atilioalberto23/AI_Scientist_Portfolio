from pathlib import Path
import json
import sys

import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from azure_governed_rag.pipeline import answer_query


CASES = [
    {
        "id": "RAG001_AUTHORIZED_RISK",
        "question": "Who must authorize a large corporate loan?",
        "user_groups": ["risk"],
        "expected_llm_called": True,
        "expected_doc_ids": ["POL001"],
    },
    {
        "id": "RAG002_UNAUTHORIZED_RANDOM",
        "question": "Who must authorize a large corporate loan?",
        "user_groups": ["random-user"],
        "expected_llm_called": False,
        "expected_doc_ids": [],
    },
    {
        "id": "RAG003_TREASURY_DENIED_TO_RISK",
        "question": "Who may access intraday liquidity reports?",
        "user_groups": ["risk"],
        "expected_llm_called": False,
        "expected_doc_ids": [],
    },
    {
        "id": "RAG004_TREASURY_AUTHORIZED",
        "question": "Who may access intraday liquidity reports?",
        "user_groups": ["treasury"],
        "expected_llm_called": True,
        "expected_doc_ids": ["POL002"],
    },
]


def load_config():

    with (LAB_ROOT / "config.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def main():

    config = load_config()

    output = []

    print(
        "\n"
        "====================================\n"
        " GOVERNED RAG REPORT CASES\n"
        "====================================\n"
    )

    for case in CASES:

        result = answer_query(
            config=config,
            question=case["question"],
            user_groups=case["user_groups"],
            top_k=5,
        )

        source_doc_ids = sorted(
            {
                source["doc_id"]
                for source in result["sources"]
            }
        )

        expected_docs = sorted(
            case["expected_doc_ids"]
        )

        passed = (
            result["llm_called"]
            == case["expected_llm_called"]
            and source_doc_ids == expected_docs
        )

        record = {
            "id": case["id"],
            "question": case["question"],
            "user_groups": case["user_groups"],
            "expected_llm_called":
                case["expected_llm_called"],
            "actual_llm_called":
                result["llm_called"],
            "expected_doc_ids":
                expected_docs,
            "actual_doc_ids":
                source_doc_ids,
            "retrieval_count":
                result["retrieval_count"],
            "answer":
                result["answer"],
            "sources":
                result["sources"],
            "passed":
                passed,
        }

        output.append(record)

        status = "PASS" if passed else "FAIL"

        print(
            f"{case['id']}: {status}"
        )

        print(
            f"  groups: "
            f"{case['user_groups']}"
        )

        print(
            f"  expected docs: "
            f"{expected_docs}"
        )

        print(
            f"  actual docs: "
            f"{source_doc_ids}"
        )

        print(
            f"  LLM called: "
            f"{result['llm_called']}"
        )

        print()

    artifacts_dir = (
        LAB_ROOT / "artifacts"
    )

    artifacts_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        artifacts_dir
        / "rag_report_cases.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    passed_count = sum(
        case["passed"]
        for case in output
    )

    print(
        f"Passed: "
        f"{passed_count}/{len(output)}"
    )

    print(
        f"\nSaved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()