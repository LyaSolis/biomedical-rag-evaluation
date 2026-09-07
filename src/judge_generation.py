import argparse
import csv
import json
from pathlib import Path

from ollama import chat


DEFAULT_INPUT = "data/generation_evaluation.jsonl"
DEFAULT_OUTPUT_JSONL = "data/generation_semantic_judgments.jsonl"
DEFAULT_OUTPUT_CSV = "data/generation_semantic_review.csv"
MODEL_NAME = "qwen3:8b"


JUDGE_SYSTEM_PROMPT = """
You are evaluating a biomedical RAG system.

You must judge the generated answer using ONLY the question, the PubMedQA
reference answer/decision, and the retrieved context supplied in the evaluation
record. Do not use outside biomedical knowledge to fill missing evidence.

Evaluate four dimensions:

1. answer_correctness:
   - correct: substantively answers the question consistently with the reference
     answer, without a material factual error.
   - partially_correct: contains a substantially correct answer but is incomplete
     or has a minor substantive problem.
   - incorrect: materially conflicts with the reference answer or fails to answer
     the question when a supported answer was available.
   - unclear: the evidence/reference is too ambiguous to make a reliable judgment.

2. evidence_grounding:
   - grounded: substantive claims in the generated answer are supported by the
     retrieved context.
   - partially_grounded: some substantive claims are supported but others are
     unsupported or go beyond the context.
   - unsupported: the answer makes substantive claims that the retrieved context
     does not support.
   - unclear: grounding cannot be determined reliably.

3. abstention_behavior:
   - appropriate_abstention: the retrieved context is insufficient for the
     question and the model appropriately avoids unsupported claims.
   - answered_with_insufficient_evidence: the retrieved context is insufficient
     but the model gives a substantive answer that is not supported by it.
   - over_abstention: the retrieved context contains sufficient evidence for a
     useful answer but the model declines to answer.
   - not_applicable: the model gives a supported answer and abstention is not
     relevant.

4. failure_attribution:
   - retrieval_failure: the required evidence is absent or inadequate in the
     retrieved context; generation behavior is not the primary problem.
   - generation_failure: adequate evidence is present, but the generated answer
     is materially wrong, unsupported, or unnecessarily incomplete.
   - retrieval_and_generation: both evidence quality and generation behavior
     materially contribute to the failure.
   - appropriate_abstention: the evidence is insufficient and the model
     appropriately abstains.
   - no_failure: the answer is correct and appropriately grounded.
   - unclear: attribution cannot be determined reliably.

Important:
- Do not treat PubMed ID matching as proof that the context contains answer-
  bearing evidence.
- Do not treat lexical similarity between the generated and reference answers
  as proof of correctness.
- Do not penalize different wording when the substantive meaning is equivalent.
- A correct abstention is not a generation failure.
- If the context is insufficient, distinguish appropriate abstention from an
  unsupported answer.
- Base the judgment on the supplied material, not general knowledge.

Return ONLY a JSON object with exactly these keys:
{
  "answer_correctness": "...",
  "evidence_grounding": "...",
  "abstention_behavior": "...",
  "failure_attribution": "...",
  "confidence": "high|medium|low",
  "reason": "brief explanation grounded in the supplied material"
}

Do not include markdown fences or any other text.
"""


def load_records(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_user_prompt(record):
    context_text = "\n\n".join(
        (
            f"[Context rank {item['rank']} | PubMed ID {item['pubid']} | "
            f"chunk {item['chunk_index']}]\n{item['text']}"
        )
        for item in record.get("retrieved_context", [])
    )

    return f"""
Evaluate this single RAG answer.

QUESTION:
{record["question"]}

PUBMEDQA REFERENCE DECISION:
{record["reference_decision"]}

PUBMEDQA REFERENCE ANSWER:
{record["reference_answer"]}

GENERATED ANSWER:
{record["generated_answer"]}

RETRIEVED CONTEXT:
{context_text}
"""


def judge_record(record):
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        options={"temperature": 0},
        format="json",
    )

    content = response["message"]["content"]
    judgment = json.loads(content)

    required = [
        "answer_correctness",
        "evidence_grounding",
        "abstention_behavior",
        "failure_attribution",
        "confidence",
        "reason",
    ]

    missing = [key for key in required if key not in judgment]
    if missing:
        raise ValueError(f"Judge response missing keys: {missing}")

    return judgment


def write_csv(records, judgments, path):
    fieldnames = [
        "pubid",
        "question",
        "reference_decision",
        "reference_answer",
        "generated_answer",
        "source_rank",
        "source_in_top5",
        "answer_correctness",
        "evidence_grounding",
        "abstention_behavior",
        "failure_attribution",
        "confidence",
        "review_notes",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record, judgment in zip(records, judgments):
            source_ranks = [
                item["rank"]
                for item in record.get("retrieved_context", [])
                if str(item.get("pubid")) == str(record.get("pubid"))
            ]

            writer.writerow(
                {
                    "pubid": record.get("pubid"),
                    "question": record.get("question"),
                    "reference_decision": record.get("reference_decision"),
                    "reference_answer": record.get("reference_answer"),
                    "generated_answer": record.get("generated_answer"),
                    "source_rank": source_ranks[0] if source_ranks else "",
                    "source_in_top5": bool(source_ranks),
                    "answer_correctness": judgment["answer_correctness"],
                    "evidence_grounding": judgment["evidence_grounding"],
                    "abstention_behavior": judgment["abstention_behavior"],
                    "failure_attribution": judgment["failure_attribution"],
                    "confidence": judgment["confidence"],
                    "review_notes": judgment["reason"],
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Judge RAG answers for correctness, grounding, abstention, and failure attribution."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of records to judge. 0 means all records.",
    )
    args = parser.parse_args()

    # Never overwrite an existing experiment artifact.
    jsonl_path = Path(args.output_jsonl)
    csv_path = Path(args.output_csv)

    if jsonl_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {jsonl_path}. "
            "Choose a new output path for a new experiment."
        )

    if csv_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {csv_path}. "
            "Choose a new output path for a new experiment."
        )

    records = load_records(args.input)
    if args.limit > 0:
        records = records[:args.limit]

    print(f"Records to judge: {len(records)}")
    print(f"Judge model: {MODEL_NAME}")

    judgments = []

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] PubMed ID {record['pubid']}")
        judgment = judge_record(record)
        judgments.append(judgment)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for record, judgment in zip(records, judgments):
            output = {
                "pubid": record["pubid"],
                "question": record["question"],
                "judgment": judgment,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    write_csv(records, judgments, csv_path)

    print(f"Saved judgments to: {jsonl_path}")
    print(f"Saved review CSV to: {csv_path}")
    print(f"Completed: {len(judgments)}")


if __name__ == "__main__":
    main()
