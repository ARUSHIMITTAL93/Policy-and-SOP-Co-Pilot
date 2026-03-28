from __future__ import annotations

import argparse
import json

from openai import OpenAI

from retrieve import get_client, retrieve_chunks


ANSWER_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 5
MAX_OUTPUT_TOKENS = 500

DEVELOPER_PROMPT = """
You answer questions about policy and SOP documents using only the retrieved context provided.

Rules:
- Use only the supplied context chunks.
- If the answer is not supported by the context, say that the documents do not provide enough evidence.
- Cite factual claims with bracketed chunk references like [1] or [2].
- Use only citation numbers that appear in the provided context.
- Do not invent policy details, timelines, approvals, or citations.
- Keep the answer concise, practical, and easy to read.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grounded answer from retrieved document chunks."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to answer from the indexed document set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks to use as context.",
    )
    parser.add_argument(
        "--model",
        default=ANSWER_MODEL,
        help="OpenAI model to use for answer generation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the answer payload as JSON.",
    )
    return parser.parse_args()


def build_context(chunks: list[dict]) -> str:
    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"source_file: {chunk['source_file']}",
                    f"document_title: {chunk['document_title']}",
                    f"page_number: {chunk['page_number']}",
                    f"chunk_id: {chunk['chunk_id']}",
                    "text:",
                    chunk["text"],
                ]
            )
        )

    return "\n\n".join(context_blocks)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    return f"""
Question:
{question}

Retrieved context:
{context}

Write a short answer grounded in the context above.
When you make a factual claim, cite it with bracketed references like [1] or [2].
End with a separate line that starts with "Citations:" and lists the references you used.
""".strip()


def extract_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    texts: list[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "").strip()
                if text:
                    texts.append(text)

    return "\n".join(texts).strip()


def build_sources(chunks: list[dict]) -> list[dict]:
    sources = []

    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "reference": index,
                "source_file": chunk["source_file"],
                "document_title": chunk["document_title"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
            }
        )

    return sources


def generate_answer(
    question: str, client: OpenAI, chunks: list[dict], model: str = ANSWER_MODEL
) -> str:
    prompt = build_user_prompt(question, chunks)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": DEVELOPER_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    answer_text = extract_output_text(response)
    if not answer_text:
        raise ValueError("The model returned an empty answer.")

    return answer_text


def answer_question(
    question: str, top_k: int = DEFAULT_TOP_K, model: str = ANSWER_MODEL
) -> dict:
    chunks = retrieve_chunks(question=question, top_k=top_k)
    if not chunks:
        raise ValueError("No retrieved chunks were returned for the question.")

    client = get_client()
    answer_text = generate_answer(question=question, client=client, chunks=chunks, model=model)

    return {
        "question": question,
        "model": model,
        "answer": answer_text,
        "sources": build_sources(chunks),
        "retrieved_chunks": chunks,
    }


def format_answer(payload: dict) -> str:
    lines = [f'Question: "{payload["question"]}"', "", "Answer:", payload["answer"], "", "Sources:"]

    for source in payload["sources"]:
        lines.append(
            f'[{source["reference"]}] {source["source_file"]} | '
            f'page {source["page_number"]} | {source["chunk_id"]}'
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()

    try:
        payload = answer_question(question=question, top_k=args.top_k, model=args.model)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_answer(payload))


if __name__ == "__main__":
    main()
