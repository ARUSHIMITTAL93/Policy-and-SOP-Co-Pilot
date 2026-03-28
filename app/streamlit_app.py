from __future__ import annotations

import html

import streamlit as st

from answer import ANSWER_MODEL, answer_question


st.set_page_config(page_title="Policy and SOP Copilot", layout="wide")

EXAMPLE_QUESTIONS = [
    "What is the escalation timeline for a high-risk issue?",
    "Who approves a permanent exception?",
    "What evidence is required before case closure?",
    "How long should high-risk remediation items take?",
]


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(239, 223, 196, 0.45), transparent 28%),
                radial-gradient(circle at top right, rgba(187, 216, 213, 0.35), transparent 30%),
                linear-gradient(180deg, #f8f5ef 0%, #eef3f1 100%);
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(247, 240, 227, 0.96), rgba(230, 240, 237, 0.94));
            border: 1px solid rgba(106, 134, 131, 0.18);
            box-shadow: 0 18px 40px rgba(49, 67, 65, 0.08);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #213231;
            margin-bottom: 0.35rem;
        }

        .hero-copy {
            font-size: 1rem;
            color: #47615e;
            line-height: 1.55;
            max-width: 48rem;
        }

        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6a8683;
            margin-bottom: 0.45rem;
        }

        .turn-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(90, 111, 108, 0.16);
            box-shadow: 0 12px 30px rgba(46, 61, 60, 0.07);
            margin-bottom: 0.9rem;
        }

        .user-question {
            font-size: 1rem;
            font-weight: 600;
            color: #243533;
            margin: 0;
        }

        .meta-chip {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            margin: 0.15rem 0.35rem 0.15rem 0;
            border-radius: 999px;
            background: #e3eeeb;
            color: #35524f;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "answer_error" not in st.session_state:
        st.session_state.answer_error = None
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "next_question_input" not in st.session_state:
        st.session_state.next_question_input = None


def apply_pending_question_update() -> None:
    if st.session_state.next_question_input is not None:
        st.session_state.question_input = st.session_state.next_question_input
        st.session_state.next_question_input = None


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Policy And SOP Copilot</div>
            <div class="hero-title">Policy and SOP Copilot</div>
            <div class="hero-copy">
                Ask grounded questions over the indexed PDF library. Each answer is built from retrieved
                policy and SOP chunks, and the supporting source passages appear directly underneath.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, str]:
    with st.sidebar:
        st.markdown("### Demo Controls")
        top_k = st.slider("Retrieved chunks", min_value=3, max_value=8, value=5, step=1)
        model = st.text_input("Answer model", value=ANSWER_MODEL)

        st.markdown("### Quick Fill")
        selected_example = st.selectbox("Example question", EXAMPLE_QUESTIONS)
        if st.button("Use Example", use_container_width=True):
            st.session_state.next_question_input = selected_example
            st.rerun()

        st.markdown("### Session")
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.answer_error = None

        st.caption("History stays in this browser session so you can compare answers side by side.")

    return top_k, model


def render_sources(retrieved_chunks: list[dict]) -> None:
    st.markdown('<div class="section-label">Retrieved Source Chunks</div>', unsafe_allow_html=True)

    for index, chunk in enumerate(retrieved_chunks, start=1):
        label = (
            f"[{index}] {chunk['source_file']} | page {chunk['page_number']} "
            f"| score {chunk['score']:.4f}"
        )
        with st.expander(label):
            st.caption(
                f"Document: {chunk['document_title']} | Chunk ID: {chunk['chunk_id']}"
            )
            st.write(chunk["text"])


def render_turn(turn: dict, turn_number: int) -> None:
    escaped_question = html.escape(turn["question"])
    st.markdown(
        f"""
        <div class="turn-card">
            <div class="section-label">Question {turn_number}</div>
            <p class="user-question">{escaped_question}</p>
            <span class="meta-chip">Model: {turn["model"]}</span>
            <span class="meta-chip">Chunks: {turn["top_k"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Final Answer")
    st.write(turn["answer"])
    render_sources(turn["retrieved_chunks"])


def submit_question(question: str, top_k: int, model: str) -> None:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.session_state.answer_error = "Please enter a question."
        return

    st.session_state.answer_error = None
    with st.spinner("Searching documents and drafting an answer..."):
        try:
            payload = answer_question(cleaned_question, top_k=top_k, model=model)
            payload["top_k"] = top_k
            st.session_state.history.insert(0, payload)
            st.session_state.next_question_input = ""
            st.rerun()
        except (FileNotFoundError, ValueError) as error:
            st.session_state.answer_error = str(error)
        except Exception as error:  # noqa: BLE001
            st.session_state.answer_error = f"Unexpected error: {error}"


def main() -> None:
    apply_styles()
    initialize_state()
    apply_pending_question_update()
    render_header()
    top_k, model = render_sidebar()

    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="What is the escalation timeline for a high-risk issue?",
            height=110,
            key="question_input",
        )
        submitted = st.form_submit_button("Ask")

    if submitted:
        submit_question(question, top_k=top_k, model=model)

    if st.session_state.answer_error:
        st.error(st.session_state.answer_error)

    if not st.session_state.history:
        st.info("Ask a question to generate your first grounded answer.")
        return

    for turn_number, turn in enumerate(st.session_state.history, start=1):
        render_turn(turn, turn_number)
        if turn_number != len(st.session_state.history):
            st.markdown("---")


if __name__ == "__main__":
    main()
