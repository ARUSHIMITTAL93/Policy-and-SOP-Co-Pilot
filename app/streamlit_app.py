from __future__ import annotations

import hmac
import html
import os
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

from answer import ANSWER_MODEL, answer_question
from document_pipeline import (
    DocumentCorpus,
    build_corpus_from_chunks,
    create_chunks_from_document,
    extract_pdf_document,
)
from retrieve import get_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

EXAMPLE_QUESTIONS = [
    "What is the purpose of this document?",
    "What approvals or sign-offs are required?",
    "What deadlines or timelines are mentioned?",
    "What evidence is needed before closure?",
]


def configure_page() -> None:
    st.set_page_config(page_title="Policy and SOP Copilot", layout="wide")


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

        .hero-card, .upload-card, .turn-card, .login-card {
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(106, 134, 131, 0.18);
            box-shadow: 0 18px 40px rgba(49, 67, 65, 0.08);
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            background: linear-gradient(135deg, rgba(247, 240, 227, 0.96), rgba(230, 240, 237, 0.94));
            margin-bottom: 1.25rem;
        }

        .upload-card, .turn-card, .login-card {
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .login-card {
            margin-top: 4rem;
        }

        .hero-title, .login-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #213231;
            margin-bottom: 0.35rem;
        }

        .login-title {
            font-size: 1.8rem;
        }

        .hero-copy, .subtle-copy, .login-copy {
            font-size: 1rem;
            color: #47615e;
            line-height: 1.55;
            max-width: 50rem;
        }

        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6a8683;
            margin-bottom: 0.45rem;
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

        .file-chip {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            margin: 0.18rem 0.4rem 0.18rem 0;
            border-radius: 999px;
            background: #edf5f2;
            color: #31504d;
            font-size: 0.82rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "history": [],
        "answer_error": None,
        "question_input": "",
        "next_question_input": None,
        "active_corpus": None,
        "processing_error": None,
        "processing_success": None,
        "processing_warnings": [],
        "is_authenticated": False,
        "authenticated_username": None,
        "auth_error": None,
        "upload_widget_version": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_pending_question_update() -> None:
    if st.session_state.next_question_input is not None:
        st.session_state.question_input = st.session_state.next_question_input
        st.session_state.next_question_input = None


def clear_processing_messages() -> None:
    st.session_state.processing_error = None
    st.session_state.processing_success = None
    st.session_state.processing_warnings = []


def reset_app_workflow_state() -> None:
    st.session_state.history = []
    st.session_state.answer_error = None
    st.session_state.question_input = ""
    st.session_state.next_question_input = None
    st.session_state.active_corpus = None
    clear_processing_messages()
    st.session_state.upload_widget_version += 1


def logout_user() -> None:
    reset_app_workflow_state()
    st.session_state.is_authenticated = False
    st.session_state.authenticated_username = None
    st.session_state.auth_error = None


def load_auth_credentials() -> tuple[str, str]:
    load_dotenv(dotenv_path=ENV_PATH)
    username = os.getenv("APP_USERNAME", "").strip()
    password = os.getenv("APP_PASSWORD", "")

    if not username or not password:
        raise ValueError(
            "Configure APP_USERNAME and APP_PASSWORD in your local .env before launching the app."
        )

    return username, password


def credentials_are_valid(
    username: str,
    password: str,
    expected_username: str,
    expected_password: str,
) -> bool:
    return hmac.compare_digest(username.strip(), expected_username) and hmac.compare_digest(
        password,
        expected_password,
    )


def authenticate_user(username: str, password: str) -> bool:
    try:
        expected_username, expected_password = load_auth_credentials()
    except ValueError as error:
        st.session_state.auth_error = str(error)
        return False

    if not credentials_are_valid(username, password, expected_username, expected_password):
        st.session_state.auth_error = "Invalid username or password."
        return False

    st.session_state.is_authenticated = True
    st.session_state.authenticated_username = expected_username
    st.session_state.auth_error = None
    return True


def render_login_screen() -> None:
    outer_left, center, outer_right = st.columns([1, 1.15, 1])

    with center:
        st.markdown(
            """
            <div class="login-card">
                <div class="section-label">Secure Access</div>
                <div class="login-title">Sign in to Policy and SOP Copilot</div>
                <div class="login-copy">
                    Enter the local demo credentials to unlock document upload, retrieval, and grounded
                    question answering for this session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        config_error = None
        try:
            load_auth_credentials()
        except ValueError as error:
            config_error = str(error)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign In",
                type="primary",
                use_container_width=True,
                disabled=config_error is not None,
            )

        if config_error:
            st.error(config_error)
            return

        if st.session_state.auth_error:
            st.error(st.session_state.auth_error)

        if submitted and authenticate_user(username, password):
            st.rerun()


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Upload-First RAG Demo</div>
            <div class="hero-title">Policy and SOP Copilot</div>
            <div class="hero-copy">
                Upload one or more PDFs, process them into a temporary session index, and ask grounded
                questions against only those uploaded documents. Nothing from the upload flow is written
                into the project data folders.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(corpus: DocumentCorpus | None) -> tuple[int, str]:
    with st.sidebar:
        st.markdown("### Demo Controls")
        top_k = st.slider("Retrieved chunks", min_value=3, max_value=8, value=5, step=1)
        model = st.text_input("Answer model", value=ANSWER_MODEL)

        st.markdown("### Session Corpus")
        if corpus is None:
            st.info("Upload PDFs and click Process Documents to activate a session corpus.")
        else:
            st.success(
                f"Active corpus: {corpus.document_count} PDF(s) and {corpus.chunk_count} chunks"
            )

        st.markdown("### Quick Fill")
        selected_example = st.selectbox("Example question", EXAMPLE_QUESTIONS)
        st.caption("These example prompts run only against the PDFs you process in this session.")
        if st.button("Use Example", use_container_width=True):
            st.session_state.next_question_input = selected_example
            st.rerun()

        st.markdown("### Session")
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.answer_error = None

        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()

        st.caption(
            "A newly processed upload batch replaces the current session corpus and clears prior answers."
        )

    return top_k, model


def build_skip_message(filename: str, reason: str) -> str:
    return f"{filename}: {reason}"


def validate_uploaded_files(uploaded_files: list[UploadedFile] | None) -> None:
    if not uploaded_files:
        raise ValueError("Upload at least one PDF before processing.")

    invalid_files = [
        uploaded_file.name
        for uploaded_file in uploaded_files
        if not uploaded_file.name.lower().endswith(".pdf")
    ]
    if invalid_files:
        joined_files = ", ".join(sorted(invalid_files))
        raise ValueError(f"Only PDF uploads are supported right now. Remove: {joined_files}")

    duplicates = sorted(
        file_name
        for file_name, count in Counter(uploaded_file.name for uploaded_file in uploaded_files).items()
        if count > 1
    )
    if duplicates:
        joined_duplicates = ", ".join(duplicates)
        raise ValueError(
            "Duplicate filenames were uploaded in the same batch. "
            f"Please rename or remove: {joined_duplicates}"
        )


def build_uploaded_corpus(
    uploaded_files: list[UploadedFile] | None,
) -> tuple[DocumentCorpus, list[str]]:
    validate_uploaded_files(uploaded_files)
    assert uploaded_files is not None

    chunks: list[dict] = []
    warnings: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            document = extract_pdf_document(uploaded_file.name, uploaded_file.getvalue())
        except Exception as error:  # noqa: BLE001
            warnings.append(
                build_skip_message(uploaded_file.name, f"could not be read as a PDF ({error})")
            )
            continue

        document_chunks = create_chunks_from_document(document)
        if not document_chunks:
            warnings.append(build_skip_message(uploaded_file.name, "no extractable text was found"))
            continue

        chunks.extend(document_chunks)

    if not chunks:
        raise ValueError(
            "None of the uploaded PDFs produced searchable text, so the existing session corpus was kept."
        )

    client = get_client()
    corpus = build_corpus_from_chunks(chunks, client, show_progress=False)
    return corpus, warnings


def process_uploaded_files(uploaded_files: list[UploadedFile] | None) -> None:
    clear_processing_messages()

    try:
        corpus, warnings = build_uploaded_corpus(uploaded_files)
    except (FileNotFoundError, ValueError) as error:
        st.session_state.processing_error = str(error)
        return
    except Exception as error:  # noqa: BLE001
        st.session_state.processing_error = (
            "Could not process the uploaded PDFs. The previous session corpus is still active. "
            f"Details: {error}"
        )
        return

    st.session_state.active_corpus = corpus
    st.session_state.history = []
    st.session_state.answer_error = None
    st.session_state.next_question_input = ""
    st.session_state.processing_warnings = warnings
    st.session_state.processing_success = (
        f"Processed {corpus.document_count} PDF(s) into {corpus.chunk_count} chunks."
    )
    st.session_state.upload_widget_version += 1
    st.rerun()


def render_upload_section() -> None:
    st.markdown(
        """
        <div class="upload-card">
            <div class="section-label">1. Upload And Process</div>
            <div class="subtle-copy">
                Upload one or more PDFs, then build a temporary in-memory index for this browser session.
                The app will answer only from the active uploaded files.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Only PDF files are supported in this version. A new processed batch replaces the active corpus.",
        key=f"uploaded_pdfs_{st.session_state.upload_widget_version}",
    )

    if uploaded_files:
        st.markdown("**Selected files**")
        st.write(", ".join(uploaded_file.name for uploaded_file in uploaded_files))

    if st.button("Process Documents", type="primary", use_container_width=True):
        with st.spinner("Extracting text, chunking pages, and building the temporary index..."):
            process_uploaded_files(uploaded_files)

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)

    if st.session_state.processing_success:
        st.success(st.session_state.processing_success)

    for warning in st.session_state.processing_warnings:
        st.warning(warning)


def render_active_corpus(corpus: DocumentCorpus | None) -> None:
    if corpus is None:
        st.info("Process at least one uploaded PDF to unlock question answering.")
        return

    st.markdown('<div class="section-label">Active Session Corpus</div>', unsafe_allow_html=True)
    st.caption("Questions will be answered only from these uploaded PDFs.")
    file_chips = "".join(
        f'<span class="file-chip">{html.escape(document_name)}</span>'
        for document_name in corpus.document_names
    )
    st.markdown(file_chips, unsafe_allow_html=True)


def render_sources(retrieved_chunks: list[dict]) -> None:
    st.markdown('<div class="section-label">Retrieved Source Chunks</div>', unsafe_allow_html=True)

    for index, chunk in enumerate(retrieved_chunks, start=1):
        label = (
            f"[{index}] {chunk['source_file']} | page {chunk['page_number']} "
            f"| score {chunk['score']:.4f}"
        )
        with st.expander(label):
            st.caption(f"Document: {chunk['document_title']} | Chunk ID: {chunk['chunk_id']}")
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


def submit_question(
    question: str,
    top_k: int,
    model: str,
    corpus: DocumentCorpus | None,
) -> None:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.session_state.answer_error = "Please enter a question."
        return

    if corpus is None:
        st.session_state.answer_error = "Upload and process PDFs before asking a question."
        return

    st.session_state.answer_error = None
    with st.spinner("Searching the uploaded PDFs and drafting an answer..."):
        try:
            payload = answer_question(
                cleaned_question,
                top_k=top_k,
                model=model,
                corpus=corpus,
            )
            payload["top_k"] = top_k
            st.session_state.history.insert(0, payload)
            st.session_state.next_question_input = ""
            st.rerun()
        except (FileNotFoundError, ValueError) as error:
            st.session_state.answer_error = str(error)
        except Exception as error:  # noqa: BLE001
            st.session_state.answer_error = f"Unexpected error: {error}"


def render_authenticated_app() -> None:
    corpus = st.session_state.active_corpus

    render_header()
    top_k, model = render_sidebar(corpus)
    render_upload_section()
    render_active_corpus(corpus)

    can_ask = corpus is not None
    placeholder = (
        "Ask a question about the uploaded PDFs."
        if can_ask
        else "Upload and process PDFs first."
    )

    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder=placeholder,
            height=110,
            key="question_input",
            disabled=not can_ask,
        )
        submitted = st.form_submit_button("Ask", disabled=not can_ask)

    if submitted:
        submit_question(question, top_k=top_k, model=model, corpus=corpus)

    if st.session_state.answer_error:
        st.error(st.session_state.answer_error)

    if not st.session_state.history:
        if can_ask:
            st.info("Ask a question to generate your first grounded answer from the uploaded PDFs.")
        return

    for turn_number, turn in enumerate(st.session_state.history, start=1):
        render_turn(turn, turn_number)
        if turn_number != len(st.session_state.history):
            st.markdown("---")


def main() -> None:
    configure_page()
    apply_styles()
    initialize_state()
    apply_pending_question_update()

    if not st.session_state.is_authenticated:
        render_login_screen()
        return

    render_authenticated_app()


if __name__ == "__main__":
    main()
