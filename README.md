# Policy and SOP Copilot

Policy and SOP Copilot is a lightweight retrieval-augmented generation (RAG) demo for asking grounded questions over synthetic policy and SOP PDF documents. It extracts text from PDFs, chunks the content for semantic retrieval, stores embeddings in FAISS, and serves answers through a Streamlit interface with source chunk citations. The repository is set up for local experimentation and GitHub publication without committing secrets or generated vector index files.

## Tech Stack

- Python
- Streamlit
- OpenAI API
- FAISS
- PyMuPDF
- NumPy and Pandas

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Create a local `.env` file with your OpenAI API key and local app credentials:

   ```env
   OPENAI_API_KEY=your_api_key_here
   APP_USERNAME=your_app_username
   APP_PASSWORD=your_app_password
   ```

   The login credentials stay local in `.env` and are not committed to GitHub.

4. Prepare the retrieval assets:

   ```bash
   python3 app/extract_pdfs.py
   python3 app/chunk_pdfs.py
   python3 app/build_index.py
   ```

## Run The Streamlit App

```bash
python3 -m streamlit run app/streamlit_app.py
```

The app opens with a sign-in screen before the upload and Q&A workflow is available.
