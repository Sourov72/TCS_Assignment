"""
Streamlit chat UI for the customer support multi-agent assistant.

- Chats over MCP (agents/mcp_client.py -> mcp_server/server.py's
  ask_support_assistant tool), rather than calling the ReAct agent
  (agents/graph.py) directly in-process. The MCP server auto-starts in the
  background the first time it's needed, over HTTP transport (not stdio,
  since stdio ties a subprocess's lifetime to one client session, which
  doesn't fit Streamlit's rerun-per-interaction execution model).
- Lets a user upload a new policy PDF and have it chunked + embedded into
  the shared policy knowledge base right away (see rag/ingest.py's
  add_pdf_to_index()) - no category to pick, it's just one combined,
  searchable index.

Run with:
    venv\\Scripts\\python.exe -m streamlit run app.py
"""

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "agents"))
sys.path.insert(0, str(Path(__file__).parent / "common"))
sys.path.insert(0, str(Path(__file__).parent / "rag"))

from config import DEEPSEEK_API_KEY  # noqa: E402
from ingest import add_pdf_to_index, is_duplicate_pdf  # noqa: E402
from mcp_client import ask as ask_assistant  # noqa: E402
from mcp_client import ensure_server_running  # noqa: E402

POLICIES_DIR = Path(__file__).parent / "data" / "policies"

st.set_page_config(page_title="Customer Support Copilot", page_icon="🎧")


def _ensure_session_id() -> str:
    """Give each browser session its own stable thread_id, so the LangGraph
    checkpointer remembers that session's conversation across reruns."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def _ensure_chat_history() -> list:
    """Local (role, content) list purely for redrawing the chat UI - the
    actual conversation memory used for answering lives in the graph's
    checkpointer, keyed by session_id."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history


def _render_chat_history(history: list) -> None:
    """Redraw all prior messages at the top of the page on every rerun."""
    for role, content in history:
        with st.chat_message(role):
            st.markdown(content)


def _handle_pdf_upload() -> None:
    """Sidebar widget: upload a policy PDF and merge it into the shared
    FAISS index - no category to choose, it's one combined knowledge base."""
    st.sidebar.subheader("Upload a policy PDF")
    uploaded_file = st.sidebar.file_uploader("PDF file", type=["pdf"])

    if uploaded_file is not None and st.sidebar.button("Ingest PDF"):
        file_bytes = uploaded_file.getvalue()

        # Check by content (not filename) before writing/ingesting anything,
        # so re-uploading the same PDF never duplicates its chunks.
        if is_duplicate_pdf(file_bytes):
            st.sidebar.info("This PDF is already in the knowledge base.")
            return

        POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        # Prefix with a short random id so re-uploading a same-named file
        # never silently overwrites a previously ingested one.
        safe_name = f"{uuid.uuid4().hex[:8]}_{Path(uploaded_file.name).name}"
        destination = POLICIES_DIR / safe_name
        destination.write_bytes(file_bytes)

        with st.sidebar.spinner("Chunking and embedding..."):
            add_pdf_to_index(destination)

        st.sidebar.success("Upload successful.")


def main() -> None:
    st.title("🎧 Customer Support Copilot")
    st.caption(
        "Ask about a customer's profile/ticket history, or about our "
        "company policies."
    )

    if not DEEPSEEK_API_KEY:
        st.error("DEEPSEEK_API_KEY is not set. Add it to .env and restart the app.")
        st.stop()

    # Start the MCP server once per session, up front, so the first chat
    # message doesn't silently eat the ~few-second startup cost.
    if "mcp_server_ready" not in st.session_state:
        with st.spinner("Starting assistant server..."):
            ensure_server_running()
        st.session_state.mcp_server_ready = True

    _handle_pdf_upload()

    session_id = _ensure_session_id()
    history = _ensure_chat_history()
    _render_chat_history(history)

    question = st.chat_input("Ask a question...")
    if question:
        history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_assistant(question, thread_id=session_id)
            st.markdown(answer)
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
