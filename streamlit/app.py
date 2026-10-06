"""Streamlit UI for the Enterprise Knowledge Assistant."""

import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL
API_BASE = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


def check_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def fetch_documents():
    """Fetch documents from API."""
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def upload_document(uploaded_file):
    """Upload a document to the API."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Upload failed")}
    except Exception as e:
        return {"error": str(e)}


def index_documents():
    """Trigger indexing of uploaded documents."""
    try:
        response = requests.post(f"{API_BASE}/documents/index", json={}, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Indexing failed")}
    except Exception as e:
        return {"error": str(e)}


def ask_question(query: str, top_k: int = 5):
    """Ask a question to the RAG system."""
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(f"{API_BASE}/chat", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Query failed")}
    except Exception as e:
        return {"error": str(e)}


# Check API health on startup
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = check_health()

# Sidebar
with st.sidebar:
    st.title("📚 Enterprise Knowledge Assistant")
    st.caption("Multi-Agent RAG System")

    # API Health
    if st.session_state.api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        if st.button("Retry Connection"):
            st.session_state.api_healthy = check_health()
            st.rerun()

    st.divider()

    # Document Upload
    st.subheader("📄 Document Management")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a PDF document to add to the knowledge base",
    )

    if uploaded_file:
        if st.button("Upload Document", use_container_width=True):
            with st.spinner("Uploading..."):
                result = upload_document(uploaded_file)
                if "error" in result:
                    st.error(f"Upload failed: {result['error']}")
                else:
                    st.success(f"Uploaded: {result['filename']} ({result.get('page_count', '?')} pages)")
                    st.rerun()

    st.divider()

    # Document List
    st.subheader("📋 Uploaded Documents")
    if st.button("Refresh Document List", use_container_width=True):
        st.session_state.documents = fetch_documents()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['filename']} ({doc.get('page_count', '?')} pages)"):
                st.write(f"**ID:** {doc['document_id']}")
                st.write(f"**Pages:** {doc.get('page_count', '?')}")
                st.write(f"**Size:** {doc.get('file_size_bytes', 0) / 1024:.1f} KB")
                st.write(f"**Indexed:** {'✅' if doc.get('indexed') else '❌'}")
    else:
        st.caption("No documents uploaded yet")

    st.divider()

    # Indexing
    st.subheader("🔍 Index Documents")
    if st.button("Index All Documents", use_container_width=True, type="primary"):
        with st.spinner("Indexing documents... This may take a moment."):
            result = index_documents()
            if "error" in result:
                st.error(f"Indexing failed: {result['error']}")
            else:
                st.success(f"Indexed {result['indexed_count']} documents, {result['total_chunks']} chunks created")
                st.session_state.index_ready = True
                st.rerun()

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5, help="Number of chunks to retrieve")

    st.divider()

    # Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Area
st.title("💬 Enterprise Knowledge Assistant")

# Check if index is ready for chat
if not check_health():
    st.warning("⚠️ API is not available. Please start the FastAPI server.")
    st.code("uvicorn app.main:app --reload", language="bash")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📎 Sources"):
                for i, citation in enumerate(message["citations"], 1):
                    section_str = f" ({citation.get('section')})" if citation.get("section") else ""
                    st.markdown(
                        f"**[{i}]** {citation['document_name']} — Page {citation['page_number']}{section_str}"
                    )
        if "retrieved_chunks_count" in message:
            st.caption(f"Retrieved {message['retrieved_chunks_count']} chunks • Grounding: {'✅ Passed' if message.get('grounding_passed') else '❌ Failed'}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(prompt, top_k=top_k)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {result['error']}",
            })
        else:
            st.markdown(result["answer"])

            # Display citations
            if result.get("citations"):
                with st.expander("📎 Sources"):
                    for i, citation in enumerate(result["citations"], 1):
                        section_str = f" ({citation.get('section')})" if citation.get("section") else ""
                        st.markdown(
                            f"**[{i}]** {citation['document_name']} — Page {citation['page_number']}{section_str}"
                        )

            # Display metadata
            st.caption(
                f"Retrieved {result.get('retrieved_chunks_count', 0)} chunks • "
                f"Grounding: {'✅ Passed' if result.get('grounding_passed') else '❌ Failed'} • "
                f"Retries: {result.get('retry_count', 0)}"
            )

            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result.get("citations", []),
                "retrieved_chunks_count": result.get("retrieved_chunks_count", 0),
                "grounding_passed": result.get("grounding_passed", False),
                "retry_count": result.get("retry_count", 0),
            })