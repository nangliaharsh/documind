import streamlit as st
import requests
import os
API_URL = os.getenv("API_URL", "https://documind-backend-production-3be0.up.railway.app")
# API_URL = "http://localhost:8000"

# Keep backend alive
try:
    requests.get(f"{API_URL}/health", timeout=5)
except:
    pass


st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1c1d21; border-right: 1px solid #E2E8F0; }
    .main-header {
        background: #1A3C5E; color: white;
        padding: 1rem 1.5rem; border-radius: 10px;
        margin-bottom: 1rem;
    }
    .source-tag {
        background: #EBF4FF; color: #1A56DB;
        font-size: 11px; padding: 2px 8px;
        border-radius: 6px; margin-right: 4px;
        display: inline-block; margin-top: 4px;
    }
    .stat-card {
        background: white; border-radius: 8px;
        padding: 10px 14px; text-align: center;
        border: 1px solid #E2E8F0;
    }
    .doc-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 8px; padding: 8px 12px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Fetch existing docs from backend ──────────────────
def fetch_docs():
    try:
        res = requests.get(f"{API_URL}/documents")
        return res.json().get("documents", [])
    except:
        return []

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:1.5rem'>
        <div style='background:#1A3C5E;color:white;width:32px;height:32px;
                    border-radius:8px;display:flex;align-items:center;
                    justify-content:center;font-weight:700;font-size:16px'>D</div>
        <div>
            <div style='font-weight:600;font-size:15px;color:#1A3C5E'>DocuMind</div>
            <div style='font-size:11px;color:#888'>Multi-Doc RAG AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📄 Upload Document")
    st.caption("Supports PDF, DOCX, TXT")
    uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt"],
    label_visibility="collapsed"
)

    if uploaded_file:
        if uploaded_file.name not in st.session_state.get("uploaded_names", []):
            with st.spinner("Processing..."):
                res = requests.post(f"{API_URL}/upload", files={"file": uploaded_file})
                if res.status_code == 200:
                    data = res.json()
                    if "uploaded_names" not in st.session_state:
                        st.session_state["uploaded_names"] = []
                    st.session_state["uploaded_names"].append(uploaded_file.name)
                    st.success(f"✅ {data['chunks_stored']} chunks indexed")
                    st.rerun()

    # Document list
    docs = fetch_docs()
    if docs:
        st.markdown("---")
        st.markdown("#### 📚 Uploaded Documents")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class='doc-card'>
                    <div style='font-size:12px;font-weight:600;color:#1A3C5E'>📄 {doc[:25]}{"..." if len(doc)>25 else ""}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑", key=f"del_{doc}", help=f"Remove {doc}"):
                    requests.delete(f"{API_URL}/documents/{doc}")
                    st.rerun()

            # Show summary expander under each doc
            with st.expander("📋 Summary", expanded=False):
                try:
                    res = requests.get(f"{API_URL}/summary/{doc}")
                    if res.status_code == 200:
                        summary = res.json().get("summary", "No summary available.")
                    else:
                        summary = "Summary not available."
                except:
                    summary = "Could not fetch summary."
                st.caption(summary)

        st.markdown("---")
        st.markdown("#### 🔍 Query Scope")
        scope = st.radio("Search in:", ["All Documents", "Select Specific"], index=0, label_visibility="collapsed")
        selected_docs = None
        if scope == "Select Specific":
            selected_docs = st.multiselect("Choose documents:", docs, default=docs[:1])

        st.session_state["selected_docs"] = selected_docs

    # Stats
    if docs:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class='stat-card'>
                <div style='font-size:11px;color:#888'>Docs</div>
                <div style='font-size:22px;font-weight:600;color:#1A3C5E'>{len(docs)}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            msgs = len(st.session_state.get("messages", []))
            st.markdown(f"""<div class='stat-card'>
                <div style='font-size:11px;color:#888'>Messages</div>
                <div style='font-size:22px;font-weight:600;color:#1A3C5E'>{msgs}</div>
            </div>""", unsafe_allow_html=True)

# ── Main Area ───────────────────────────────────────────
docs = fetch_docs()

if not docs:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem'>
        <div style='font-size:52px;margin-bottom:1rem'>🧠</div>
        <h2 style='color:#1A3C5E;margin-bottom:0.5rem'>Welcome to DocuMind</h2>
        <p style='color:#666;font-size:15px'>Upload one or more PDFs from the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)
else:
    scope_label = "All Documents" if not st.session_state.get("selected_docs") else ", ".join(st.session_state["selected_docs"])
    st.markdown(f"""
    <div class='main-header'>
        <div style='display:flex;align-items:center;justify-content:space-between'>
            <div>
                <span style='font-size:16px;font-weight:600'>🧠 DocuMind</span>
                <span style='background:#0077B6;font-size:11px;padding:2px 8px;
                            border-radius:6px;margin-left:10px'>✓ {len(docs)} doc{"s" if len(docs)>1 else ""} loaded</span>
            </div>
            <span style='font-size:12px;opacity:0.8'>Scope: {scope_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
            # Show reformulation notice if present
            if msg["role"] == "assistant" and msg.get("reformulation", {}).get("was_reformulated"):
                ref = msg["reformulation"]
                st.markdown(f"""
                <div style='background:#EBF4FF;border:1px solid #BFD9FF;border-radius:8px;
                            padding:8px 12px;margin-bottom:10px;font-size:11px;color:#1A56DB'>
                    🔄 <b>Query reformulated</b><br>
                    <span style='color:#555'>"{ref.get("original")}"</span>
                    → <span style='color:#1A3C5E;font-weight:600'>"{ref.get("reformulated")}"</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                source_html = "".join([f"<span class='source-tag'>📄 {s}</span>" for s in msg["sources"]])
                st.markdown(f"<div style='margin-top:8px'>{source_html}</div>", unsafe_allow_html=True)

            # Explainability panel
            if msg["role"] == "assistant" and msg.get("explainability"):
                with st.expander("🔍 How did I find this?", expanded=False):
                    for item in msg["explainability"]:
                        score = item["relevance_score"]
                        verified = item.get("verified", True)
                        reason = item.get("reason", "")
                        bar_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
                        verified_badge = "✅ Verified" if verified else "⚠️ Unverified"
                        badge_color = "#22c55e" if verified else "#ef4444"

                        st.markdown(f"""
                        <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                                    padding:10px 14px;margin-bottom:8px'>
                            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                                <span style='font-size:11px;font-weight:600;color:#1A3C5E'>
                                    📄 {item["source"]} — Chunk {item["chunk_index"]}
                                </span>
                                <div style='display:flex;gap:8px;align-items:center'>
                                    <span style='font-size:11px;font-weight:700;color:{badge_color}'>
                                        {verified_badge}
                                    </span>
                                    <span style='font-size:11px;font-weight:700;color:{bar_color}'>
                                        {score}%
                                    </span>
                                </div>
                            </div>
                            <div style='background:#E2E8F0;border-radius:4px;height:6px;margin-bottom:8px'>
                                <div style='background:{bar_color};width:{score}%;height:6px;border-radius:4px'></div>
                            </div>
                            <div style='font-size:11px;color:#555;line-height:1.5;margin-bottom:6px'>
                                {item["chunk_preview"]}
                            </div>
                            <div style='font-size:10px;color:{badge_color};font-style:italic'>
                                {reason}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    if question := st.chat_input("Ask anything across your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching documents..."):
                try:
                    res = requests.post(f"{API_URL}/query", json={
                        "question": question,
                        "doc_ids": st.session_state.get("selected_docs")
                    })
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        explainability = data.get("explainability", [])
                        citation_results = data.get("citation_results", [])
                        reformulation = data.get("reformulation", {})
                    else:
                        answer = f"Backend error {res.status_code}: {res.text}"
                        sources = []
                except Exception as e:
                    answer = f"Connection error: {str(e)}"
                    sources = []

                st.markdown(answer)
                if sources:
                    source_html = "".join([f"<span class='source-tag'>✅ {s}</span>" for s in sources])
                    st.markdown(f"<div style='margin-top:8px'>{source_html}</div>", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "explainability": explainability,
                    "citation_results": citation_results,
                    "reformulation": reformulation
                })