import os
import tempfile
import traceback
from dotenv import load_dotenv
import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""\n<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#080b14; --panel:#101626; --panel2:#141c2e; --border:#252e45; --text:#e9edf8; --muted:#8c99b5; --purple:#8b5cf6; --cyan:#22d3ee; }
.stApp { background:radial-gradient(ellipse at 15% 0%, #17142d 0, transparent 38%),var(--bg); color:var(--text); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; }
[data-testid="stSidebar"] { background:#0b101c; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--text); }
.block-container { padding-top:1.8rem; padding-bottom:3rem; max-width:1500px; }
.hero { padding:.2rem 0 1.2rem; }
.eyebrow { color:#b8a4ff; text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; }
.hero h1 { font-size:clamp(2rem,4vw,3.1rem); font-weight:700; letter-spacing:-.04em; margin:0; }
.hero p { color:var(--muted); margin-top:.45rem; font-size:1rem; }
.metric { background:linear-gradient(145deg,#151d31,#101625); border:1px solid var(--border); border-radius:14px; padding:1rem; }
.metric-value { font-size:1.35rem; font-weight:700; color:#d9ceff; }
.metric-label { color:var(--muted); font-size:.78rem; }
.stButton button { border-radius:10px !important; border:1px solid #6746c7 !important; background:linear-gradient(115deg,#7c3aed,#536dfe) !important; color:white !important; font-weight:700 !important; min-height:2.65rem; }
.stButton button:hover { border-color:#a78bfa !important; filter:brightness(1.08); }
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea { background:#111a2b !important; color:var(--text) !important; border-color:var(--border) !important; border-radius:10px !important; }
[data-testid="stFileUploader"] { background:#111a2b; border:1px dashed #35415e; border-radius:12px; padding:.55rem; }
[data-testid="stChatMessage"] { background:rgba(19,26,43,.72); border:1px solid var(--border); border-radius:14px; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {},
    "chat_input_key": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

PIPELINE = [
    ("audio", "Audio processing", "🔊"),
    ("transcript", "Transcription", "📝"),
    ("title", "Title generation", "🏷️"),
    ("summary", "Summarization", "📋"),
    ("extract", "Insights extraction", "🔎"),
    ("rag", "RAG indexing", "🧠"),
]

def status_mark(status):
    return {"done":"✅", "active":"⏳", "error":"❌"}.get(status, "▫️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:2.2rem;color:#a78bfa;font-weight:800'>▶</div><div style='font-size:1.45rem;font-weight:800'>AI Video Assistant</div><div style='color:#8c99b5'>Transcribe · Summarize · Ask</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### New analysis")
    source_text = st.text_input("YouTube URL or local file path", placeholder="Paste video URL or path", key="source_text")
    uploaded_file = st.file_uploader("Or upload media", type=["mp4", "mov", "mkv", "mp3", "wav", "m4a"])
    language = st.selectbox("Transcript language", ["english", "hinglish"], index=0)
    analyze_button = st.button("✨  Analyze video", use_container_width=True)
    st.divider()
    st.markdown("### Processing pipeline")
    for key, label, emoji in PIPELINE:
        current = st.session_state.pipeline_steps.get(key, "pending")
        st.markdown(f"{status_mark(current)} {emoji} **{label}**")
    if st.session_state.result:
        st.divider()
        if st.button("＋ Start new analysis", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.rerun()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""\n<div class="hero">
  <div class="eyebrow">Video intelligence workspace</div>
  <h1>Turn videos into <span style="color:#a78bfa">useful insights.</span></h1>
  <p>Transcribe, summarize, extract action items, and chat with your video content.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.result:
    intro_cols = st.columns(3)
    intro_data = [("01", "Transcribe", "Convert video audio to text"), ("02", "Summarize", "Get the key takeaways"), ("03", "Ask questions", "Search your video with RAG")]
    for col, (num, label, desc) in zip(intro_cols, intro_data):
        with col:
            st.markdown(f"<div class='metric'><div class='metric-value'>{num} · {label}</div><div class='metric-label'>{desc}</div></div>", unsafe_allow_html=True)

# ── Pipeline execution ────────────────────────────────────────────────────────
if analyze_button:
    if not source_text.strip() and uploaded_file is None:
        st.sidebar.warning("Enter a YouTube URL/file path or upload a media file.")
    else:
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {k: "pending" for k, _, _ in PIPELINE}
        temp_path = None
        source_to_process = source_text.strip()

        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            source_to_process = temp_path

        pipeline_box = st.status("Preparing analysis…", expanded=True)
        try:
            def run_step(key, label, fn):
                st.session_state.pipeline_steps[key] = "active"
                pipeline_box.write(f"⏳ {label}")
                value = fn()
                st.session_state.pipeline_steps[key] = "done"
                pipeline_box.write(f"✅ {label} complete")
                return value

            audio_chunks = run_step("audio", "Processing audio", lambda: process_input(source_to_process))
            transcript = run_step("transcript", "Transcribing audio", lambda: transcribe_all(audio_chunks, language))
            video_title = run_step("title", "Generating title", lambda: generate_title(transcript))
            video_summary = run_step("summary", "Summarizing", lambda: summarize(transcript))

            def extract_insights():
                return {
                    "action_items": extract_action_items(transcript),
                    "key_decisions": extract_key_decisions(transcript),
                    "open_questions": extract_questions(transcript),
                }
            insights = run_step("extract", "Extracting insights", extract_insights)
            rag = run_step("rag", "Building RAG index", lambda: build_rag_chain(transcript))

            st.session_state.result = {
                "title": video_title,
                "transcript": transcript,
                "summary": video_summary,
                **insights,
                "rag_chain": rag,
                "source": uploaded_file.name if uploaded_file else source_text.strip(),
            }
            pipeline_box.update(label="Analysis finished successfully", state="complete", expanded=False)
            st.rerun()
        except Exception as error:
            for key, _, _ in PIPELINE:
                if st.session_state.pipeline_steps.get(key) == "active":
                    st.session_state.pipeline_steps[key] = "error"
            pipeline_box.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Analysis failed: {error}")
            st.code(traceback.format_exc())
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

# ── Results ───────────────────────────────────────────────────────────────────
r = st.session_state.result
if r:
    st.markdown(f"## {r.get('title') or 'Video analysis'}")
    st.caption(f"Source: {r.get('source', 'Video')}")
    transcript = str(r.get("transcript", ""))
    word_count = len(transcript.split())

    metric_cols = st.columns(3)
    metric_values = [(f"📝 {word_count:,}", "Transcript words"), ("🧠 4", "Retrieved chunks per question"), ("✨ 3", "Insight categories")]
    for col, (value, label) in zip(metric_cols, metric_values):
        with col:
            st.markdown(f"<div class='metric'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("## Session insights")
    summary_col, action_col = st.columns([1.2, 1])
    with summary_col:
        with st.container(border=True):
            st.markdown("### 📋 Summary")
            st.markdown(r.get("summary") or "No summary available.")
    with action_col:
        with st.container(border=True):
            st.markdown("### ✅ Action items")
            st.markdown(r.get("action_items") or "No action items found.")

    decision_col, question_col = st.columns(2)
    with decision_col:
        with st.container(border=True):
            st.markdown("### 🔑 Key decisions")
            st.markdown(r.get("key_decisions") or "No key decisions found.")
    with question_col:
        with st.container(border=True):
            st.markdown("### ❓ Open questions")
            st.markdown(r.get("open_questions") or "No open questions found.")

    with st.expander("📄 View full transcript"):
        st.text(transcript)

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button("⬇️ Download transcript", transcript, file_name="transcript.txt", mime="text/plain")
    with download_cols[1]:
        markdown_report = f"# {r.get('title', 'Video analysis')}\n\n## Summary\n{r.get('summary', '')}\n\n## Action items\n{r.get('action_items', '')}\n\n## Key decisions\n{r.get('key_decisions', '')}\n\n## Open questions\n{r.get('open_questions', '')}\n\n## Transcript\n{transcript}"
        st.download_button("⬇️ Download analysis report", markdown_report, file_name="video-analysis.md", mime="text/markdown")

    st.divider()
    st.markdown("## 💬 Chat with your video")
    st.caption("Answers are grounded in transcript passages retrieved by your RAG pipeline.")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about this video…", key=f"chat_input_{st.session_state.chat_input_key}")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript…"):
                answer = ask_question(r["rag_chain"], question)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.chat_input_key += 1
        st.rerun()
else:
    st.markdown("""\n    <div class="metric" style="text-align:center;padding:2.5rem;margin-top:1rem">
      <div style="font-size:3rem">🎬</div>
      <h2>Workspace ready</h2>
      <p style="color:#8c99b5">Paste a YouTube link or upload a media file from the sidebar to begin your analysis.</p>
    </div>
    """, unsafe_allow_html=True)
