import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Smart RAG System", page_icon="🤖", layout="wide")

# --- Custom CSS for ChatGPT-like UI ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #23234a 0%, #6b2e5c 100%) !important;
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }
    .main {
        background: transparent !important;
    }
    .chat-bubble {
        max-width: 80%%;
        padding: 1em 1.5em;
        margin-bottom: 0.5em;
        border-radius: 1.5em;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        font-size: 1.08em;
        word-break: break-word;
        white-space: pre-wrap;
    }
    .bubble-user {
        background: #0b93f6;
        color: #fff;
        border-bottom-right-radius: 0.3em !important;
        margin-left: auto;
    }
    .bubble-assistant {
        background: rgba(255,255,255,0.07);
        color: #fff;
        border-bottom-left-radius: 0.3em !important;
        margin-right: auto;
    }
    .sidebar-btn {
        width: 48px;
        height: 48px;
        margin: 0.25em 0;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        border: none;
        color: #fff;
        font-size: 1.5em;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s;
        cursor: pointer;
        position: relative;
    }
    .sidebar-btn:hover {
        background: #4F8BF9;
    }
    .sidebar-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        left: 60px;
        top: 50%%;
        transform: translateY(-50%%);
        background: #23234a;
        color: #fff;
        padding: 0.25em 0.8em;
        border-radius: 8px;
        font-size: 0.95em;
        white-space: nowrap;
        z-index: 10;
        transition: opacity 0.15s;
    }
    .sidebar-btn:hover .sidebar-tooltip {
        visibility: visible;
        opacity: 1;
    }
    .sidebar {
        background: #0b1220;
        min-width: 70px;
        max-width: 70px;
        height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1em 0.5em 0.5em 0.5em;
        box-shadow: 2px 0 16px 0 rgba(0,0,0,0.10);
    }
    .sidebar-bottom {
        margin-top: auto;
        margin-bottom: 1em;
        display: flex;
        flex-direction: column;
        gap: 0.5em;
    }
    .chat-header {
        font-size: 1.3em;
        font-weight: bold;
        color: #fff;
        padding: 1.1em 0 0.5em 0;
        text-align: left;
    }
    .chat-subheader {
        font-size: 1em;
        color: #e0e0e0;
        margin-bottom: 1.5em;
        text-align: left;
    }
    .input-bar {
        background: rgba(30,30,40,0.85);
        border-radius: 1.5em;
        box-shadow: 0 2px 12px rgba(0,0,0,0.10);
        padding: 0.5em 1.5em;
        display: flex;
        align-items: center;
        margin-top: 1em;
    }
    .input-bar textarea {
        background: transparent;
        border: none;
        color: #fff;
        font-size: 1.1em;
        width: 100%%;
        outline: none;
        resize: none;
        min-height: 36px;
        max-height: 120px;
    }
    .input-bar button {
        background: #0b93f6;
        color: #fff;
        border: none;
        border-radius: 1em;
        padding: 0.5em 1.2em;
        margin-left: 0.7em;
        font-size: 1em;
        cursor: pointer;
        transition: background 0.15s;
    }
    .input-bar button:hover {
        background: #4F8BF9;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar and state ---
sidebar_icons = [
    ("💬", "Chat", "Query Assistant"),
    ("📤", "Upload", "Upload Document"),
    ("🔄", "Convert", "Convert Files"),
    ("📑", "Compare", "Compare Documents"),
    ("🎤", "Audio", "Audio Transcription"),
    ("📚", "Docs", "Docs"),
]

if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = "Query Assistant"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi — I'm your assistant. Ask me anything!"}
    ]
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --- Layout ---
cols = st.columns([1, 8], gap="small")

# --- Sidebar ---
with cols[0]:
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    for icon, tooltip, feature in sidebar_icons:
        btn = st.button(
            f"{icon}",
            key=f"sidebar_{feature}",
            help=tooltip,
            use_container_width=True
        )
        if btn:
            st.session_state.selected_feature = feature
    st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)
    if st.button("🌙" if st.session_state.theme == "dark" else "☀️", key="theme_toggle", help="Toggle theme", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    if st.button("🗑️", key="clear_chat", help="Clear chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "text": "Hi — I'm your assistant. Ask me anything!"}
        ]
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main area ---
with cols[1]:
    st.markdown('<div style="max-width:700px;margin:auto;">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">Smart RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-subheader">Full-screen ChatGPT-like UI</div>', unsafe_allow_html=True)

    # --- Chat/Q&A ---
    if st.session_state.selected_feature == "Query Assistant":
        for m in st.session_state.messages:
            bubble_class = "bubble-user" if m["role"] == "user" else "bubble-assistant"
            st.markdown(
                f'<div class="chat-bubble {bubble_class}">{m["text"]}</div>',
                unsafe_allow_html=True,
            )
        with st.form("chat_input", clear_on_submit=True):
            user_input = st.text_area(
                "Ask your question",
                placeholder="Type a message. Press Ctrl+Enter or click Send.",
                label_visibility="collapsed",
                key="chat_input_box"
            )
            send = st.form_submit_button("Send")
            if send and user_input.strip():
                st.session_state.messages.append({"role": "user", "text": user_input})
                try:
                    response = requests.post(f"{API_URL}/query/", json={"query": user_input})
                    if response.ok:
                        answer = response.json().get("answer", "No answer returned.")
                        st.session_state.messages.append({"role": "assistant", "text": answer})
                    else:
                        st.session_state.messages.append({"role": "assistant", "text": "Error: " + response.text})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {e}"})

    # --- Upload Document ---
    elif st.session_state.selected_feature == "Upload Document":
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader("Choose a file to upload")
        if uploaded_file and st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(f"{API_URL}/upload/", files=files)
                if response.ok:
                    st.success("File uploaded successfully!")
                else:
                    st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # --- Convert Files ---
    elif st.session_state.selected_feature == "Convert Files":
        st.header("🔄 Convert Files")
        uploaded_file = st.file_uploader("Choose a file to convert")
        output_format = st.selectbox("Convert to format", ["pdf", "docx", "jpg", "xlsx"])
        if uploaded_file and st.button("Convert"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"output_format": output_format}
            try:
                response = requests.post(f"{API_URL}/convert/", files=files, data=data)
                if response.ok:
                    out_name = response.headers.get("content-disposition", "").split("filename=")[-1]
                    st.success("Conversion successful!")
                    st.download_button("Download Converted File", response.content, file_name=out_name)
                else:
                    st.error(f"Conversion failed: {response.text}")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    # --- Compare Documents ---
    elif st.session_state.selected_feature == "Compare Documents":
        st.header("📑 Compare Documents")
        file1 = st.file_uploader("Upload first document", key="file1")
        file2 = st.file_uploader("Upload second document", key="file2")
        if file1 and file2 and st.button("Compare"):
            files = {
                "file1": (file1.name, file1, file1.type),
                "file2": (file2.name, file2, file2.type)
            }
            try:
                response = requests.post(f"{API_URL}/compare/", files=files)
                if response.ok:
                    st.success(response.json().get("result", "No result returned."))
                else:
                    st.error(f"Comparison failed: {response.text}")
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    # --- Audio Transcription ---
    elif st.session_state.selected_feature == "Audio Transcription":
        st.header("🎤 Audio Transcription")
        audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
        if audio_file and st.button("Transcribe"):
            files = {"file": (audio_file.name, audio_file, audio_file.type)}
            try:
                response = requests.post(f"{API_URL}/audio/", files=files)
                if response.ok:
                    st.success(response.json().get("transcription", "No transcription returned."))
                else:
                    st.error(f"Transcription failed: {response.text}")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    # --- Docs ---
    elif st.session_state.selected_feature == "Docs":
        st.header("📚 Documents")
        try:
            response = requests.get(f"{API_URL}/docs/list")
            docs = response.json()
            st.json(docs)
        except Exception as e:
            st.error("Docs endpoint did not return valid JSON.")
            st.write(str(e))

    st.markdown('</div>', unsafe_allow_html=True)