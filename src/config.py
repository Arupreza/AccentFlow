import os
import tempfile
import streamlit as st

@st.cache_resource
def init_dirs():
    temp_dir = tempfile.mkdtemp()
    uploads_dir = os.path.join(temp_dir, "uploads")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return temp_dir, uploads_dir, output_dir

def init_session_state():
    defaults = {
        'processed': False,
        'current_tab': "Upload",
        'generated_files': {},
        'log_messages': [],
        'progress': 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def set_page_config():
    st.set_page_config(
        page_title="Audio Processing App",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )