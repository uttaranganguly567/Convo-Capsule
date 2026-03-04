import streamlit as st
import os
from pathlib import Path
import re
import time
from diarizer import Diarizer

# Page Config
st.set_page_config(
    page_title="Convo Capsule | AI Meeting Analysis",
    page_icon="🎙️",
    layout="wide"
)

# Title & styling
st.title("🎙️ Convo Capsule")
st.markdown("### AI-Powered Speaker Diarization & Transcription")

# Sidebar for Config
st.sidebar.header("Configuration")

def get_best_checkpoint():
    """Finds the latest checkpoint in checkpoints/ folder."""
    base_dir = Path("checkpoints")
    if not base_dir.exists():
        return None
    
    models = list(base_dir.glob("*.model"))
    if not models:
        return None
        
    # Sort by modification time (or name if numbered)
    # Using name epoch_X is safer if timestamps are messed up
    # regex to find epoch number
    def get_epoch(p):
        m = re.search(r"epoch_(\d+)", p.name)
        return int(m.group(1)) if m else 0
    
    best_model = max(models, key=get_epoch)
    return str(best_model)

# Model Selection
checkpoint_path = get_best_checkpoint()
if checkpoint_path:
    st.sidebar.success(f"Loaded Model: `{Path(checkpoint_path).name}`")
else:
    st.sidebar.warning("No custom model found. Using default weights if available.")
    checkpoint_path = "default" # Let Diarizer handle or fail gracefully

# Speakers Config
num_speakers = st.sidebar.number_input(
    "Number of Speakers (Optional)", 
    min_value=2, 
    max_value=10, 
    value=2,
    help="Leave as 2 if unknown, or set accurate number for better clustering."
)

@st.cache_resource
def load_diarizer(model_path):
    """Loads the model once and caches it."""
    # If model_path is invalid/default, we might need a fallback or let it error
    # For now, assuming user has trained or has a model.
    if model_path == "default" or not os.path.exists(model_path):
         st.error("No model checkpoint found! Please train the model first.")
         return None
         
    return Diarizer(model_path=model_path, use_gpu=True)

# Main Area
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a", "flac"])

if uploaded_file:
    # Save to temp
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(temp_path)
    
    if st.button("🔍 Analyze Audio", type="primary"):
        diarizer = load_diarizer(checkpoint_path)
        
        if diarizer:
            status_container = st.status("Processing Audio...", expanded=True)
            
            try:
                # 1. Start Analysis
                status_container.write("🧠 Detect Speakers & Language...")
                # Note: We are mocking progress steps because diarize() is blocking.
                # Ideally, diarize would accept a callback.
                # For now, we trust the logs or simple wait.
                
                start_time = time.time()
                
                # RUN DIARIZATION
                # If num_speakers is specified by user, use it.
                # Otherwise pass None to let router decide? 
                # The UI input defaults to 2. Let's send it.
                transcript = diarizer.diarize(temp_path, num_speakers=num_speakers)
                
                duration = time.time() - start_time
                status_container.update(label=f"Analysis Complete in {duration:.1f}s!", state="complete", expanded=False)
                
                # 2. Display Results
                st.divider()
                st.subheader("📝 Meeting Transcript")
                
                # Parse Transcript for Chat UI
                # Format: [0.0s - 2.5s] SPEAKER_0: Hello world
                pattern = re.compile(r"\[([\d\.]+)s - ([\d\.]+)s\] (SPEAKER_\d+): (.*)")
                
                for line in transcript.split('\n'):
                    match = pattern.match(line.strip())
                    if match:
                        start, end, speaker, text = match.groups()
                        
                        # Avatar mapping
                        avatars = {
                            "SPEAKER_0": "👤",
                            "SPEAKER_1": "🧑‍💼",
                            "SPEAKER_2": "👩‍💻",
                            "SPEAKER_3": "👨‍🏫"
                        }
                        avatar = avatars.get(speaker, "🗣️")
                        
                        with st.chat_message(speaker, avatar=avatar):
                            st.write(f"**{speaker}** ({start}s): {text}")
                    else:
                        # Fallback for weird lines
                        if line.strip():
                            st.text(line)
                            
                # 3. Download
                st.download_button(
                    label="📥 Download Transcript",
                    data=transcript,
                    file_name=f"{Path(uploaded_file.name).stem}_transcript.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                status_container.update(label="Analysis Failed", state="error")
                st.error(f"An error occurred: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Convo Capsule Team")
