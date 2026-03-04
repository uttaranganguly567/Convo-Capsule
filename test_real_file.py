import torch
import sys
from pathlib import Path
from router import AudioRouter

# Update this path if you change where the model is saved
CHECKPOINT = "checkpoints/ecapa_epoch_20.model" 
# Fallback to epoch 5 if 20 doesn't exist yet
if not Path(CHECKPOINT).exists():
     CHECKPOINT = "checkpoints/ecapa_epoch_5.model"

def test_file(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at {path}")
        return

    print(f"Loading Router with model: {CHECKPOINT}")
    try:
        router = AudioRouter(speaker_model_path=CHECKPOINT)
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Please train the model first!")
        return

    print(f"\n--- Analyzing Real File: {path.name} ---")
    print(f"Full path: {path}")
    
    try:
        # Load audio (torchaudio supports mp3/m4a if backend is available)
        config = router.route(path)
        
        print("\n=== Result ===")
        print(config)
        
        if config.num_speakers > 1:
            print("✅ Result: Multi-Speaker Detected -> Needs Diarization")
        else:
            print("👤 Result: Single-Speaker Detected -> No Diarization Needed")
            
    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        print("Note: For .m4a/.mp3 files, you might need 'ffmpeg' installed on Windows.")

if __name__ == "__main__":
    # If a path is provided via command line, use it
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default to the file user asked about
        target = r"D:\DOCUMENTS\Coding\Projects\Convo Capsule (Final Year Project)\Datasets\Single_Language_Multi_Speaker\Alex_and_Jamie.m4a"
        
    test_file(target)
