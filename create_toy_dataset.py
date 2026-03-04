import os
import numpy as np
import scipy.io.wavfile as wav
import shutil
from pathlib import Path

# Configuration
NUM_SPEAKERS = 10
SAMPLES_PER_SPEAKER = 5
DURATION = 3.0 # seconds
SAMPLE_RATE = 16000
DATASET_DIR = Path("toy_dataset")
TRAIN_LIST_FILE = Path("toy_train_list.txt")

from scipy import signal

def generate_tone(frequency, duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Use Sawtooth wave (rich harmonics) to mimic voice timbre
    # This creates energy in f, 2f, 3f... bins, giving the Conv nets 'texture' to see.
    tone = 0.5 * signal.sawtooth(2 * np.pi * frequency * t)
    # Add random noise
    noise = np.random.normal(0, 0.05, tone.shape)
    return (tone + noise).astype(np.float32)

def create_dataset():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir()
    
    train_lines = []
    
    print(f"Generating Toy Dataset for {NUM_SPEAKERS} speakers...")
    
    for speaker_id in range(NUM_SPEAKERS):
        # Assign a specific base frequency to each speaker (e.g. 200Hz, 300Hz...)
        # This makes them easily distinguishable by the model
        base_freq = 200 + (speaker_id * 50) 
        
        speaker_dir = DATASET_DIR / f"id{10000+speaker_id}"
        speaker_dir.mkdir()
        
        for i in range(SAMPLES_PER_SPEAKER):
            # Vary frequency slightly per sample so it's not identical
            freq = base_freq + np.random.uniform(-10, 10)
            
            audio_data = generate_tone(freq, DURATION, SAMPLE_RATE)
            
            filename = f"sample_{i:03d}.wav"
            file_path = speaker_dir / filename
            
            wav.write(str(file_path), SAMPLE_RATE, audio_data)
            
            # Add to train list
            # Path relative to DATASET_DIR root
            rel_path = f"id{10000+speaker_id}/{filename}"
            train_lines.append(f"{rel_path} {speaker_id}")
            
    with open(TRAIN_LIST_FILE, "w") as f:
        f.write("\n".join(train_lines))
        
    print(f"✅ Success! Dataset created at: {DATASET_DIR.absolute()}")
    print(f"✅ Train list created at: {TRAIN_LIST_FILE.absolute()}")
    print("\nYou can now run training with:")
    print(f"python train.py --data_root \"{DATASET_DIR}\" --train_list \"{TRAIN_LIST_FILE}\" --n_classes {NUM_SPEAKERS}")

if __name__ == "__main__":
    create_dataset()
