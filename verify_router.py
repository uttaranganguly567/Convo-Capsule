import torch
import numpy as np
from pathlib import Path
from router import AudioRouter
import scipy.io.wavfile as wav
import os

# Configuration
TOY_DATA_ROOT = Path("toy_dataset")
CHECKPOINT = "checkpoints/ecapa_epoch_5.model"

def create_mixed_audio():
    """Synthesize a multi-speaker file for testing"""
    spk1 = TOY_DATA_ROOT / "id10000" / "sample_000.wav"
    spk2 = TOY_DATA_ROOT / "id10001" / "sample_000.wav"
    
    if not spk1.exists():
        print("Run generate_toy_dataset.py first!")
        return None

    sr, d1 = wav.read(str(spk1))
    sr, d2 = wav.read(str(spk2))
    
    # Mix them (alternating 1s chunks to simulate turn taking)
    # Target length: 6 seconds
    # A (1s) - B (1s) - A (1s) - B (1s) - A (1s) - B (1s)
    
    # Ensure they are at least 1s
    d1 = d1[:16000]
    d2 = d2[:16000]
    
    mixed = np.concatenate((d1, d2, d1, d2, d1, d2)) 
    
    out_path = TOY_DATA_ROOT / "mixed_test.wav"
    wav.write(str(out_path), sr, mixed)
    return out_path

def verify_router():
    if not os.path.exists(CHECKPOINT):
        print("❌ Model checkoint not found!")
        print("Please run: python train.py --data_root \"toy_dataset\" --train_list \"toy_train_list.txt\" --n_classes 10 --epochs 5")
        return

    print("Initializing Router...")
    router = AudioRouter(speaker_model_path=CHECKPOINT)
    
    # Test 1: Single Speaker
    # Pick a random file
    single_file = TOY_DATA_ROOT / "id10000" / "sample_000.wav"
    print(f"\n--- Testing Single Speaker File: {single_file} ---")
    config1 = router.route(single_file)
    print(config1)
    
    if config1.num_speakers == 1:
        print("✅ Correctly identified as Single Speaker")
    else:
        print("❌ Incorrectly identified as Multi Speaker")

    # Test 2: Multi Speaker
    mixed_file = create_mixed_audio()
    print(f"\n--- Testing Multi Speaker File: {mixed_file} ---")
        
    # DEBUG: Check raw embeddings to see if model collapsed
    from audio_processing import load_audio
    waveform = load_audio(mixed_file)
    embeddings = router.get_embeddings(waveform)
    
    # Check variance (if < 0.001, model outputs same vector for everything)
    std_dev = np.std(embeddings, axis=0).mean()
    print(f"[Debug] Mean SD of embeddings: {std_dev:.5f} (If < 0.01, model is collapsed)")
    
    config2 = router.route(mixed_file)
    print(config2)
    
    if config2.num_speakers > 1:
        print("✅ Correctly identified as Multi Speaker")
    else:
        print("❌ Incorrectly identified as Single Speaker")

if __name__ == "__main__":
    verify_router()
