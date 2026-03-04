import torch
import torch.nn.functional as F
from model import ECAPA_TDNN
from audio_processing import load_audio, get_mel_spectrogram
import argparse
from pathlib import Path
import random
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_cosine_similarity(embed1, embed2):
    # Normalize
    embed1 = F.normalize(embed1, p=2, dim=1)
    embed2 = F.normalize(embed2, p=2, dim=1)
    return torch.mm(embed1, embed2.transpose(0, 1)).item()

def load_and_embed(model, device, path):
    waveform = load_audio(path)
    # Ensure at least 2 seconds (pad if needed)
    if waveform.size(1) < 32000:
        pad_amt = 32000 - waveform.size(1)
        waveform = F.pad(waveform, (0, pad_amt))
        
    mel = get_mel_spectrogram(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(mel)
    return embedding

def validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Validating on: {device}")
    
    # Load Model
    model = ECAPA_TDNN(C=512).to(device)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("WARNING: No checkpoint loaded. Using random weights (expect EER ~50%)")
    model.eval()
    
    # Generate Pairs from Data Root
    # We assume folder structure: root/id001/file.wav
    root = Path(args.data_root)
    speakers = [d for d in root.iterdir() if d.is_dir()]
    
    if len(speakers) < 2:
        print("Need at least 2 speakers for validation.")
        return

    scores = []
    labels = [] # 1 for same, 0 for different
    
    print("Generating pairs and computing scores...")
    # Generate 1000 random pairs
    for _ in range(1000):
        # 50% Same Speaker
        if random.random() > 0.5:
            spk = random.choice(speakers)
            files = list(spk.glob("*.wav"))
            if len(files) < 2: continue
            f1, f2 = random.sample(files, 2)
            label = 1
        else:
            # Different Speaker
            s1, s2 = random.sample(speakers, 2)
            f1 = random.choice(list(s1.glob("*.wav")))
            f2 = random.choice(list(s2.glob("*.wav")))
            label = 0
            
        try:
            emb1 = load_and_embed(model, device, f1)
            emb2 = load_and_embed(model, device, f2)
            score = compute_cosine_similarity(emb1, emb2)
            
            scores.append(score)
            labels.append(label)
        except Exception as e:
            continue
            
    # Calculate EER
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    
    print(f"\nResults over {len(scores)} pairs:")
    print(f"EER (Equal Error Rate): {eer*100:.2f}%")
    print(f"Threshold: {thresh:.4f}")
    print("\n(Lower EER is better. Random guess = 50%. Ideal = 0%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder of audio files (toy_dataset)")
    parser.add_argument("--checkpoint", type=str, help="Path to trained model .model file")
    args = parser.parse_args()
    validate(args)
