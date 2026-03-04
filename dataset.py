import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from audio_processing import load_audio, get_mel_spectrogram

class VoxCelebDataset(Dataset):
    def __init__(self, base_path, train_list_path, segment_length=2.0):
        """
        Args:
            base_path: Path to the root of the extracted VoxCeleb data.
            train_list_path: Path to a txt file with lines: "speaker_id/video_id/audio_id.wav speaker_int_id"
            segment_length: Length of audio segment to train on (in seconds).
        """
        self.base_path = Path(base_path)
        self.segment_length = segment_length
        self.data = []
        
        # Parse the training list
        # It's better to verify files exist first, but for speed we'll assume valid list
        with open(train_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_path = parts[0]
                    label = int(parts[1])
                    self.data.append((rel_path, label))
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rel_path, label = self.data[index]
        audio_path = self.base_path / rel_path
        
        try:
            waveform = load_audio(audio_path)
            
            # Random Crop for Fixed Length Training
            # (1, samples)
            total_samples = waveform.size(1)
            target_samples = int(self.segment_length * 16000)
            
            if total_samples <= target_samples:
                # Pad if too short
                pad_amt = target_samples - total_samples
                # Repeat audio if it's very short? Or just zero pad. Zero pad is safer for now.
                # Actually, repeating is often better for speaker ID so we don't learn "silence" = "speaker X"
                num_repeats = (target_samples // total_samples) + 1
                waveform = waveform.repeat(1, num_repeats)
                waveform = waveform[:, :target_samples]
            else:
                # Random start
                start_idx = random.randint(0, total_samples - target_samples)
                waveform = waveform[:, start_idx:start_idx+target_samples]
            
            # Convert to Mel Spectrogram
            # (1, n_mels, time)
            mel = get_mel_spectrogram(waveform)
            
            # Squeeze channel dim: (n_mels, time)
            mel = mel.squeeze(0)
            
            return mel, label
            
        except Exception as e:
            # Fallback for bad files (shouldn't happen often in verified dataset)
            print(f"Error loading {rel_path}: {e}")
            # Recursively try another random item so batch doesn't crash
            return self.__getitem__(random.randint(0, len(self.data)-1))
