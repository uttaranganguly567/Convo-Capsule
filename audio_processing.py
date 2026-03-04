import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MELS = 80

def load_audio(file_path):
    """
    Loads audio file, resamples to 16kHz, and converts to mono.
    Supports .wav via torchaudio and .mp3/.m4a via librosa fallback.
    """
    try:
        # Try default backend (torchaudio/soundfile)
        waveform, sr = torchaudio.load(file_path)
    except Exception:
        # Fallback to librosa (handles m4a/mp3 better on Windows)
        # librosa loads as numpy (mono=True mixes down immediately)
        # returns (data, sr)
        data, sr = librosa.load(str(file_path), sr=None, mono=False)
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        
    return waveform

def get_mel_spectrogram(waveform):
    """
    Converts waveform to Mel Spectrogram.
    Output shape: (Batch, n_mels, time)
    """
    transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        n_mels=N_MELS,
        window_fn=torch.hamming_window
    )
    
    # Add epsilon to avoid log(0)
    mel_spec = transform(waveform)
    log_mel = torch.log(mel_spec + 1e-6)
    
    # Cepstral Mean Variance Normalization (CMVN)
    mean = log_mel.mean(dim=2, keepdim=True)
    std = log_mel.std(dim=2, keepdim=True)
    log_mel = (log_mel - mean) / (std + 1e-6)
    
    return log_mel

def energy_vad(waveform, threshold=0.001, window_size=512):
    """
    Simple Energy-based Voice Activity Detection.
    Returns indices of active speech.
    """
    # Calculate energy per window
    # Simple squared amplitude
    energy = waveform.pow(2)
    
    # Smooth energy
    energy = torch.nn.functional.avg_pool1d(energy, kernel_size=window_size, stride=window_size)
    
    # Thresholding
    is_speech = energy > threshold
    
    # Run Length Encoding to get segments (start, end)
    # TODO: Implement full segment logic if needed for inference
    
    return is_speech

def segment_audio(waveform, window_duration=2.0, overlap=0.5):
    """
    Cuts the audio into fixed-length windows for the model.
    Window duration in seconds.
    """
    samples_per_window = int(window_duration * SAMPLE_RATE)
    hop = int(samples_per_window * (1 - overlap))
    
    segments = []
    
    # Make sure we have enough samples
    if waveform.size(1) < samples_per_window:
        # Pad
        pad_amt = samples_per_window - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        
    # Sliding window
    # Using unfold: (1, duration, num_windows)
    windows = waveform.unfold(1, samples_per_window, hop)
    
    # Permute to (num_windows, 1, samples)
    windows = windows.permute(1, 0, 2)
    
    return windows
