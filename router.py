import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from model import ECAPA_TDNN
from audio_processing import load_audio, get_mel_spectrogram, segment_audio
import whisper

class PipelineConfig:
    def __init__(self, num_speakers, languages, requires_diarization, requires_translation):
        self.num_speakers = num_speakers # 1 or >1
        self.languages = languages # List['en', 'hi', etc.]
        self.requires_diarization = requires_diarization
        self.requires_translation = requires_translation
        
    def __repr__(self):
        return (f"PipelineConfig(\n"
                f"  Speakers: {self.num_speakers}\n"
                f"  Languages: {self.languages}\n"
                f"  Diarization: {self.requires_diarization}\n"
                f"  Translation: {self.requires_translation}\n"
                f")")

class AudioRouter:
    def __init__(self, speaker_model_path, language_model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load Speaker Model
        self.speaker_model = ECAPA_TDNN(C=512).to(self.device)
        self.speaker_model.load_state_dict(torch.load(speaker_model_path, map_location=self.device))
        self.speaker_model.eval()
        
        # Load Language Model (Whisper Tiny for speed)
        print("Loading Language Model (Whisper Tiny)...")
        self.language_model = whisper.load_model("tiny", device=self.device)
        
    def get_embeddings(self, waveform):
        """Turn audio into a sequence of embeddings (sliding window)"""
        # Segment audio into 1.5s chunks with 0.75s overlap for density
        windows = segment_audio(waveform, window_duration=1.5, overlap=0.5).to(self.device)
        # windows shape: (batch, 1, samples)
        
        mels = []
        for i in range(windows.size(0)):
             # Individual mel extraction due to varying batch sizes logic in memory
             m = get_mel_spectrogram(windows[i])
             mels.append(m)
        
        # Input shape: (Batch, 80, Time)
        mels_tensor = torch.stack(mels).squeeze(1).to(self.device)
        
        with torch.no_grad():
            embeddings = self.speaker_model(mels_tensor)
            
        return embeddings.cpu().numpy()

    def estimate_speaker_count(self, embeddings):
        """
        Use Eigengap analysis or Silhouette Score to estimate k.
        For simple 1 vs Many:
        Run Spectral Clustering for k=1, k=2, k=3.
        Check the stability.
        
        Simplified Heuristic for now:
        1. Compute Similarity Matrix (Cosine)
        2. If verify high average similarity (> 0.7) -> Single Speaker
        3. If high variance -> Multi Speaker
        """
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-6)
        
        # Cosine Similarity Matrix
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Average off-diagonal similarity
        n = sim_matrix.shape[0]
        if n < 2: return 1
        
        # Upper triangle
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        avg_sim = np.mean(upper_tri)
        
        print(f"[Router] Average Embedding Similarity: {avg_sim:.3f}")
        
        # If segments are very similar, it's 1 speaker
        # This threshold depends heavily on the model training.
        # A well trained model has avg_sim ~0.8 for same speaker, <0.2 for diff.
        if avg_sim > 0.65: 
            return 1
        else:
            # It's likely multi-speaker. 
            # We could refine 'k' estimation later.
            return 2 

    def identify_languages(self, waveform):
        """
        Uses Whisper to detect the main language of the audio.
        """
        # Whisper expects 30s of audio. Pad or Trim.
        # Waveform is (1, Samples)
        
        # 1. Prepare Audio
        audio = waveform.squeeze().numpy()
        
        # 2. Pad/Trim to 30s (Sample Rate 16000)
        # However, Whisper's internal 'detect_language' handles raw audio well.
        # But we should ensure it's not empty.
        
        if len(audio) < 16000: # Less than 1s
            return ['en'] # Default
            
        # 3. Detect
        # We need to compute the Mel spectrogram using Whisper's own tools or just pass audio
        # decode() or detect_language()
        
        # Quickest way:
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # detect the spoken language
        _, probs = self.language_model.detect_language(mel)
        detect_lang = max(probs, key=probs.get)
        
        print(f"[Router] Detected Language: {detect_lang}")
        
        return [detect_lang]

    def route(self, audio_path):
        print(f"Analyzing {audio_path}...")
        waveform = load_audio(audio_path)
        
        # 1. Speaker Check
        embeddings = self.get_embeddings(waveform)
        num_speakers = self.estimate_speaker_count(embeddings)
        
        # 2. Language Check
        langs = self.identify_languages(waveform)
        
        # 3. Construct Config
        is_multi_speaker = num_speakers > 1
        is_multi_lang = len(langs) > 1
        
        # If output language is different from input, we need translation
        # For simplicity, assume we always want English output
        req_trans = ('en' not in langs) or is_multi_lang
        
        return PipelineConfig(
            num_speakers=num_speakers,
            languages=langs,
            requires_diarization=is_multi_speaker,
            requires_translation=req_trans
        )

# Example Usage
if __name__ == "__main__":
    # Ensure you have a checkpoint before running
    import sys
    checkpoint = "checkpoints/ecapa_epoch_5.model" 
    # Create dummy check if not exists so import doesn't crash on 'main' run
    if not torch.cuda.is_available():
        pass # Just demonstrating logic
    
    print("Router module loaded.")
