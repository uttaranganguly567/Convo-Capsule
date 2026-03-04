import torch
import numpy as np
import sklearn.cluster
from pathlib import Path
from audio_processing import load_audio, get_mel_spectrogram
from router import AudioRouter
import whisper

class Diarizer:
    def __init__(self, model_path, use_gpu=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Loading Diarizer on {self.device}...")
        
        # Load our Custom Model for Embedding extraction (Identity)
        self.router = AudioRouter(model_path)
        
        # Load Whisper for Transcription (Content)
        print("Loading Whisper Model (Small)...")
        self.asr_model = whisper.load_model("small", device=self.device)
        
    def segment_audio(self, waveform, sr=16000, window=2.0, step=2.0):
        """
        Cutting audio into non-overlapping chunks for cleaner timestamps.
        Previous overlap caused confusion in transcription timing.
        """
        window_size = int(window * sr)
        step_size = int(step * sr)
        
        segments = []
        timestamps = []
        
        for start in range(0, waveform.shape[1] - window_size + 1, step_size):
            end = start + window_size
            segment = waveform[:, start:end]
            segments.append(segment)
            timestamps.append((start / sr, end / sr))
            
        return segments, timestamps

    def diarize(self, file_path, num_speakers=2):
        print(f"Diarizing: {file_path}")
        
        # 1. Load Audio
        waveform = load_audio(file_path) # Returns (1, samples) tensor
        
        # 2. Extract Embeddings for every chunk
        segments, timestamps = self.segment_audio(waveform)
        embeddings = []
        
        print(f"Extracting embeddings for {len(segments)} segments...")
        with torch.no_grad():
            for seg in segments:
                # Convert to Mel Spectrogram: (1, 80, Time)
                mel = get_mel_spectrogram(seg)
                emb = self.router.speaker_model(mel.to(self.device))
                embeddings.append(emb.cpu().numpy().squeeze())
                
        embeddings = np.array(embeddings)
        
        # 3. Clustering (Who is who?)
        print(f"Clustering into {num_speakers} speakers...")
        clustering = sklearn.cluster.SpectralClustering(
            n_clusters=num_speakers,
            affinity='cosine',
            assign_labels='discretize',
            random_state=0
        ).fit(embeddings)
        
        labels = clustering.labels_
        
        # 4. Merge & Transcribe
        print("Transcribing segments...")
        
        results = []
        current_speaker = labels[0]
        current_start = timestamps[0][0]
        current_end = timestamps[0][1]
        
        for i in range(1, len(labels)):
            if labels[i] == current_speaker:
                # Extend the current segment
                current_end = timestamps[i][1]
            else:
                # Speaker changed! Save previous segment.
                results.append({
                    "speaker": f"SPEAKER_{current_speaker}",
                    "start": current_start,
                    "end": current_end
                })
                # Start new segment
                current_speaker = labels[i]
                current_start = timestamps[i][0] # Should be == previous current_end
                current_end = timestamps[i][1]
        
        # Append last segment
        results.append({ "speaker": f"SPEAKER_{current_speaker}", "start": current_start, "end": current_end })
        
        # 5. ASR Pass (Transcribe each merged segment)
        final_lines = []
        audio_np = waveform.squeeze().numpy()
        
        print(f"Processing {len(results)} merged segments...")
        for res in results:
            start_sample = int(res['start'] * 16000)
            end_sample = int(res['end'] * 16000)
            
            # Ensure we don't go out of bounds
            if start_sample >= len(audio_np): continue
            end_sample = min(end_sample, len(audio_np))
            
            chunk = audio_np[start_sample:end_sample]
            
            # Use Whisper to transcribe (upgraded to 'small' or 'medium' for better accuracy)
            # You can change model size in __init__
            if len(chunk) > 1600: # Ignore tiny chunks < 0.1s
                text = self.asr_model.transcribe(chunk, fp16=(self.device.type=='cuda'))['text']
                entry = f"[{res['start']:.1f}s - {res['end']:.1f}s] {res['speaker']}: {text.strip()}"
                final_lines.append(entry)
                print(entry)
        
        transcript = "\n".join(final_lines)
        
        # 6. Save to File
        output_dir = Path("Meeting_Analyses")
        output_dir.mkdir(exist_ok=True)
        
        input_name = Path(file_path).stem
        out_file = output_dir / f"{input_name}_analysis.txt"
        
        with open(out_file, "w", encoding='utf-8') as f:
            f.write(transcript)
            
        print(f"\n✅ Transcript saved to: {out_file}")
        return transcript

if __name__ == "__main__":
    # Test script
    import sys
    checkpoint = "checkpoints/ecapa_epoch_20.model" 
    if not Path(checkpoint).exists(): checkpoint = "checkpoints/ecapa_epoch_5.model"
    
    dz = Diarizer(model_path=checkpoint)
    
    target = r"D:\DOCUMENTS\Coding\Projects\Convo Capsule (Final Year Project)\Datasets\Single_Language_Multi_Speaker\Alex_and_Jamie.m4a"
    if len(sys.argv) > 1: target = sys.argv[1]
    
    dz.diarize(target)
