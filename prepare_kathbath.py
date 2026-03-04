import os
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse
import soundfile as sf
import io

def prepare_hindi_dataset(output_dir=r"D:\DOCUMENTS\Coding\Projects\Convo Capsule (Final Year Project)\Datasets\Hindi_OpenSLR"):
    """
    Downloads Hindi ASR Challenge (SLR103) dataset from OpenSLR.
    This is an OPEN dataset, no login required.
    """
    url = "https://www.openslr.org/resources/103/Hindi_train.tar.gz"
    filename = "Hindi_train.tar.gz"
    
    print(f"⬇️  Downloading Hindi Data from OpenSLR (4.2GB)...")
    print(f"URL: {url}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = output_path / filename
    
    # 1. Download File
    if not tar_path.exists():
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print("✅ Archive already exists. Skipping download.")

    # 2. Extract
    print("📦 Extracting audio files...")
    train_list_file = "hindi_train.txt"
    lines = []
    
    speaker_map = {}
    next_id = 0
    count = 0
    
    # OpenSLR 103 structure: Usually just a folder of .wav files or subfolders
    # We will inspect on the fly
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar):
            if not member.isfile():
                continue
            if not member.name.endswith(".wav"):
                continue
                
            # Extract to disk
            f = tar.extractfile(member)
            audio_bytes = f.read()
            
            # File name mapping
            # Typically: speaker_id_utterance_id.wav or similar
            # Example: 001_001.wav
            fname = Path(member.name).name
            
            # Try to guess speaker ID from filename
            # Often first part before underscore
            parts = fname.split('_')
            if len(parts) > 1:
                speaker_id = parts[0]
            else:
                speaker_id = "unknown"
                
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = next_id
                next_id += 1
            
            spk_int_id = speaker_map[speaker_id]
            
            # Save Structure: output_dir/speaker_id/filename
            spk_dir = output_path / speaker_id
            spk_dir.mkdir(exist_ok=True)
            
            save_path = spk_dir / fname
            
            if not save_path.exists():
                with open(save_path, "wb") as wb:
                    wb.write(audio_bytes)
            
            lines.append(f"{str(save_path)} {spk_int_id}")
            count += 1
            
            if count > 5000: # Limit for quick start
               # break # Remove break to process full if desired, but 5k is enough for demo
               pass

    # Write List
    with open(train_list_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            
    print(f"\n✅ Done! Processed {count} files.")
    print(f"📝 Training list saved to: {train_list_file}")
    print(f"👥 Found {len(speaker_map)} unique speakers.")
    print(f"\nTo Train, run: python train.py --train_list {train_list_file} --n_classes {len(speaker_map)}")

if __name__ == "__main__":
    prepare_hindi_dataset()
