import os
from pathlib import Path
import argparse
import random

def prepare_voxceleb(data_root, output_txt="voxceleb_train.txt"):
    """
    Scans the data_root for speaker folders and generates a training list.
    Expected usage: data_root points to 'VoxCeleb1/dev/wav' or similar.
    Structure:
      data_root/
        id10001/
          12345/
            00001.wav
        id10002/
          ...
    """
    root = Path(data_root)
    if not root.exists():
        print(f"Error: Path {root} does not exist.")
        return

    print(f"Scanning {root}...")
    
    lines = []
    # Get all speaker folders (id10001, id10002, ...)
    speakers = [d for d in root.iterdir() if d.is_dir()]
    speakers.sort()
    
    speaker_map = {}
    
    print(f"Found {len(speakers)} speaker folders.")
    
    for idx, spk_dir in enumerate(speakers):
        # Map folder name (id10001) to integer (0)
        speaker_map[spk_dir.name] = idx
        
        # Find all .wav files recursively
        wav_files = list(spk_dir.glob("**/*.wav"))
        
        for wav_path in wav_files:
            # Get path relative to data_root
            rel_path = wav_path.relative_to(root)
            # Format: relative/path/to/audio.wav class_id
            lines.append(f"{str(rel_path)} {idx}")
            
    # Shuffle for better training
    random.shuffle(lines)
    
    # Write to file
    with open(output_txt, "w") as f:
        f.write("\n".join(lines))
        
    print(f"✅ Created {output_txt} with {len(lines)} samples from {len(speakers)} speakers.")
    print("Next steps:")
    print(f"1. Run training:")
    print(f"   python train.py --data_root \"{data_root}\" --train_list \"{output_txt}\" --n_classes {len(speakers)} --epochs 20")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the folder containing speaker subfolders (e.g., VoxCeleb1/wav)")
    parser.add_argument("--output", type=str, default="voxceleb_train.txt", help="Output text file path")
    
    args = parser.parse_args()
    prepare_voxceleb(args.data_root, args.output)
