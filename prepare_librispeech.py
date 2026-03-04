import os
from pathlib import Path
import argparse
import random

def prepare_librispeech(data_root, output_txt="librispeech_train.txt"):
    """
    Scans the data_root for LibriSpeech speaker folders.
    Structure:
      data_root/ (e.g. dev-clean)
        1272/ (Speaker ID)
          128104/ (Chapter ID)
            1272-128104-0000.flac
    """
    root = Path(data_root)
    if not root.exists():
        print(f"Error: Path {root} does not exist.")
        return

    print(f"Scanning {root} for FLAC files...")
    
    lines = []
    # Get all speaker folders
    # In LibriSpeech, the first level of folders are Speaker IDs
    speakers = [d for d in root.iterdir() if d.is_dir()]
    speakers.sort()
    
    print(f"Found {len(speakers)} potential speaker folders.")
    
    valid_speakers = 0
    
    for idx, spk_dir in enumerate(speakers):
        # Recursively find all .flac files
        flac_files = list(spk_dir.glob("**/*.flac"))
        
        if not flac_files:
            continue
            
        valid_speakers += 1
        
        for file_path in flac_files:
            # Get path relative to data_root
            rel_path = file_path.relative_to(root)
            # Use 'idx' as the class label (0, 1, 2...)
            lines.append(f"{str(rel_path)} {idx}")
            
    # Shuffle
    random.shuffle(lines)
    
    # Write
    with open(output_txt, "w") as f:
        f.write("\n".join(lines))
        
    print(f"✅ Created {output_txt} with {len(lines)} samples.")
    print(f"✅ Found {valid_speakers} valid speakers.")
    print("Next steps:")
    print(f"python train.py --data_root \"{data_root}\" --train_list \"{output_txt}\" --n_classes {valid_speakers} --epochs 20")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the extracted LibriSpeech folder (e.g. D:/.../LibriSpeech/dev-clean)")
    parser.add_argument("--output", type=str, default="librispeech_train.txt", help="Output text file path")
    
    args = parser.parse_args()
    prepare_librispeech(args.data_root, args.output)
