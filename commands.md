# Convo Capsule: User Manual & Commands

This guide lists every command needed to run the system from scratch, in the correct order.

## Phase 1: Data Preparation

### Option A: Toy Data (Fastest, for debugging code)
Generates synthetic "Sawtooth" waves to test the pipeline logic.
```bash
python create_toy_dataset.py
```

### Option B: Real Data (VoxCeleb / LibriSpeech)
Scans your downloaded dataset and creates a file list (`train_list.txt`).
**For VoxCeleb:**
```bash
python prepare_voxceleb.py --data_root "D:\Path\To\VoxCeleb" --output "vox_train.txt"
```
**For LibriSpeech:**
```bash
python prepare_librispeech.py --data_root "D:\Path\To\LibriSpeech" --output "libri_train.txt"
```

---

## Phase 2: Training the Model

### Train from Scratch (Reset)
Trains the ECAPA-TDNN model to recognize speakers.
*   `--n_classes`: Set this to the number of speakers found in Phase 1.
*   `--epochs`: 20 is a good starting point.
```bash
python train.py --data_root "D:\Path\To\Data" --train_list "train_list.txt" --n_classes 100 --epochs 20
```

### Resume Training
If training stopped or loss is still high, run the same command again. It loads the latest checkpoint automatically.

---

## Phase 3: Verification (Does it work?)

### Verify Router Logic
Checks if the model can distinguish between Single and Multi-speaker audio.
```bash
python verify_router.py
```
*   **Success**: Single Speaker -> "1 Speaker", Mixed Audio -> "Multi Speaker".
*   **Failure**: Both say "1 Speaker" (Model Collapse -> Retrain!).

### Test on Real File
Runs the Router on a specific file (e.g., your recording).
```bash
python test_real_file.py "path/to/audio.mp3"
```

---

## Phase 4: Execution ( The Final Result )

### Run Full Diarization
This splits the audio, identifies speakers, and transcribes the text.
```bash
python diarizer.py "path/to/audio.mp3"
```
**Output**:
```text
[0.0s - 5.0s] SPEAKER_01: Hello world.
[5.0s - 10.0s] SPEAKER_02: Hi there.
```

---

## Troubleshooting

*   **"Model Collapse"** (Similarity > 0.9): The model isn't learning. Check your learning rate or data.
*   **"CUDA out of memory"**: Reduce `--batch_size` in `train.py` (e.g., to 16 or 8).
*   **"Format not recognised"**: Install `ffmpeg` or ensure `librosa` is installed (`pip install librosa`).
