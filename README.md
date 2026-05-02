Convo Capsule is a fully offline, edge-compatible AI meeting analysis tool built entirely in Python. Designed with strict privacy constraints in mind, it completely bypasses cloud APIs to process meeting audio locally.

The pipeline ingests multi-speaker audio, normalizes it, and performs robust speaker diarization using the lightweight sherpa-onnx runtime (eliminating heavy PyTorch dependencies). After transcription via Whisper, the segmented speaker data is fed into a custom-built machine learning model specifically trained to extract action items, summarize discussions, and generate structured Minutes of the Meeting (MoM).

Key Features:

100% Local & Privacy-First: No audio or text data ever leaves the local machine.

Edge-Compatible Diarization: Uses ONNX-runtime for segmentation and embedding, avoiding "DLL hell" and heavy framework overhead.

Custom MoM Model: Replaces standard LLM API calls with a proprietary, locally trained model for specialized meeting summarization.

Robust Audio Pipeline: Dynamically handles sample rate conversion (16kHz) and stereo-to-mono downmixing using librosa and soundfile.
