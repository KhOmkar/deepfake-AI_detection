## Deepfake Detection System

AI-powered multimodal deepfake detection for video and audio. The system combines:

- Video deepfake detection using a pre-trained classifier on face crops
- Audio synthetic voice detection via signal-processing features (no training)
- Lip-sync mismatch analysis using facial landmarks and audio–visual correlation

### Highlights
- Multi-modal: video, audio, and lip‑sync analysis
- Uses state-of-the-art pre-trained model `dima806/deepfake_vs_real_image_detection`
- MediaPipe for robust face detection and face mesh landmarks
- Streamlit and Flask apps, plus a CLI workflow
- JSON report exporting (Streamlit)

---

## Project Structure
```
deepfake-detector-1/
  app.py                  # Flask web app (file upload + JSON results)
  main.py                 # CLI: analyze a video/audio file end-to-end
  streamlit_app.py        # Streamlit UI with progress + export
  requirements.txt
  models/README.md        # Model documentation & references
  templates/index.html    # Flask UI template
  src/
    video2.py             # DeepfakeDetector (video)
    audio1.py             # VoiceModulationAnalyzer (audio)
    lipsync.py            # LipSyncMismatchDetector (audio–visual)
    utils/                # Extra utils (optional helpers)
  tests/                  # Example test scripts (interactive)
```

---

## Installation

1) Python 3.9+ recommended. Ensure FFmpeg is installed and available on PATH (for audio extraction in some paths).

2) Create and activate a virtual environment:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
# or: source .venv/bin/activate  (macOS/Linux)
```

3) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

First run will download model weights (Hugging Face) and MediaPipe assets.

---

## Usage

### 1) Command Line (main.py)
Run and enter a file path when prompted:
```bash
python main.py
```
- Video files: runs video deepfake detection, extracts audio (if present), runs audio analysis and lip‑sync.
- Audio files: runs audio synthetic voice analysis only.

### 2) Flask Web App (app.py)
```bash
python app.py
# Open browser at http://localhost:5000
```
- Upload a video/audio file and receive JSON results.
- Max upload size: 100MB (configurable).

### 3) Streamlit App (streamlit_app.py)
```bash
streamlit run streamlit_app.py
```
- Drag & drop files, see progress and rich results.
- Exports a JSON report combining results across modules.

---

## Modules Overview

### src/video2.py — DeepfakeDetector
- Face detection: MediaPipe Face Detection (model_selection=1)
- Classifier: Hugging Face pipeline `dima806/deepfake_vs_real_image_detection`
- Flow: sample consecutive frames → detect face → crop with padding → classify → aggregate across frames
- Outputs: final prediction (REAL/FAKE), confidence, frame consistency, temporal consistency, per-frame scores
- Realtime: optional webcam/stream overlay (`analyze_realtime`)

### src/audio1.py — VoiceModulationAnalyzer
- Signal-processing based (no deep learning)
- Preprocessing: mono @16kHz, normalize, trim silence
- Segmentation: 3.0s segments, 50% overlap; pads short audio
- Features: pitch/jitter/shimmer, modulation spectrum, speech timing uniformity
- Rule-based scoring over segments → aggregate decision with risk level

### src/lipsync.py — LipSyncMismatchDetector
- Landmarks: MediaPipe Face Mesh (468 points), mouth/inner-lip indices
- Audio extraction: MoviePy → FFmpeg fallback → librosa last-resort
- Mouth metrics: openness, width, area, aspect ratio per sampled frame
- Audio features: per-timestamp short window energy + coarse speech probability
- Sync: sweep temporal offsets to maximize correlation; analyze segment consistency
- Outputs: `mismatch_classification` (SYNCHRONIZED/MISMATCHED), confidence, optimal offset, score distribution

---

## Saving Artifacts (optional)
By default, most processing is in-memory. To persist artifacts:
- Lip‑sync debug frames: set `self.debug_mode=True` and `self.save_debug_frames=True` in `LipSyncMismatchDetector` to save annotated frames under `lipsync_debug/`.
- Face crops / frames (video2): add optional saving in `analyze_video` to write crops under `runs/video_frames/<video_name>/`.
- Audio segments (audio1): after segmentation, write WAVs to `runs/audio_segments/<base_name>/`.

Suggested structure:
```
runs/
  video_frames/<video_name>/frame_XXXXXX_face.jpg
  audio_segments/<base_name>/seg_###.wav
lipsync_debug/
```

---

## Tests
The `tests/` folder contains standalone scripts demonstrating each module. They prompt for a file path and print results. They are not automated unit tests but helpful for manual verification.

Run a script, e.g.:
```bash
python tests/test_video_analyzer.py
```

---

## Troubleshooting
- FFmpeg not found: install FFmpeg and ensure it’s on PATH (required by some audio extraction paths).
- Model download issues: confirm internet access; try `pip install --upgrade transformers`.
- CUDA/GPU: the apps auto-detect GPU. If drivers/toolkit are missing, it falls back to CPU.
- No face detected: ensure the video contains a frontal, sufficiently large face; try changing lighting or start position.
- Very short audio: the analyzer pads very short clips, but <0.1s will be skipped.

---

## References & Models
- Video classifier: `dima806/deepfake_vs_real_image_detection` (Hugging Face)
- MediaPipe Face Detection and Face Mesh documentation
- Librosa for audio feature extraction

See `models/README.md` for more details.

---

## Privacy & Ethics
Use responsibly. Do not upload sensitive content without consent. Predictions are probabilistic and should support, not replace, expert judgment.

---

## Contributors / Group Details

- <Name Omkar Khilare>
- <Name Sachin Bhabad>
- <Name Prajwal Jadhav>
- <Name Krishna Gite>
- <Name Shravan Mole>



