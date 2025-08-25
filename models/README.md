# Pre-trained Models in Deepfake Detector

This directory contains information about the pre-trained models used in the deepfake detection system. The system uses several specialized models for different aspects of deepfake detection.

## Video Deepfake Detection Model

### `dima806/deepfake_vs_real_image_detection`

**Model Type**: Image Classification (Hugging Face Transformers)

**Function**: Identifies manipulated facial images in video frames by classifying them as real or fake.

**Technical Details**:
- Architecture: Vision Transformer (ViT) based model fine-tuned on deepfake datasets
- Input: Facial region extracted from video frames using MediaPipe
- Output: Binary classification (REAL/FAKE) with confidence scores
- Performance: High accuracy on common deepfake manipulations including face swaps and GAN-generated faces

**Implementation**: Used in `DeepfakeDetector` class in `src/video2.py`

**Usage Context**: 
- Frame-by-frame analysis of video content
- Temporal consistency analysis across consecutive frames

## MediaPipe Face Detection

### `MediaPipe Face Detection`

**Model Type**: Face Detection

**Function**: Locates and extracts face regions from video frames for further analysis.

**Technical Details**:
- Pre-trained model from Google's MediaPipe library
- Two model options: Model 0 (optimized for shorter distances) and Model 1 (optimized for longer distances)
- Our implementation uses Model 1 with a minimum detection confidence of 0.5

**Implementation**: Used in `DeepfakeDetector` class in `src/video2.py`

## MediaPipe Face Mesh

### `MediaPipe Face Mesh`

**Model Type**: Facial Landmark Detection

**Function**: Provides detailed facial landmarks (468 points) used for lip-sync analysis.

**Technical Details**:
- Tracks 468 3D face landmarks in real-time
- Used specifically for mouth region analysis in our lip-sync detection
- Configuration: Static image mode disabled, maximum 1 face, with 0.5 confidence thresholds

**Implementation**: Used in `LipSyncMismatchDetector` class in `src/lipsync.py`

**Usage Context**:
- Extraction of mouth shape and movement metrics
- Calculation of lip synchronization with audio

## Audio Analysis

### Audio Feature Extraction (Non-DL Approach)

While not a pre-trained deep learning model, our system uses sophisticated signal processing techniques from librosa for audio analysis:

**Function**: Detects synthetic audio patterns and voice modulation anomalies commonly found in deepfakes.

**Technical Details**:
- Pitch (F0) extraction using pYIN algorithm from librosa
- Jitter and shimmer calculations for voice quality analysis
- Modulation spectrum analysis for detecting unnatural speech patterns
- Speech rate uniformity measurements

**Implementation**: Used in `VoiceModulationAnalyzer` class in `src/audio1.py`

## Model Storage

When using this system, the pre-trained models are automatically downloaded and cached:

- Hugging Face Transformers models: Cached in `~/.cache/huggingface/`
- MediaPipe models: Cached in system temporary directory

## References

1. `dima806/deepfake_vs_real_image_detection`: [Hugging Face Model Card](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
2. MediaPipe Face Detection: [Documentation](https://developers.google.com/mediapipe/solutions/vision/face_detector)
3. MediaPipe Face Mesh: [Documentation](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
