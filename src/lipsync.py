import cv2
import numpy as np
import torch
import librosa
import mediapipe as mp
from scipy.signal import correlate
from scipy import stats
import warnings
import os
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')

class LipSyncMismatchDetector:
    def __init__(self, use_gpu=True):
        """
        Lip-sync mismatch detector that analyzes audio-visual synchronization
        without requiring model training
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe face mesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define mouth landmark indices
        # MediaPipe Face Mesh provides 468 landmarks
        # These indices correspond to mouth landmarks
        self.mouth_landmarks = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]
        
        # Inner lip landmarks for measuring mouth openness
        self.inner_mouth = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        
        # Analysis parameters
        self.segment_duration = 0.5  # Duration of analysis segments in seconds
        self.offset_range = 0.5  # Maximum offset to check (±seconds)
        self.step_size = 0.05  # Step size for offset testing
        
        # Debug mode
        self.debug_mode = False
        self.save_debug_frames = False
        self.debug_dir = "lipsync_debug"
    
    def extract_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio track from video file using multiple methods with fallback
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Tuple[np.ndarray, int]: Audio samples and sample rate
        """
        import tempfile
        import subprocess
        import os
        
        # Create temp file for audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        try:
            # Method 1: Try using MoviePy
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                audio, sr = librosa.load(temp_audio, sr=None, mono=True)
                print("Audio extracted successfully using MoviePy")
                return audio, sr
            except Exception as e:
                print(f"MoviePy extraction failed: {e}")
            
            # Method 2: Try using FFmpeg directly
            try:
                cmd = [
                    'ffmpeg', '-i', video_path, 
                    '-f', 'wav', '-ab', '192000', 
                    '-vn', temp_audio, '-y'
                ]
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                    audio, sr = librosa.load(temp_audio, sr=None, mono=True)
                    print("Audio extracted successfully using FFmpeg")
                    return audio, sr
            except Exception as e:
                print(f"FFmpeg extraction failed: {e}")
            
            # Method 3: Last resort - try librosa directly (rarely works with videos)
            try:
                audio, sr = librosa.load(video_path, sr=None, mono=True)
                print("Audio extracted successfully using librosa directly")
                return audio, sr
            except Exception as e:
                print(f"Librosa direct extraction failed: {e}")
            
            # If we got here, all methods failed
            raise Exception("All audio extraction methods failed")
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return np.array([]), 0
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio):
                try:
                    os.unlink(temp_audio)
                except:
                    pass
    
    def extract_mouth_metrics(self, frame, landmarks) -> Dict:
        """
        Extract mouth shape and movement metrics from landmarks
        
        Args:
            frame: Video frame
            landmarks: Facial landmarks from MediaPipe
            
        Returns:
            Dict: Mouth metrics including openness, width, area
        """
        h, w = frame.shape[:2]
        
        # Extract mouth landmarks
        mouth_points = []
        inner_mouth_points = []
        
        for idx in self.mouth_landmarks:
            lm = landmarks.landmark[idx]
            mouth_points.append((int(lm.x * w), int(lm.y * h)))
            
        for idx in self.inner_mouth:
            lm = landmarks.landmark[idx]
            inner_mouth_points.append((int(lm.x * w), int(lm.y * h)))
        
        # Convert to numpy arrays
        mouth_points = np.array(mouth_points)
        inner_mouth_points = np.array(inner_mouth_points)
        
        if len(mouth_points) == 0 or len(inner_mouth_points) == 0:
            return {
                "openness": 0,
                "width": 0,
                "area": 0,
                "aspect_ratio": 0,
                "valid": False
            }
        
        # Calculate mouth metrics
        # 1. Mouth openness (vertical distance)
        top_lip = np.min(inner_mouth_points[:, 1])
        bottom_lip = np.max(inner_mouth_points[:, 1])
        openness = bottom_lip - top_lip
        
        # 2. Mouth width (horizontal distance)
        left_corner = np.min(mouth_points[:, 0])
        right_corner = np.max(mouth_points[:, 0])
        width = right_corner - left_corner
        
        # 3. Mouth area (approximated)
        area = cv2.contourArea(inner_mouth_points) if len(inner_mouth_points) > 3 else 0
        
        # 4. Aspect ratio (width/height)
        aspect_ratio = width / max(openness, 1)  # Avoid division by zero
        
        return {
            "openness": openness,
            "width": width,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "valid": True
        }
    
    def extract_audio_features(self, audio: np.ndarray, sr: int, timestamps: List[float]) -> List[Dict]:
        """
        Extract audio features corresponding to frame timestamps
        
        Args:
            audio (np.ndarray): Audio samples
            sr (int): Sample rate
            timestamps (List[float]): Frame timestamps in seconds
            
        Returns:
            List[Dict]: Audio features for each timestamp
        """
        features = []
        
        # Calculate window size for feature extraction (centered on each timestamp)
        window_size = int(0.1 * sr)  # 100ms window
        
        for timestamp in timestamps:
            # Convert timestamp to sample index
            center_idx = int(timestamp * sr)
            
            # Define window boundaries
            start_idx = max(0, center_idx - window_size // 2)
            end_idx = min(len(audio), center_idx + window_size // 2)
            
            if end_idx <= start_idx or start_idx >= len(audio):
                # Invalid segment, add empty features
                features.append({
                    "energy": 0,
                    "speech_prob": 0,
                    "valid": False
                })
                continue
            
            # Extract audio segment
            segment = audio[start_idx:end_idx]
            
            # 1. Calculate energy
            energy = np.mean(np.abs(segment))
            
            # 2. Calculate spectral features
            if len(segment) > 0:
                # Calculate spectrogram
                spec = np.abs(librosa.stft(segment, n_fft=512, hop_length=128))
                
                # Simple speech probability based on energy in speech frequency range
                # (Higher energy in 200-3000Hz range indicates speech)
                if spec.shape[1] > 0:
                    speech_freq_mask = np.logical_and(
                        librosa.fft_frequencies(sr=sr, n_fft=512) >= 200,
                        librosa.fft_frequencies(sr=sr, n_fft=512) <= 3000
                    )
                    speech_energy = np.mean(spec[speech_freq_mask, :])
                    total_energy = np.mean(spec)
                    speech_prob = speech_energy / (total_energy + 1e-10)
                else:
                    speech_prob = 0
            else:
                speech_prob = 0
            
            features.append({
                "energy": energy,
                "speech_prob": speech_prob,
                "valid": True
            })
        
        return features
    
    def calculate_sync_score(self, mouth_metrics: List[Dict], audio_features: List[Dict], 
                            temporal_offset: float = 0.0) -> float:
        """
        Calculate synchronization score between mouth movements and audio features
        
        Args:
            mouth_metrics (List[Dict]): Mouth shape metrics
            audio_features (List[Dict]): Audio features
            temporal_offset (float): Temporal offset to apply (in seconds)
            
        Returns:
            float: Synchronization score (higher = better sync)
        """
        if not mouth_metrics or not audio_features:
            return 0.0
        
        # Filter valid entries
        valid_pairs = [(m, a) for m, a in zip(mouth_metrics, audio_features) 
                       if m.get("valid", False) and a.get("valid", False)]
        
        if len(valid_pairs) < 3:  # Need at least 3 points for correlation
            return 0.0
        
        # Extract features
        openness = np.array([m["openness"] for m, _ in valid_pairs])
        area = np.array([m["area"] for m, _ in valid_pairs])
        energy = np.array([a["energy"] for _, a in valid_pairs])
        speech_prob = np.array([a["speech_prob"] for _, a in valid_pairs])
        
        # Normalize features
        def normalize(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-10)
        
        if len(openness) > 0 and np.std(openness) > 0:
            openness_norm = normalize(openness)
            area_norm = normalize(area)
            energy_norm = normalize(energy)
            
            # Calculate correlation between mouth openness and audio energy
            corr_openness = stats.pearsonr(openness_norm, energy_norm)[0] if len(openness_norm) > 1 else 0
            
            # Calculate correlation between mouth area and audio energy
            corr_area = stats.pearsonr(area_norm, energy_norm)[0] if len(area_norm) > 1 else 0
            
            # Weight correlations based on speech probability
            weight = np.mean(speech_prob)
            
            # Final sync score (weighted average of correlations)
            sync_score = weight * (0.7 * corr_openness + 0.3 * corr_area)
            
            # Handle NaN values
            if np.isnan(sync_score):
                sync_score = 0.0
            
            return max(0, sync_score)  # Only positive correlations are meaningful for sync
        else:
            return 0.0
    
    def find_optimal_offset(self, video_path: str, 
                           sample_rate: int = 25, 
                           max_frames: int = 100) -> Dict:
        """
        Find the optimal temporal offset between audio and video
        
        Args:
            video_path (str): Path to video file
            sample_rate (int): Frame sampling rate
            max_frames (int): Maximum number of frames to analyze
            
        Returns:
            Dict: Offset analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract audio
        audio, audio_sr = self.extract_audio_from_video(video_path)
        if len(audio) == 0:
            return {"error": "Failed to extract audio from video"}
        
        # Determine frame indices to sample
        if total_frames <= max_frames:
            frame_indices = np.arange(total_frames)
        else:
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        # Extract mouth metrics and timestamps
        mouth_metrics = []
        timestamps = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate timestamp for this frame
            timestamp = frame_idx / fps
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with face mesh
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]  # First face only
                metrics = self.extract_mouth_metrics(frame, landmarks)
                
                if metrics["valid"]:
                    mouth_metrics.append(metrics)
                    timestamps.append(timestamp)
            
            if self.debug_mode and self.save_debug_frames and len(timestamps) % 10 == 0:
                if not os.path.exists(self.debug_dir):
                    os.makedirs(self.debug_dir)
                # Draw landmarks on face for debugging
                annotated_frame = frame.copy()
                for idx in self.mouth_landmarks + self.inner_mouth:
                    lm = landmarks.landmark[idx]
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
                cv2.imwrite(f"{self.debug_dir}/frame_{len(timestamps)}.jpg", annotated_frame)
        
        cap.release()
        
        if len(mouth_metrics) < 10:
            return {"error": "Not enough valid frames with mouth landmarks"}
        
        # Extract audio features for each timestamp
        audio_features = self.extract_audio_features(audio, audio_sr, timestamps)
        
        # Test different offsets to find the optimal one
        offsets = np.arange(-self.offset_range, self.offset_range + self.step_size, self.step_size)
        sync_scores = []
        
        for offset in offsets:
            # Adjust audio features based on offset
            adjusted_indices = []
            for i, ts in enumerate(timestamps):
                # Find closest audio timestamp after applying offset
                adjusted_ts = ts + offset
                if 0 <= adjusted_ts < duration:
                    adjusted_indices.append(i)
            
            if len(adjusted_indices) < 3:
                sync_scores.append(0)
                continue
            
            # Calculate sync score with this offset
            subset_mouth = [mouth_metrics[i] for i in adjusted_indices]
            subset_audio = [audio_features[i] for i in adjusted_indices]
            score = self.calculate_sync_score(subset_mouth, subset_audio)
            sync_scores.append(score)
        
        # Find optimal offset
        best_idx = np.argmax(sync_scores)
        optimal_offset = offsets[best_idx]
        max_score = sync_scores[best_idx]
        
        # Calculate score distribution statistics
        score_mean = np.mean(sync_scores)
        score_std = np.std(sync_scores)
        
        # Detect unusual patterns
        score_range = max(sync_scores) - min(sync_scores)
        unusually_flat = score_range < 0.1  # Suspiciously flat correlation curve
        
        # Calculate mismatch confidence based on maximum correlation 
        # and difference from other correlations
        mismatch_confidence = 0.0
        if max_score < 0.2:
            # Very low correlation across all offsets indicates poor sync
            mismatch_confidence = 0.8
        elif unusually_flat:
            # Suspiciously flat correlation curve
            mismatch_confidence = 0.7
        else:
            # Higher mismatch confidence when max score is lower
            mismatch_confidence = 1.0 - max_score
            # Adjust based on score distribution
            mismatch_confidence = min(mismatch_confidence + (1.0 - score_range), 1.0)
        
        # Define threshold for definitive lip-sync mismatch
        is_mismatched = mismatch_confidence > 0.6
        
        return {
            "optimal_offset": optimal_offset,
            "max_sync_score": max_score,
            "is_mismatched": is_mismatched,
            "mismatch_confidence": mismatch_confidence,
            "analyzed_frames": len(mouth_metrics),
            "score_distribution": {
                "mean": score_mean,
                "std": score_std,
                "range": score_range,
                "unusually_flat": unusually_flat
            }
        }
    
    def analyze_sync_consistency(self, video_path: str, segment_duration: float = 3.0) -> Dict:
        """
        Analyze consistency of audio-visual synchronization across video segments
        
        Args:
            video_path (str): Path to video file
            segment_duration (float): Duration of each analysis segment
            
        Returns:
            Dict: Sync consistency analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract audio
        audio, audio_sr = self.extract_audio_from_video(video_path)
        if len(audio) == 0:
            return {"error": "Failed to extract audio from video"}
        
        # Define segments
        segment_length = int(segment_duration * fps)
        n_segments = max(1, int(total_frames / segment_length))
        
        # Limit number of segments to analyze
        max_segments = 10
        n_segments = min(n_segments, max_segments)
        
        # Analyze each segment
        segment_results = []
        
        for i in range(n_segments):
            start_frame = i * segment_length
            end_frame = min(total_frames, (i + 1) * segment_length)
            
            print(f"Analyzing segment {i+1}/{n_segments} (frames {start_frame}-{end_frame})")
            
            # Extract frames for this segment
            frames_to_analyze = min(30, end_frame - start_frame)  # Limit frames per segment
            frame_indices = np.linspace(start_frame, end_frame-1, frames_to_analyze, dtype=int)
            
            # Extract mouth metrics and timestamps
            mouth_metrics = []
            timestamps = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Calculate timestamp for this frame
                timestamp = frame_idx / fps
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with face mesh
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]  # First face only
                    metrics = self.extract_mouth_metrics(frame, landmarks)
                    
                    if metrics["valid"]:
                        mouth_metrics.append(metrics)
                        timestamps.append(timestamp)
            
            if len(mouth_metrics) < 5:
                segment_results.append({
                    "segment_idx": i,
                    "start_time": start_frame / fps,
                    "end_time": end_frame / fps,
                    "error": "Not enough valid frames",
                    "is_mismatched": False,
                    "sync_score": 0.0
                })
                continue
            
            # Extract audio features for this segment
            audio_features = self.extract_audio_features(audio, audio_sr, timestamps)
            
            # Test different offsets to find the optimal one for this segment
            offsets = np.arange(-0.3, 0.3 + 0.05, 0.05)  # Smaller range for segments
            sync_scores = []
            
            for offset in offsets:
                # Calculate sync score with this offset
                score = self.calculate_sync_score(mouth_metrics, audio_features, offset)
                sync_scores.append(score)
            
            # Find optimal offset for this segment
            best_idx = np.argmax(sync_scores)
            segment_offset = offsets[best_idx]
            max_score = sync_scores[best_idx]
            
            # Calculate mismatch confidence
            mismatch_confidence = 1.0 - max_score
            
            segment_results.append({
                "segment_idx": i,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
                "frames_analyzed": len(mouth_metrics),
                "optimal_offset": segment_offset,
                "sync_score": max_score,
                "is_mismatched": max_score < 0.3,  # Threshold for segment mismatch
                "mismatch_confidence": mismatch_confidence
            })
        
        cap.release()
        
        if not segment_results:
            return {"error": "No valid segments analyzed"}
        
        # Analyze offset consistency across segments
        valid_segments = [s for s in segment_results if "error" not in s]
        
        if not valid_segments:
            return {"error": "No valid segments for consistency analysis"}
        
        offsets = [s["optimal_offset"] for s in valid_segments]
        scores = [s["sync_score"] for s in valid_segments]
        
        # Calculate consistency metrics
        offset_std = np.std(offsets)
        offset_range = max(offsets) - min(offsets)
        
        # Variable offset across segments is suspicious
        offset_consistency = 1.0 - min(1.0, offset_std / 0.2)
        
        # Calculate percentage of mismatched segments
        mismatched_segments = sum(1 for s in valid_segments if s["is_mismatched"])
        mismatch_percentage = mismatched_segments / len(valid_segments)
        
        # Final consistency assessment
        is_consistent = offset_consistency > 0.7 and mismatch_percentage < 0.3
        
        return {
            "video_path": video_path,
            "total_segments": n_segments,
            "valid_segments": len(valid_segments),
            "segment_results": segment_results,
            "offset_consistency": offset_consistency,
            "offset_std": offset_std,
            "offset_range": offset_range,
            "avg_sync_score": np.mean(scores),
            "mismatch_percentage": mismatch_percentage,
            "is_consistent": is_consistent,
            "overall_assessment": "CONSISTENT" if is_consistent else "INCONSISTENT"
        }
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform comprehensive lip-sync mismatch analysis on video
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Dict: Analysis results
        """
        print(f"Analyzing lip-sync in video: {video_path}")
        
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        try:
            # 1. Global offset analysis
            offset_results = self.find_optimal_offset(video_path)
            
            if "error" in offset_results:
                return offset_results
            
            # 2. Consistency analysis across segments
            consistency_results = self.analyze_sync_consistency(video_path)
            
            if "error" in consistency_results:
                print(f"Warning: Consistency analysis failed: {consistency_results['error']}")
                consistency_results = {"is_consistent": True}  # Default to true if analysis fails
            
            # 3. Combine results for final assessment
            mismatch_indicators = 0
            total_indicators = 0
            
            # Indicator 1: Global sync score is poor
            if offset_results["max_sync_score"] < 0.3:
                mismatch_indicators += 1
            total_indicators += 1
            
            # Indicator 2: Sync is inconsistent across segments
            if not consistency_results.get("is_consistent", True):
                mismatch_indicators += 1
            total_indicators += 1
            
            # Indicator 3: Unusually flat correlation curve
            if offset_results["score_distribution"]["unusually_flat"]:
                mismatch_indicators += 1
            total_indicators += 1
            
            # Final mismatch probability
            mismatch_probability = mismatch_indicators / total_indicators
            
            # Binary classification with confidence
            is_mismatched = mismatch_probability > 0.5
            confidence = abs(mismatch_probability - 0.5) * 2  # Scale to [0,1]
            
            return {
                "video_path": video_path,
                "is_mismatched": is_mismatched,
                "mismatch_probability": mismatch_probability,
                "confidence": confidence,
                "mismatch_classification": "MISMATCHED" if is_mismatched else "SYNCHRONIZED",
                "offset_analysis": offset_results,
                "consistency_analysis": consistency_results
            }
            
        except Exception as e:
            return {"error": f"Lip-sync analysis failed: {str(e)}"}
    
    def print_results(self, results: Dict):
        """Print results in a formatted way"""
        print("\n" + "="*60)
        print("LIP-SYNC MISMATCH DETECTION RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"Video: {results['video_path']}")
        print(f"\n🎯 OVERALL ASSESSMENT: {results['mismatch_classification']}")
        print(f"📊 Confidence: {results['confidence']:.3f}")
        print(f"🔢 Mismatch Probability: {results['mismatch_probability']:.3f}")
        
        print("\n🎬 OFFSET ANALYSIS:")
        offset = results['offset_analysis']['optimal_offset']
        print(f"  Optimal Audio-Video Offset: {offset:.3f} seconds")
        print(f"  Maximum Sync Score: {results['offset_analysis']['max_sync_score']:.3f}")
        
        if 'consistency_analysis' in results and 'segment_results' in results['consistency_analysis']:
            print("\n⏱️ CONSISTENCY ANALYSIS:")
            print(f"  Offset Consistency: {results['consistency_analysis']['offset_consistency']:.3f}")
            print(f"  Average Sync Score: {results['consistency_analysis']['avg_sync_score']:.3f}")
            print(f"  Mismatched Segments: {results['consistency_analysis']['mismatch_percentage']*100:.1f}%")
            
            print("\n📊 SEGMENT BREAKDOWN:")
            for segment in results['consistency_analysis']['segment_results'][:5]:  # Show first 5 segments
                if "error" in segment:
                    continue
                start = segment['start_time']
                end = segment['end_time']
                score = segment['sync_score']
                status = "MISMATCHED" if segment['is_mismatched'] else "SYNCED"
                print(f"  {start:>5.1f}s - {end:>5.1f}s: {status:<9} (score: {score:.3f})")
            
            if len(results['consistency_analysis']['segment_results']) > 5:
                print(f"  ... and {len(results['consistency_analysis']['segment_results']) - 5} more segments")
        
        # Print explanation of results
        print("\n" + "-"*60)
        print("ANALYSIS EXPLANATION:")
        if results['is_mismatched']:
            print("  ⚠️ This video shows signs of lip-sync mismatch, which may indicate:")
            print("     - Deepfake manipulation where audio doesn't match lip movements")
            print("     - Poor audio-video synchronization during editing")
            print("     - Dubbed content")
        else:
            print("  ✅ This video shows good synchronization between audio and lip movements")
            print("     - Natural audio-visual correlation is present")
            print("     - Consistent synchronization across video segments")

def main():
    """Example usage of the LipSyncMismatchDetector"""
    
    # Initialize detector
    detector = LipSyncMismatchDetector(use_gpu=True)
    
    print("=== LIP-SYNC MISMATCH DETECTION ===")
    
    # Example video file path - replace with your video path
    video_path = input("Enter path to video file: ").strip()
    if not video_path:
        video_path = "C:\\Users\\Karan Doifode\\Downloads\\real-video1.mp4"  # Default example
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please provide a valid video file path.")
        return
    
    try:
        # Analyze video
        results = detector.analyze_video(video_path)
        detector.print_results(results)
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()