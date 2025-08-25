import librosa
import numpy as np
import warnings
from collections import Counter
import os
from typing import Tuple, List, Dict, Optional
from scipy import signal

warnings.filterwarnings('ignore')

class VoiceModulationAnalyzer:
    def __init__(self):
        """
        Audio deepfake detector focused only on voice modulation analysis
        """
        # Audio processing parameters
        self.sample_rate = 16000  # Standard rate for speech processing
        self.hop_length = 512
        self.segment_duration = 3.0  # Seconds per segment
        self.overlap = 0.5  # Overlap between segments
        
        print("Voice Modulation Analyzer initialized")
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            if len(audio) == 0:
                raise ValueError("Audio file is empty or only contains silence")
            
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")
    
    def segment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Segment audio into overlapping chunks
        
        Args:
            audio (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            List[np.ndarray]: List of audio segments
        """
        audio_duration = len(audio) / sr
        segment_length = int(self.segment_duration * sr)
        
        # Handle short audio files
        if audio_duration < self.segment_duration:
            print(f"Audio is short ({audio_duration:.2f}s), analyzing as single segment")
            # For very short audio, use the entire audio as one segment
            if audio_duration >= 0.1:  # At least 0.1 seconds
                # Pad to minimum length for stable feature extraction
                min_length = int(0.5 * sr)  # 0.5 second minimum
                if len(audio) < min_length:
                    padded_audio = np.pad(audio, (0, min_length - len(audio)), 'constant')
                    return [padded_audio]
                else:
                    return [audio]
            else:
                return []  # Too short to analyze
        
        # Normal segmentation for longer audio
        hop_length = int(segment_length * (1 - self.overlap))
        segments = []
        start = 0
        
        while start + segment_length <= len(audio):
            segment = audio[start:start + segment_length]
            segments.append(segment)
            start += hop_length
        
        # Add the last segment if there's remaining audio
        if start < len(audio):
            remaining = audio[start:]
            if len(remaining) >= segment_length // 2:  # Only if it's at least half the segment length
                # Pad with zeros if necessary
                if len(remaining) < segment_length:
                    padded = np.pad(remaining, (0, segment_length - len(remaining)), 'constant')
                    segments.append(padded)
                else:
                    segments.append(remaining)
        
        return segments
    
    def analyze_voice_modulation(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Analyze voice modulation characteristics to detect synthetic patterns
        
        Args:
            audio (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            Dict: Voice modulation features
        """
        features = {}
        
        try:
            # 1. Extract pitch (F0) contour with higher precision
            hop_length = 256
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=65,  # Lower bound for typical human voice
                fmax=400,  # Upper bound for typical human voice
                hop_length=hop_length
            )
            
            # Calculate timestamps for each F0 value
            times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
            
            # Filter out NaN values (unvoiced frames)
            valid_f0_indices = ~np.isnan(f0)
            valid_f0 = f0[valid_f0_indices]
            valid_times = times[valid_f0_indices]
            
            if len(valid_f0) > 1:
                # 2. Analyze pitch variation and stability
                f0_diff = np.diff(valid_f0)
                features['pitch_jump_rate'] = np.sum(np.abs(f0_diff) > 20) / len(f0_diff)
                features['pitch_stability'] = 1.0 / (1.0 + np.std(f0_diff))
                
                # 3. Calculate jitter (cycle-to-cycle pitch variation)
                if len(valid_f0) > 2:
                    jitter_samples = np.abs(np.diff(valid_f0))
                    features['jitter_mean'] = np.mean(jitter_samples)
                    features['jitter_std'] = np.std(jitter_samples)
                    
                    # Normalized jitter - abnormally low in synthetic speech
                    features['jitter_normalized'] = features['jitter_mean'] / (np.mean(valid_f0) + 1e-8)
                else:
                    features['jitter_mean'] = 0
                    features['jitter_std'] = 0
                    features['jitter_normalized'] = 0
                
                # 4. Detect pitch modulation rate
                if len(valid_f0) > 5:
                    # Detrend the pitch contour
                    detrended_f0 = signal.detrend(valid_f0)
                    
                    # Get modulation spectrum using FFT
                    modulation_spec = np.abs(np.fft.rfft(detrended_f0))
                    
                    # Synthetic voices often lack natural micro-modulations
                    # or have too-regular modulations
                    features['mod_energy_low'] = np.sum(modulation_spec[:len(modulation_spec)//10])
                    features['mod_energy_high'] = np.sum(modulation_spec[len(modulation_spec)//10:])
                    features['mod_ratio'] = features['mod_energy_high'] / (features['mod_energy_low'] + 1e-8)
                else:
                    features['mod_energy_low'] = 0
                    features['mod_energy_high'] = 0
                    features['mod_ratio'] = 0
                
                # 5. Check for speech rate consistency
                if len(valid_times) > 1:
                    # Calculate time intervals between voiced segments
                    intervals = np.diff(valid_times)
                    features['speech_rate_std'] = np.std(intervals)
                    features['speech_rate_uniformity'] = np.mean(intervals) / (np.std(intervals) + 1e-8)
                else:
                    features['speech_rate_std'] = 0
                    features['speech_rate_uniformity'] = 0
            else:
                # Not enough pitch information
                features['pitch_jump_rate'] = 0
                features['pitch_stability'] = 0
                features['jitter_mean'] = 0
                features['jitter_std'] = 0
                features['jitter_normalized'] = 0
                features['mod_energy_low'] = 0
                features['mod_energy_high'] = 0
                features['mod_ratio'] = 0
                features['speech_rate_std'] = 0
                features['speech_rate_uniformity'] = 0
            
            # 6. Analyze amplitude modulation (shimmer)
            # Get the amplitude envelope
            amplitude_envelope = np.abs(librosa.stft(audio, hop_length=hop_length))
            amplitude_envelope = np.mean(amplitude_envelope, axis=0)
            
            if len(amplitude_envelope) > 1:
                # Calculate shimmer (amplitude variation)
                shimmer = np.abs(np.diff(amplitude_envelope))
                features['shimmer_mean'] = np.mean(shimmer)
                features['shimmer_std'] = np.std(shimmer)
                
                # Normalized shimmer
                features['shimmer_normalized'] = features['shimmer_mean'] / (np.mean(amplitude_envelope) + 1e-8)
            else:
                features['shimmer_mean'] = 0
                features['shimmer_std'] = 0
                features['shimmer_normalized'] = 0
                
        except Exception as e:
            print(f"Error analyzing voice modulation: {e}")
            # Set default values for all features
            features = {
                'pitch_jump_rate': 0,
                'pitch_stability': 0,
                'jitter_mean': 0,
                'jitter_std': 0,
                'jitter_normalized': 0,
                'mod_energy_low': 0,
                'mod_energy_high': 0,
                'mod_ratio': 0,
                'speech_rate_std': 0,
                'speech_rate_uniformity': 0,
                'shimmer_mean': 0,
                'shimmer_std': 0,
                'shimmer_normalized': 0
            }
        
        return features
    
    def predict_segment(self, audio_segment: np.ndarray, sr: int) -> Dict:
        """
        Predict if audio segment is fake or real based on modulation analysis
        
        Args:
            audio_segment (np.ndarray): Audio segment
            sr (int): Sample rate
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Extract only modulation features - the key focus
            modulation_features = self.analyze_voice_modulation(audio_segment, sr)
            
            # Initialize score tracking
            fake_indicators = 0
            total_indicators = 0
            
            # Modulation indicators 
            if 'jitter_normalized' in modulation_features:
                # Abnormally low jitter is common in synthetic speech
                if modulation_features['jitter_normalized'] < 0.005:
                    fake_indicators += 2
                total_indicators += 2
            
            if 'shimmer_normalized' in modulation_features:
                # Abnormally low shimmer is common in synthetic speech
                if modulation_features['shimmer_normalized'] < 0.02:
                    fake_indicators += 2
                total_indicators += 2
            
            if 'pitch_stability' in modulation_features:
                # Too stable pitch is suspicious
                if modulation_features['pitch_stability'] > 10:
                    fake_indicators += 2
                total_indicators += 2
            
            if 'mod_ratio' in modulation_features:
                # Unusual modulation spectrum ratio
                if modulation_features['mod_ratio'] < 0.1 or modulation_features['mod_ratio'] > 10:
                    fake_indicators += 2
                total_indicators += 2
                
            if 'speech_rate_uniformity' in modulation_features:
                # Too uniform speech rate
                if modulation_features['speech_rate_uniformity'] > 15:
                    fake_indicators += 1
                total_indicators += 1
            
            # Pitch jump rate - natural speech has some jumps
            if 'pitch_jump_rate' in modulation_features:
                if modulation_features['pitch_jump_rate'] < 0.05:
                    fake_indicators += 1
                total_indicators += 1
            
            # Calculate fake probability
            if total_indicators > 0:
                fake_score = fake_indicators / total_indicators
            else:
                fake_score = 0.5  # Neutral if no indicators
            
            real_score = 1 - fake_score
            is_fake = fake_score > 0.5
            confidence = max(fake_score, real_score)
            
            return {
                "is_fake": is_fake,
                "prediction": "FAKE" if is_fake else "REAL",
                "fake_score": fake_score,
                "real_score": real_score,
                "confidence": confidence,
                "features": modulation_features,
                "fake_indicators": fake_indicators,
                "total_indicators": total_indicators
            }
            
        except Exception as e:
            return {"error": f"Segment prediction failed: {str(e)}"}
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze audio file for deepfake detection
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            Dict: Analysis results
        """
        print(f"Analyzing audio: {audio_path}")
        
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        try:
            # Preprocess audio
            audio, sr = self.preprocess_audio(audio_path)
            duration = len(audio) / sr
            
            print(f"Audio duration: {duration:.2f} seconds")
            print(f"Sample rate: {sr} Hz")
            
            # Segment audio
            segments = self.segment_audio(audio, sr)
            print(f"Created {len(segments)} segments for analysis")
            
            if not segments:
                return {"error": "No valid audio segments found"}
            
            # Analyze each segment
            segment_results = []
            for i, segment in enumerate(segments):
                print(f"Analyzing segment {i+1}/{len(segments)}")
                result = self.predict_segment(segment, sr)
                
                if "error" not in result:
                    segment_result = {
                        "segment_index": i,
                        "start_time": i * self.segment_duration * (1 - self.overlap),
                        "end_time": (i * self.segment_duration * (1 - self.overlap)) + self.segment_duration,
                        **result
                    }
                    segment_results.append(segment_result)
                else:
                    print(f"Error in segment {i}: {result['error']}")
            
            if not segment_results:
                return {"error": "No valid segment predictions"}
            
            # Aggregate results
            aggregated = self.aggregate_results(segment_results)
            
            # Add analysis type info
            analysis_type = "single_segment" if duration < self.segment_duration else "multi_segment"
            
            return {
                "audio_path": audio_path,
                "duration": duration,
                "sample_rate": sr,
                "analysis_type": analysis_type,
                "total_segments": len(segments),
                "analyzed_segments": len(segment_results),
                "segment_results": segment_results,
                **aggregated
            }
            
        except Exception as e:
            return {"error": f"Audio analysis failed: {str(e)}"}
    
    def aggregate_results(self, segment_results: List[Dict]) -> Dict:
        """
        Aggregate predictions from multiple audio segments
        
        Args:
            segment_results (List[Dict]): Results from individual segments
            
        Returns:
            Dict: Aggregated results
        """
        fake_scores = []
        real_scores = []
        predictions = []
        confidences = []
        
        for result in segment_results:
            if 'error' not in result:
                fake_scores.append(result['fake_score'])
                real_scores.append(result['real_score'])
                predictions.append(result['is_fake'])
                confidences.append(result['confidence'])
        
        if not fake_scores:
            return {"error": "No valid predictions for aggregation"}
        
        # Calculate aggregated metrics
        avg_fake_score = np.mean(fake_scores)
        avg_real_score = np.mean(real_scores)
        avg_confidence = np.mean(confidences)
        
        # Final prediction based on average scores
        final_is_fake = avg_fake_score > avg_real_score
        final_confidence = max(avg_fake_score, avg_real_score)
        
        # Calculate consistency (how many segments agree with final prediction)
        consistency = sum(1 for pred in predictions if pred == final_is_fake) / len(predictions)
        
        # Calculate segment statistics
        fake_segment_count = sum(predictions)
        real_segment_count = len(predictions) - fake_segment_count
        
        # Risk assessment
        risk_level = "LOW"
        if final_is_fake and final_confidence > 0.8 and consistency > 0.7:
            risk_level = "HIGH"
        elif final_is_fake and (final_confidence > 0.6 or consistency > 0.6):
            risk_level = "MEDIUM"
        
        return {
            "final_prediction": "FAKE" if final_is_fake else "REAL",
            "is_fake": final_is_fake,
            "confidence": final_confidence,
            "consistency": consistency,
            "risk_level": risk_level,
            "avg_fake_score": avg_fake_score,
            "avg_real_score": avg_real_score,
            "avg_confidence": avg_confidence,
            "fake_segments": fake_segment_count,
            "real_segments": real_segment_count,
            "total_analyzed_segments": len(predictions)
        }
    
    def print_results(self, results: Dict):
        """Print results in a formatted way"""
        print("\n" + "="*60)
        print("AUDIO DEEPFAKE DETECTION RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"Audio File: {results['audio_path']}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Sample Rate: {results['sample_rate']} Hz")
        print(f"Analysis Type: {results.get('analysis_type', 'unknown').replace('_', ' ').title()}")
        print(f"Total Segments: {results['total_segments']}")
        print(f"Analyzed Segments: {results['analyzed_segments']}")
        
        print(f"\n🎯 FINAL PREDICTION: {results['final_prediction']}")
        print(f"📊 Confidence: {results['confidence']:.3f}")
        print(f"🔄 Consistency: {results['consistency']:.3f}")
        print(f"⚠️  Risk Level: {results['risk_level']}")
        
        print(f"\n📈 DETAILED SCORES:")
        print(f"Average Fake Score: {results['avg_fake_score']:.3f}")
        print(f"Average Real Score: {results['avg_real_score']:.3f}")
        
        print(f"\n📊 SEGMENT BREAKDOWN:")
        print(f"Fake Segments: {results['fake_segments']}/{results['total_analyzed_segments']}")
        print(f"Real Segments: {results['real_segments']}/{results['total_analyzed_segments']}")
        
        # Show per-segment timeline
        if results.get('segment_results'):
            print(f"\n⏰ TIMELINE ANALYSIS:")
            for result in results['segment_results'][:5]:  # Show first 5 segments
                start = result['start_time']
                end = result['end_time']
                pred = result['prediction']
                conf = result['confidence']
                print(f"  {start:>5.1f}s - {end:>5.1f}s: {pred:<4} (confidence: {conf:.3f})")
            
            if len(results['segment_results']) > 5:
                print(f"  ... and {len(results['segment_results']) - 5} more segments")
    
    def print_modulation_analysis(self, results):
        """Print detailed voice modulation analysis with focus on abnormal patterns"""
        if 'segment_results' not in results or not results['segment_results']:
            return
        
        print("\n" + "="*60)
        print("🎙️ VOICE MODULATION ANALYSIS")
        print("="*60)
        
        # Get average modulation features across segments
        mod_features = {}
        for segment in results['segment_results']:
            if 'features' in segment:
                for key, value in segment['features'].items():
                    if key in mod_features:
                        mod_features[key].append(value)
                    else:
                        mod_features[key] = [value]
        
        # Calculate average values
        avg_features = {}
        for key, values in mod_features.items():
            if values:
                avg_features[key] = np.mean(values)
        
        # Group features for better readability
        feature_groups = {
            "PITCH VARIATION": ["pitch_stability", "pitch_jump_rate"],
            "JITTER ANALYSIS": ["jitter_normalized", "jitter_mean", "jitter_std"],
            "AMPLITUDE MODULATION": ["shimmer_normalized", "shimmer_mean", "shimmer_std"],
            "MODULATION SPECTRUM": ["mod_ratio", "mod_energy_low", "mod_energy_high"],
            "SPEECH TIMING": ["speech_rate_uniformity", "speech_rate_std"]
        }
        
        # Print feature groups
        for group_name, feature_keys in feature_groups.items():
            print(f"\n{group_name}:")
            for key in feature_keys:
                if key in avg_features:
                    print(f"  {key}: {avg_features[key]:.6f}")
                    
                    # Add reference values for comparison
                    if key == "jitter_normalized":
                        print(f"    → Natural speech typically: 0.008-0.02")
                        print(f"    → Synthetic often: <0.005")
                    elif key == "shimmer_normalized":
                        print(f"    → Natural speech typically: 0.04-0.08")
                        print(f"    → Synthetic often: <0.02")
                    elif key == "pitch_stability":
                        print(f"    → Natural speech typically: 1.0-8.0")
                        print(f"    → Synthetic often: >10.0")
                    elif key == "mod_ratio":
                        print(f"    → Natural speech typically: 0.3-3.0")
                        print(f"    → Synthetic often: <0.1 or >10.0")
        
        # Print overall assessment
        print("\n" + "-"*60)
        print("ABNORMAL MODULATION ASSESSMENT:")
        
        # Count abnormal indicators
        abnormal_count = 0
        total_indicators = 0
        
        indicators = [
            (avg_features.get('jitter_normalized', 1.0) < 0.005, "Abnormally low jitter"),
            (avg_features.get('shimmer_normalized', 1.0) < 0.02, "Abnormally low shimmer"),
            (avg_features.get('pitch_stability', 0) > 10, "Suspiciously stable pitch"),
            (avg_features.get('mod_ratio', 1.0) < 0.1 or avg_features.get('mod_ratio', 1.0) > 10, "Unusual modulation spectrum"),
            (avg_features.get('speech_rate_uniformity', 0) > 15, "Mechanically uniform speech rate"),
            (avg_features.get('pitch_jump_rate', 0.1) < 0.05, "Suspiciously smooth pitch transitions")
        ]
        
        for is_abnormal, description in indicators:
            if is_abnormal:
                print(f"  ⚠️ {description}")
                abnormal_count += 1
            total_indicators += 1
        
        # Final assessment
        if total_indicators > 0:
            abnormal_score = abnormal_count / total_indicators
            print(f"\nAbnormal modulation score: {abnormal_score:.2f}")
            
            if abnormal_score > 0.5:
                print("CONCLUSION: HIGH PROBABILITY OF SYNTHETIC VOICE")
            elif abnormal_score > 0.3:
                print("CONCLUSION: MODERATE INDICATORS OF SYNTHETIC VOICE")
            else:
                print("CONCLUSION: LIKELY NATURAL VOICE MODULATION")

def main():
    """Example usage of the VoiceModulationAnalyzer"""
    
    # Initialize analyzer
    analyzer = VoiceModulationAnalyzer()
    
    print("=== AUDIO DEEPFAKE ANALYSIS ===")
    
    # Example audio file path - replace with your audio file
    audio_path = input("Enter path to audio file: ").strip()
    if not audio_path:
        audio_path = "C:\\Users\\Karan Doifode\\Downloads\\audio_df3.wav"  # Default example

    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        print("Please provide a valid audio file path.")
        return
    
    try:
        # Analyze audio
        results = analyzer.analyze_audio(audio_path)
        analyzer.print_results(results)
        analyzer.print_modulation_analysis(results)
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()