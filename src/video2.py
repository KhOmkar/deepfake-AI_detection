import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

class DeepfakeDetector:
    def __init__(self):
        """
        Streamlined deepfake detector using only the best performing model
        Model: dima806/deepfake_vs_real_image_detection
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        
        # Load the best performing model
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the deepfake detection model"""
        print("Loading deepfake detection model...")
        
        try:
            self.model = pipeline(
                "image-classification", 
                model="dima806/deepfake_vs_real_image_detection",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Model loaded successfully")
            
            # Print model information
            if hasattr(self.model.model, 'config') and hasattr(self.model.model.config, 'id2label'):
                labels = self.model.model.config.id2label
                print(f"Model labels: {labels}")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None
    
    def detect_face(self, frame):
        """Detect and extract face region using MediaPipe"""
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            h, w = frame.shape[:2]
            detection = results.detections[0]  # Get the most confident detection
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate bounding box coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding around the face
            padding = 0.2
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + width + pad_x)
            y2 = min(h, y + height + pad_y)
            
            return rgb_frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        
        return None, None
    
    def predict(self, face_region):
        """Predict if face region is fake or real"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(face_region)
            
            # Get prediction
            result = self.model(pil_image)
            
            # Parse results
            fake_score = 0.0
            real_score = 0.0
            
            for pred in result:
                label = pred['label'].lower()
                score = pred['score']
                
                if 'fake' in label or 'deepfake' in label or 'generated' in label:
                    fake_score = max(fake_score, score)
                elif 'real' in label or 'authentic' in label or 'original' in label:
                    real_score = max(real_score, score)
            
            # Determine final prediction
            is_fake = fake_score > real_score
            confidence = max(fake_score, real_score)
            
            return {
                "is_fake": is_fake,
                "prediction": "FAKE" if is_fake else "REAL",
                "fake_score": fake_score,
                "real_score": real_score,
                "confidence": confidence,
                "raw_results": result
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    

    
    def analyze_video(self, video_path, max_frames=30, start_position=0.0):
        """
        Analyze video for deepfake detection using consecutive frames
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of consecutive frames to analyze
            start_position: Position in video to start analysis (0.0-1.0)
        """
        print(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate starting frame
        start_frame = int(total_frames * start_position)
        end_frame = min(start_frame + max_frames, total_frames)
        
        print(f"Analyzing {end_frame - start_frame} consecutive frames starting at frame #{start_frame}")
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_results = []
        faces_detected = 0
        current_frame = start_frame
        
        while len(frame_results) < max_frames and current_frame < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            print(f"Processing frame {current_frame - start_frame + 1}/{min(max_frames, end_frame - start_frame)} (frame #{current_frame})")
            
            # Detect face
            face_region, bbox = self.detect_face(frame)
            if face_region is None:
                # If no face detected, still increment frame counter
                current_frame += 1
                continue
            
            faces_detected += 1
            
            # Get prediction for this frame
            prediction = self.predict(face_region)
            
            frame_result = {
                "frame_index": current_frame,
                "timestamp": current_frame / fps if fps > 0 else 0,
                **prediction
            }
            frame_results.append(frame_result)
            current_frame += 1
        
        cap.release()
        
        if not frame_results:
            return {"error": "No faces detected in any frames"}
        
        # Aggregate results across all frames
        aggregated = self.aggregate_results(frame_results)
        
        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "analyzed_frames": len(frame_results),
            "faces_detected": faces_detected,
            "fps": fps,
            "start_frame": start_frame,
            "frame_results": frame_results,
            **aggregated
        }
    
    def aggregate_results(self, frame_results):
        """Aggregate predictions from multiple consecutive frames with temporal analysis"""
        fake_scores = []
        real_scores = []
        predictions = []
        
        for result in frame_results:
            if 'error' not in result:
                fake_scores.append(result['fake_score'])
                real_scores.append(result['real_score'])
                predictions.append(result['is_fake'])
        
        if not fake_scores:
            return {"error": "No valid predictions"}
        
        # Calculate aggregated metrics
        avg_fake_score = np.mean(fake_scores)
        avg_real_score = np.mean(real_scores)
        median_fake_score = np.median(fake_scores)
        median_real_score = np.median(real_scores)
        
        # Final prediction based on average scores
        final_is_fake = avg_fake_score > avg_real_score
        final_confidence = max(avg_fake_score, avg_real_score)
        
        # If prediction is fake but confidence is low, change to real
        if final_is_fake and final_confidence < 0.600:
            final_is_fake = False
            print("Low confidence fake detection (< 0.600). Changing prediction to REAL.")
        
        # Calculate frame consistency (percentage of frames with same prediction as final)
        frame_consistency = sum(1 for pred in predictions if pred == final_is_fake) / len(predictions)
        
        # Calculate temporal consistency (changes between consecutive frames)
        temporal_changes = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i-1])
        temporal_consistency = 1.0 - (temporal_changes / (len(predictions) - 1)) if len(predictions) > 1 else 1.0
        
        # Analyze score stability across consecutive frames
        score_stability = 1.0 - np.mean([abs(fake_scores[i] - fake_scores[i-1]) for i in range(1, len(fake_scores))])
        
        # Calculate frame-level statistics
        fake_frame_count = sum(predictions)
        real_frame_count = len(predictions) - fake_frame_count
        
        return {
            "final_prediction": "FAKE" if final_is_fake else "REAL",
            "is_fake": final_is_fake,
            "confidence": final_confidence,
            "frame_consistency": frame_consistency,
            "temporal_consistency": temporal_consistency,
            "score_stability": score_stability,
            "avg_fake_score": avg_fake_score,
            "avg_real_score": avg_real_score,
            "median_fake_score": median_fake_score,
            "median_real_score": median_real_score,
            "fake_frames": fake_frame_count,
            "real_frames": real_frame_count,
            "total_analyzed_frames": len(predictions)
        }
    
    def analyze_realtime(self, source=0, display=True):
        """Real-time deepfake detection from webcam or video stream"""
        print(f"Starting real-time analysis from source: {source}")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect face
                face_region, bbox = self.detect_face(frame)
                
                if face_region is not None and bbox is not None:
                    # Get prediction
                    prediction = self.predict(face_region)
                    
                    if 'error' not in prediction:
                        # Draw bounding box and prediction on frame
                        x1, y1, x2, y2 = bbox
                        color = (0, 0, 255) if prediction['is_fake'] else (0, 255, 0)  # Red for fake, green for real
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add prediction text
                        label = f"{prediction['prediction']}: {prediction['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if display:
                    cv2.imshow('Deepfake Detection', frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
    def print_results(self, results):
        """Print results in a formatted way"""
        print("\n" + "="*60)
        print("DEEPFAKE DETECTION RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        if "video_path" in results:
            # Video results
            print(f"Video: {results['video_path']}")
            print(f"Total frames: {results['total_frames']}")
            print(f"Analyzed frames: {results['analyzed_frames']}")
            print(f"Faces detected: {results['faces_detected']}")
            print(f"FPS: {results.get('fps', 0):.1f}")
            
            if "start_frame" in results:
                print(f"Starting frame: {results['start_frame']}")
            
            print(f"\nFinal Prediction: {results['final_prediction']}")
            print(f"Confidence: {results['confidence']:.3f}")
            print(f"Frame Consistency: {results['frame_consistency']:.3f}")
            print(f"Temporal Consistency: {results['temporal_consistency']:.3f}")
            print(f"Score Stability: {results['score_stability']:.3f}")
            print(f"Average Fake Score: {results['avg_fake_score']:.3f}")
            print(f"Average Real Score: {results['avg_real_score']:.3f}")
            print(f"Fake Frames: {results['fake_frames']}/{results['total_analyzed_frames']}")
            print(f"Real Frames: {results['real_frames']}/{results['total_analyzed_frames']}")


def main():
    """Example usage of the DeepfakeDetector for video and real-time analysis"""
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    if detector.model is None:
        print("Failed to initialize detector. Exiting.")
        return
    
    # Example 1: Analyze video
    print("=== VIDEO ANALYSIS ===")
    video_path = "C:\\Users\\Karan Doifode\\Downloads\\ai8.mp4"  # Replace with your video path
    try:
        # Analyze consecutive frames (30 frames from the 10% mark in the video)
        video_results = detector.analyze_video(video_path, max_frames=40, start_position=0.1)
        detector.print_results(video_results)
    except Exception as e:
        print(f"Video analysis failed: {e}")
    
    # Example 2: Real-time detection (uncomment to use)
    # print("\n=== REAL-TIME ANALYSIS ===")
    # print("Press 'q' to quit real-time detection")
    # try:
    #     detector.analyze_realtime(source=0)  # 0 for webcam, or path to video file
    # except Exception as e:
    #     print(f"Real-time analysis failed: {e}")


if __name__ == "__main__":
    main()