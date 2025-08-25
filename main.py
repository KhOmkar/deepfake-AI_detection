# filepath: c:\Users\Karan Doifode\Desktop\kurukshetra\deepfake-detector\main.py

import os
import librosa
from moviepy.editor import VideoFileClip
import tempfile
from audio1 import VoiceModulationAnalyzer
from video2 import DeepfakeDetector
from lipsync import LipSyncMismatchDetector

def main():
    """Main function to analyze video and audio files for deepfake detection."""
    
    # Get user input for video or audio file
    file_path = input("Enter the path to the video or audio file: ").strip()
    
    if not os.path.exists(file_path):
        print("File not found. Please provide a valid file path.")
        return
    
    # Check if the file is a video or audio file
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # It's a video file
        print("Analyzing video file...")
        
        # Initialize video detector
        video_detector = DeepfakeDetector()
        
        # Analyze video
        video_results = video_detector.analyze_video(file_path)
        video_detector.print_results(video_results)
        
        # Check for audio presence using moviepy
        audio_present = False
        temp_audio_path = None
        try:
            with VideoFileClip(file_path) as video:
                if video.audio is not None:
                    # Save audio to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                    audio_present = True
        except Exception as e:
            print(f"Could not extract audio from video: {e}")

        if audio_present and temp_audio_path:
            # Perform audio analysis on extracted audio
            audio_analyzer = VoiceModulationAnalyzer()
            audio_results = audio_analyzer.analyze_audio(temp_audio_path)
            #audio_analyzer.print_results(audio_results)
            audio_analyzer.print_modulation_analysis(audio_results)

            # Perform lip-sync analysis on the original video file
            lipsync_detector = LipSyncMismatchDetector()
            lipsync_results = lipsync_detector.analyze_video(file_path)
            lipsync_detector.print_results(lipsync_results)

            # Clean up temporary audio file
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
        else:
            print("No audio data found for lip-sync analysis.")
    
    elif file_extension in ['.wav', '.mp3', '.flac']:
        # It's an audio file
        print("Analyzing audio file...")
        
        # Initialize audio analyzer
        audio_analyzer = VoiceModulationAnalyzer()
        
        # Perform audio analysis
        audio_results = audio_analyzer.analyze_audio(file_path)
        audio_analyzer.print_results(audio_results)
        audio_analyzer.print_modulation_analysis(audio_results)
        
        print("No video data found for deepfake detection and lip-sync analysis.")
    
    else:
        print("Unsupported file type. Please provide a video or audio file.")

if __name__ == "__main__":
    main()