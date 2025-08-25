# filepath: c:\Users\Karan Doifode\Desktop\kurukshetra\deepfake-detector\main.py

import os
import librosa
from audio1 import VoiceModulationAnalyzer
from video2 import DeepfakeDetector
from lipsync import LipSyncMismatchDetector

def analyze_video(video_path):
    detector = DeepfakeDetector()
    results = detector.analyze_video(video_path)
    detector.print_results(results)
    return results

def analyze_audio(audio_path):
    analyzer = VoiceModulationAnalyzer()
    results = analyzer.analyze_audio(audio_path)
    analyzer.print_results(results)
    analyzer.print_modulation_analysis(results)
    return results

def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"')
    return audio_path

def main():
    print("=== DEEPFAKE DETECTION SYSTEM ===")
    
    file_path = input("Enter path to video or audio file: ").strip()
    
    if not os.path.exists(file_path):
        print("File not found. Please provide a valid file path.")
        return
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        print("Analyzing video...")
        audio_path = extract_audio_from_video(file_path)
        
        if os.path.exists(audio_path):
            print("Audio extracted from video. Analyzing audio...")
            analyze_audio(audio_path)
        else:
            print("No audio found in the video. Performing video detection only.")
            analyze_video(file_path)
    
    elif file_extension in ['.wav', '.mp3', '.aac']:
        print("Analyzing audio...")
        analyze_audio(file_path)
    else:
        print("Unsupported file type. Please provide a video or audio file.")

if __name__ == "__main__":
    main()