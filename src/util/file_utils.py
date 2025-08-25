# filepath: c:\Users\Karan Doifode\Desktop\kurukshetra\deepfake-detector\main.py

import os
import cv2
import librosa
import numpy as np
from audio1 import VoiceModulationAnalyzer
from video2 import DeepfakeDetector
from lipsync import LipSyncMismatchDetector

def analyze_video(video_path):
    detector = DeepfakeDetector()
    results = detector.analyze_video(video_path)
    return results

def analyze_audio(audio_path):
    analyzer = VoiceModulationAnalyzer()
    results = analyzer.analyze_audio(audio_path)
    return results

def extract_audio_from_video(video_path):
    audio, sr = librosa.load(video_path, sr=None, mono=True)
    return audio, sr

def main():
    print("=== DEEPFAKE ANALYSIS ===")
    
    file_path = input("Enter path to video or audio file: ").strip()
    
    if not os.path.exists(file_path):
        print("File not found. Please provide a valid file path.")
        return
    
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Analyzing video...")
        video_results = analyze_video(file_path)
        
        # Check for audio in the video
        audio, sr = extract_audio_from_video(file_path)
        if len(audio) > 0:
            print("Audio detected in video. Performing audio analysis...")
            audio_results = analyze_audio(file_path)
            print("Audio analysis results:")
            print(audio_results)
        else:
            print("No audio detected in video. Skipping audio analysis.")
        
        print("Video analysis results:")
        print(video_results)
    
    elif file_path.lower().endswith(('.wav', '.mp3', '.flac')):
        print("Analyzing audio...")
        audio_results = analyze_audio(file_path)
        print("Audio analysis results:")
        print(audio_results)
        print("No video data available for lip-sync analysis.")
    
    else:
        print("Unsupported file type. Please provide a video or audio file.")

if __name__ == "__main__":
    main()