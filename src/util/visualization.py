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
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"'
    os.system(command)
    return audio_path

def main():
    print("=== DEEPFAKE DETECTION ANALYSIS ===")
    file_path = input("Enter path to video or audio file: ").strip()

    if not os.path.exists(file_path):
        print("File not found. Please provide a valid file path.")
        return

    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Analyzing video...")
        audio_present = False

        # Check if audio is present in the video
        cap = cv2.VideoCapture(file_path)
        audio_present = cap.get(cv2.CAP_PROP_AUDIO_STREAM_COUNT) > 0
        cap.release()

        if audio_present:
            print("Audio detected in video. Extracting audio...")
            audio_path = extract_audio_from_video(file_path)
            analyze_audio(audio_path)
        else:
            print("No audio detected in video. Performing video analysis only.")
        
        analyze_video(file_path)

    elif file_path.lower().endswith(('.wav', '.mp3', '.flac')):
        print("Analyzing audio file...")
        analyze_audio(file_path)
        print("No data for video and lip-sync analysis.")

    else:
        print("Unsupported file format. Please provide a video or audio file.")

if __name__ == "__main__":
    main()