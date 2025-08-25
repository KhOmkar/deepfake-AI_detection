from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
from werkzeug.utils import secure_filename
from src.audio1 import VoiceModulationAnalyzer
from src.video2 import DeepfakeDetector
from src.lipsync import LipSyncMismatchDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize analyzers
audio_analyzer = VoiceModulationAnalyzer()
video_detector = DeepfakeDetector()
lipsync_detector = LipSyncMismatchDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Analyze based on file type
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                results = analyze_video_file(filepath)
            elif file_extension in ['.wav', '.mp3', '.flac']:
                results = analyze_audio_file(filepath)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(results)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

def analyze_video_file(filepath):
    """Analyze video file and return results"""
    results = {
        'file_type': 'video',
        'filename': os.path.basename(filepath)
    }
    
    # Video analysis
    try:
        video_results = video_detector.analyze_video(filepath)
        results['video_analysis'] = video_results
    except Exception as e:
        results['video_analysis'] = {'error': str(e)}
    
    # Audio analysis (if video has audio)
    try:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(filepath) as video:
            if video.audio is not None:
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                
                # Analyze audio
                audio_results = audio_analyzer.analyze_audio(temp_audio_path)
                results['audio_analysis'] = audio_results
                
                # Lip-sync analysis
                lipsync_results = lipsync_detector.analyze_video(filepath)
                results['lipsync_analysis'] = lipsync_results
                
                # Clean up temp audio
                os.remove(temp_audio_path)
            else:
                results['audio_analysis'] = {'message': 'No audio track found'}
                results['lipsync_analysis'] = {'message': 'No audio for lip-sync analysis'}
    except Exception as e:
        results['audio_analysis'] = {'error': str(e)}
        results['lipsync_analysis'] = {'error': str(e)}
    
    return results

def analyze_audio_file(filepath):
    """Analyze audio file and return results"""
    try:
        audio_results = audio_analyzer.analyze_audio(filepath)
        return {
            'file_type': 'audio',
            'filename': os.path.basename(filepath),
            'audio_analysis': audio_results
        }
    except Exception as e:
        return {
            'file_type': 'audio',
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
