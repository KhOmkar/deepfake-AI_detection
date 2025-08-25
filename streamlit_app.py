import streamlit as st
import os
import tempfile
import numpy as np
from src.audio1 import VoiceModulationAnalyzer
from src.video2 import DeepfakeDetector
from src.lipsync import LipSyncMismatchDetector
from moviepy.editor import VideoFileClip

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Add minimal CSS for better output display
st.markdown("""
<style>
    .result-card {
        background-color: rgba(28, 30, 33, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #FF4B4B;
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 15px;
    }
    .metric-item {
        flex: 1;
        background-color: rgba(28, 30, 33, 0.1);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        min-width: 120px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.8;
    }
    .conclusion-box {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .conclusion-fake {
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    .conclusion-real {
        background-color: rgba(45, 211, 111, 0.1);
        border: 1px solid rgba(45, 211, 111, 0.3);
    }
    .conclusion-warning {
        background-color: rgba(255, 196, 9, 0.1);
        border: 1px solid rgba(255, 196, 9, 0.3);
    }
    .details-table {
        width: 100%;
        border-collapse: collapse;
    }
    .details-table tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.05);
    }
    .details-table td, .details-table th {
        padding: 8px;
        text-align: left;
    }
    .details-table th {
        background-color: rgba(255, 75, 75, 0.1);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize analyzers
@st.cache_resource
def load_analyzers():
    return {
        'audio': VoiceModulationAnalyzer(),
        'video': DeepfakeDetector(),
        'lipsync': LipSyncMismatchDetector()
    }

analyzers = load_analyzers()

# Display functions - moved to top to avoid "name not defined" error
def display_audio_results(audio_results):
    if 'error' in audio_results:
        st.error(f"Audio analysis error: {audio_results['error']}")
        return
    
    # Main audio modulation analysis results
    st.subheader("🎙️ Voice Modulation Analysis")
    
    # Show audio metrics with better styling
    st.markdown("<h4>Audio Properties</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{audio_results.get('duration', 0):.2f}s</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Sample Rate</div>
            <div class="metric-value">{audio_results.get('sample_rate', 0)} Hz</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Segments</div>
            <div class="metric-value">{audio_results.get('total_segments', 0)}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Analysis Type</div>
            <div class="metric-value">{audio_results.get('analysis_type', 'Unknown')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show abnormal modulation score and conclusion
    if 'segment_results' in audio_results and audio_results['segment_results']:
        # Calculate average modulation features across segments
        mod_features = {}
        for segment in audio_results['segment_results']:
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
        
        # Display detected abnormalities with better styling
        st.markdown("<h4>Detected Abnormalities</h4>", unsafe_allow_html=True)
        
        abnormal_found = False
        for is_abnormal, description in indicators:
            if is_abnormal:
                st.markdown(f"""
                <div style="background-color: rgba(255, 196, 9, 0.1); margin: 5px 0; padding: 10px; 
                           border-radius: 5px; border-left: 3px solid #FFC409;">
                    ⚠️ <strong>{description}</strong>
                </div>
                """, unsafe_allow_html=True)
                abnormal_count += 1
                abnormal_found = True
            total_indicators += 1
        
        if not abnormal_found:
            st.markdown(f"""
            <div style="background-color: rgba(45, 211, 111, 0.1); margin: 5px 0; padding: 10px; 
                       border-radius: 5px; border-left: 3px solid #2DD36F;">
                ✅ <strong>No significant abnormalities detected</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate abnormal modulation score
        if total_indicators > 0:
            abnormal_score = abnormal_count / total_indicators
            
            # Display abnormal score with better visualization
            st.markdown("<h4>Voice Authenticity Score</h4>", unsafe_allow_html=True)
            
            # Determine color based on score
            if abnormal_score > 0.5:
                score_color = "#FF4B4B"  # Red
            elif abnormal_score > 0.3:
                score_color = "#FFC409"  # Yellow
            else:
                score_color = "#2DD36F"  # Green
            
            # Custom progress bar for score
            st.markdown(f"""
            <div style="margin: 10px 0; background-color: rgba(255,255,255,0.1); 
                      border-radius: 5px; height: 25px; width: 100%;">
                <div style="background-color: {score_color}; width: {abnormal_score*100}%; height: 100%; 
                          border-radius: 5px; display: flex; align-items: center; justify-content: center; 
                          color: white; font-weight: bold;">
                    {abnormal_score:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show CONCLUSION with better styling
            st.markdown("<h3 style='text-align: center; margin: 20px 0 10px 0;'>🎯 CONCLUSION</h3>", unsafe_allow_html=True)
            
            if abnormal_score > 0.5:
                st.markdown(f"""
                <div class="conclusion-box conclusion-fake">
                    <h2>🚨 HIGH PROBABILITY OF SYNTHETIC VOICE</h2>
                    <p>Multiple markers of artificially generated or manipulated audio detected.</p>
                </div>
                """, unsafe_allow_html=True)
            elif abnormal_score > 0.3:
                st.markdown(f"""
                <div class="conclusion-box conclusion-warning">
                    <h2>⚠️ INDICATORS OF SYNTHETIC VOICE</h2>
                    <p>Some characteristics of this audio suggest possible manipulation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="conclusion-box conclusion-real">
                    <h2>✅ LIKELY NATURAL VOICE MODULATION</h2>
                    <p>Voice patterns consistent with natural human speech.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add technical details in expander
            with st.expander("View Technical Voice Features"):
                st.markdown(f"""
                <table class="details-table">
                    <tr>
                        <th>Feature</th>
                        <th>Value</th>
                        <th>Normal Range</th>
                    </tr>
                    <tr>
                        <td>Jitter (normalized)</td>
                        <td>{avg_features.get('jitter_normalized', 0):.4f}</td>
                        <td>&gt; 0.005</td>
                    </tr>
                    <tr>
                        <td>Shimmer (normalized)</td>
                        <td>{avg_features.get('shimmer_normalized', 0):.4f}</td>
                        <td>&gt; 0.02</td>
                    </tr>
                    <tr>
                        <td>Pitch Stability</td>
                        <td>{avg_features.get('pitch_stability', 0):.4f}</td>
                        <td>&lt; 10</td>
                    </tr>
                    <tr>
                        <td>Modulation Ratio</td>
                        <td>{avg_features.get('mod_ratio', 0):.4f}</td>
                        <td>0.1 - 10</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
        
    

def display_lipsync_results(lipsync_results):
    if 'error' in lipsync_results:
        st.error(f"Lip-sync analysis error: {lipsync_results['error']}")
        return
    
    st.subheader("Lip-sync Analysis Results")
    
    # Get status and confidence
    status = lipsync_results.get('mismatch_classification', 'UNKNOWN')
    confidence = lipsync_results.get('confidence', 0)
    
    # Display status with enhanced styling
    if status == 'MISMATCHED':
        st.markdown(f"""
        <div class="conclusion-box conclusion-warning">
            <h2>⚠️ LIPSYNC MISMATCH DETECTED</h2>
            <h3>Confidence: {confidence:.1%}</h3>
            <p>Audio and video lip movements appear to be out of sync, suggesting possible manipulation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="conclusion-box conclusion-real">
            <h2>✅ LIPSYNC PROPERLY SYNCHRONIZED</h2>
            <h3>Confidence: {confidence:.1%}</h3>
            <p>Audio aligns well with lip movements, suggesting authentic content.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display metrics with better styling
    if 'offset_analysis' in lipsync_results:
        offset = lipsync_results['offset_analysis']
        st.markdown("<h4>Sync Metrics</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-label">Optimal Offset</div>
                <div class="metric-value">{offset.get('optimal_offset', 0):.3f}s</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Max Sync Score</div>
                <div class="metric-value">{offset.get('max_sync_score', 0):.3f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Analyzed Frames</div>
                <div class="metric-value">{offset.get('analyzed_frames', 0)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a progress bar to visualize sync quality
        sync_quality = confidence if status != 'MISMATCHED' else 1.0 - confidence
        st.markdown("<h4>Sync Quality</h4>", unsafe_allow_html=True)
        
        # Custom styled progress bar
        st.markdown(f"""
        <div style="margin: 10px 0; background-color: rgba(255,255,255,0.1); border-radius: 5px; height: 25px; width: 100%;">
            <div style="background-color: {'#2DD36F' if status != 'MISMATCHED' else '#FFC409'}; 
                      width: {sync_quality*100}%; height: 100%; border-radius: 5px; 
                      display: flex; align-items: center; justify-content: center; 
                      color: white; font-weight: bold;">
                {sync_quality:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main analysis function that integrates main.py logic
def run_main_analysis(filepath, filename):
    """Main analysis function that mimics the main.py workflow"""
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video file analysis
        st.header("🎬 Video Analysis (Main.py Integration)")
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            # Video analysis
            status_text.text("Analyzing video frames...")
            progress_bar.progress(25)
            video_results = analyzers['video'].analyze_video(filepath)
            progress_bar.progress(50)
            # Store video results for export
            st.session_state['last_results'] = video_results
            st.session_state['last_file_name'] = filename
            st.session_state['last_analysis_type'] = 'video'
            # Display enhanced video results
            st.subheader("Video Detection Results")
            
            # Main prediction with enhanced styling
            prediction = video_results.get('final_prediction', 'UNKNOWN')
            confidence = video_results.get('confidence', 0)
            
            # Conclusion box with improved styling
            if prediction == 'FAKE':
                st.markdown(f"""
                <div class="conclusion-box conclusion-fake">
                    <h2>🚨 FAKE DETECTED</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p>The video analysis indicates this content has likely been manipulated.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="conclusion-box conclusion-real">
                    <h2>✅ LIKELY AUTHENTIC</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p>The video analysis suggests this content is likely authentic.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Key metrics in visually appealing format
            st.markdown("<h4>Key Metrics</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">Analyzed Frames</div>
                    <div class="metric-value">{video_results.get('analyzed_frames', 0)}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Faces Detected</div>
                    <div class="metric-value">{video_results.get('faces_detected', 0)}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Frame Consistency</div>
                    <div class="metric-value">{video_results.get('frame_consistency', 0):.1%}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">FPS</div>
                    <div class="metric-value">{video_results.get('fps', 0):.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics in expandable section with nice table
            with st.expander("View Detailed Analysis"):
                st.markdown(f"""
                <table class="details-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Frames</td>
                        <td>{video_results.get('total_frames', 0)}</td>
                    </tr>
                    <tr>
                        <td>Temporal Consistency</td>
                        <td>{video_results.get('temporal_consistency', 0):.1%}</td>
                    </tr>
                    <tr>
                        <td>Score Stability</td>
                        <td>{video_results.get('score_stability', 0):.1%}</td>
                    </tr>
                    <tr>
                        <td>Analysis Duration</td>
                        <td>{video_results.get('analysis_duration', 0):.2f} seconds</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
            # Check for audio in video (main.py logic)
            status_text.text("Checking for audio track...")
            progress_bar.progress(75)
            audio_present = False
            temp_audio_path = None
            try:
                with VideoFileClip(filepath) as video:
                    if video.audio is not None:
                        st.header("🎵 Audio Analysis (from video)")
                        # Save audio to temp file (main.py approach)
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                            temp_audio_path = temp_audio.name
                            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                        audio_present = True
                        # Analyze audio
                        audio_results = analyzers['audio'].analyze_audio(temp_audio_path)
                        display_audio_results(audio_results)
                        # Store audio results for export
                        st.session_state['last_results'] = audio_results
                        st.session_state['all_results']['audio'] = audio_results
                        st.session_state['last_file_name'] = filename
                        st.session_state['last_analysis_type'] = 'audio'
                        # Lip-sync analysis (main.py approach)
                        st.header("👄 Lip-sync Analysis")
                        lipsync_results = analyzers['lipsync'].analyze_video(filepath)
                        display_lipsync_results(lipsync_results)
                        # Store lipsync results for export
                        st.session_state['last_results'] = lipsync_results
                        st.session_state['all_results']['lipsync'] = lipsync_results
                        st.session_state['last_file_name'] = filename
                        st.session_state['last_analysis_type'] = 'lipsync'
                        # Clean up temp audio
                        if temp_audio_path and os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
                    else:
                        st.info("No audio track found in video")
            except Exception as e:
                st.warning(f"Audio analysis failed: {str(e)}")
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
        except Exception as e:
            st.error(f"Video analysis failed: {str(e)}")
    
    elif file_extension in ['.wav', '.mp3', '.flac']:
        # Audio file analysis (main.py logic)
        st.header("🎵 Audio Analysis (Main.py Integration)")
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("Analyzing audio...")
            progress_bar.progress(50)
            audio_results = analyzers['audio'].analyze_audio(filepath)
            progress_bar.progress(100)
            display_audio_results(audio_results)
            status_text.text("Analysis complete!")
            # Store audio results for export
            st.session_state['last_results'] = audio_results
            st.session_state['all_results']['audio'] = audio_results
            st.session_state['last_file_name'] = filename
            st.session_state['last_analysis_type'] = 'audio'
            st.info("No video data found for deepfake detection and lip-sync analysis.")
        except Exception as e:
            st.error(f"Audio analysis failed: {str(e)}")
    
    else:
        st.error("Unsupported file type. Please provide a video or audio file.")

# Main interface
st.title("🛡️ Deepfake Detection System")
st.markdown("Upload video or audio files to detect potential deepfakes using AI-powered analysis.")
st.markdown("**Now integrated with main.py logic for comprehensive analysis!**")

# File upload
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['mp4', 'avi', 'mov', 'mkv', 'wav', 'mp3', 'flac'],
    help="Supported formats: Video (MP4, AVI, MOV, MKV) and Audio (WAV, MP3, FLAC)"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Show file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Use the main.py integrated analysis
        run_main_analysis(tmp_file_path, uploaded_file.name)
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    
    st.markdown("""
    This system uses multiple AI models to detect deepfakes:
    
    <h3 style='color: #FF4B4B; border-bottom: 2px solid #FF4B4B; padding-bottom: 8px; margin-bottom: 15px;'>🎬 Video Analysis:</h3>
    <ul style='margin-left: 20px; margin-bottom: 15px;'>
        <li>Advanced face detection and analysis</li>
        <li>Frame-by-frame deepfake detection</li>
        <li>Temporal consistency analysis</li>
    </ul>
    
    <h3 style='color: #FF4B4B; border-bottom: 2px solid #FF4B4B; padding-bottom: 8px; margin-bottom: 15px;'>🎵 Audio Analysis:</h3>
    <ul style='margin-left: 20px; margin-bottom: 15px;'>
        <li>Voice modulation analysis</li>
        <li>Synthetic speech detection</li>
        <li>Audio quality assessment</li>
    </ul>
    
    <h3 style='color: #FF4B4B; border-bottom: 2px solid #FF4B4B; padding-bottom: 8px; margin-bottom: 15px;'>👄 Lip-sync Analysis:</h3>
    <ul style='margin-left: 20px; margin-bottom: 15px;'>
        <li>Audio-visual synchronization</li>
        <li>Mouth movement correlation</li>
        <li>Temporal offset detection</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.header("📊 Model Information")
    st.markdown("""
    - **Video Model:** dima806/deepfake_vs_real_image_detection
    - **Face Detection:** MediaPipe Face Detection
    - **Audio Analysis:** Librosa-based signal processing
    - **Lip-sync:** MediaPipe Face Mesh + correlation analysis
    """)

# ---
# ---
# Additional Features: Metadata Export & Evidence System Integration


import json
def np_encoder(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def collect_metadata(file_name, all_results):
    """Collects metadata and all results for export/reporting."""
    from datetime import datetime
    return {
        "file_name": file_name,
        "analysis_timestamp": datetime.now().isoformat(),
        "results": {k: v for k, v in all_results.items() if v is not None}
    }

# Store last analysis results in session state for export
if 'last_results' not in st.session_state:
    st.session_state['last_results'] = None
# Store all analysis results in session state for export
if 'all_results' not in st.session_state:
    st.session_state['all_results'] = {'video': None, 'audio': None, 'lipsync': None}
if 'last_file_name' not in st.session_state:
    st.session_state['last_file_name'] = None
if 'last_analysis_type' not in st.session_state:
    st.session_state['last_analysis_type'] = None

# After each analysis, store results (add this after analysis in your main logic if needed)
# Example usage (pseudo):
# st.session_state['last_results'] = { ... }
# st.session_state['last_file_name'] = uploaded_file.name
# st.session_state['last_analysis_type'] = 'video' or 'audio'

# Export section with enhanced UI
st.markdown("<hr style='margin: 30px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.header("📝 Export & Evidence Integration")

if any(st.session_state['all_results'].values()):
    metadata = collect_metadata(
        st.session_state['last_file_name'],
        st.session_state['all_results']
    )
    json_report = json.dumps(metadata, indent=2, default=np_encoder)
    
    # Enhanced export UI with columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h4>📊 Analysis Report Ready</h4>
            <p>Your analysis for <strong>{st.session_state['last_file_name']}</strong> is complete.</p>
            <p>The report contains detailed results from all detection modules.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.download_button(
            label="⬇️ Download Analysis Report (JSON)",
            data=json_report,
            file_name=f"deepfake_analysis_{st.session_state['last_file_name']}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        if st.button("🔄 Send to Digital Evidence System", use_container_width=True):
            st.info("Integration with digital evidence systems is not yet implemented. This is a placeholder.")
else:
    st.markdown("""
    <div style="background-color: rgba(56, 128, 255, 0.1); border-radius: 10px; 
               padding: 15px; border-left: 4px solid #3880FF;">
        <h4>ℹ️ No Analysis Results Yet</h4>
        <p>Upload and analyze a file to generate an exportable report.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Python, Flask, and AI models")
