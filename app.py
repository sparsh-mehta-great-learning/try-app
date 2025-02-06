import streamlit as st
import torch
import os
import numpy as np
import librosa
import whisper
from openai import OpenAI
import tempfile
import warnings
import re
from contextlib import contextmanager
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import subprocess
import json
import shutil
from pathlib import Path
import time
from faster_whisper import WhisperModel
import soundfile as sf
import logging
from typing import Optional, Dict, Any, List, Tuple
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

@contextmanager
def temporary_file(suffix: Optional[str] = None):
    """Context manager for temporary file handling"""
    temp_path = tempfile.mktemp(suffix=suffix)
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

class ProgressTracker:
    """Handles progress tracking and ETA calculations"""
    def __init__(self, status_element, progress_bar):
        self.status = status_element
        self.progress = progress_bar
        self.start_time = time.time()
    
    def update(self, progress: float, message: str):
        """Update progress with ETA calculation"""
        self.status.update(label=f"{message} ({progress:.1%})")
        self.progress.progress(progress)
        
        if progress > 0:
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            self.status.update(label=f"{message} ({progress:.1%}) - ETA: {remaining:.0f}s")

class AudioFeatureExtractor:
    """Handles audio feature extraction with improved pause detection"""
    def __init__(self):
        self.sr = 16000
        self.hop_length = 512
        self.n_fft = 2048
        self.chunk_duration = 300
        # Parameters for pause detection
        self.min_pause_duration = 4  # minimum pause duration in seconds
        self.silence_threshold = -40    # dB threshold for silence
        
    def _analyze_pauses(self, silent_frames, frame_time):
        """Analyze pauses with minimal memory usage."""
        pause_durations = []
        current_pause = 0

        for is_silent in silent_frames:
            if is_silent:
                current_pause += 1
            elif current_pause > 0:
                duration = current_pause * frame_time
                if duration > 0.5:  # Only count pauses longer than 300ms
                    pause_durations.append(duration)
                current_pause = 0

        if pause_durations:
            return {
                'total_pauses': len(pause_durations),
                'mean_pause_duration': float(np.mean(pause_durations))
            }
        return {
            'total_pauses': 0,
            'mean_pause_duration': 0.0
        }

    def extract_features(self, audio_path: str, progress_callback=None) -> Dict[str, float]:
        try:
            if progress_callback:
                progress_callback(0.1, "Loading audio file...")
            
            # Load audio with proper sample rate
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate amplitude features
            rms = librosa.feature.rms(y=audio)[0]
            mean_amplitude = float(np.mean(rms)) * 100  # Scale for better readability
            
            # Calculate pitch features using pyin
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                sr=sr,
                fmin=70,
                fmax=400,
                frame_length=2048
            )
            
            # Filter out zero and NaN values
            valid_f0 = f0[np.logical_and(voiced_flag == 1, ~np.isnan(f0))]
            
            # Improved pause detection
            S = np.abs(librosa.stft(audio))
            db = librosa.amplitude_to_db(S, ref=np.max)
            frame_length = 2048  # ~128ms at 16kHz
            hop_length = 512     # ~32ms at 16kHz
            
            # Calculate energy in dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # Detect silence with improved threshold
            silence_threshold = -40
            is_silence = rms_db < silence_threshold
            
            # Count pauses (silence segments longer than 0.5 seconds)
            min_pause_frames = int(0.5 * sr / hop_length)  # 0.5 seconds minimum pause duration
            silence_runs = np.split(is_silence, np.where(np.diff(is_silence))[0] + 1)
            
            # Count valid pauses (longer than minimum duration)
            valid_pauses = sum(1 for run in silence_runs if len(run) >= min_pause_frames and run[0])
            
            # Calculate duration in minutes
            duration_minutes = len(audio) / sr / 60
            
            # Calculate pauses per minute
            pauses_per_minute = valid_pauses / duration_minutes if duration_minutes > 0 else 0
            
            return {
                "pitch_mean": float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0,
                "pitch_std": float(np.std(valid_f0)) if len(valid_f0) > 0 else 0,
                "mean_amplitude": mean_amplitude,
                "amplitude_deviation": float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0,
                "pauses_per_minute": float(pauses_per_minute),
                "duration": float(len(audio) / sr),
                "rising_patterns": int(np.sum(np.diff(valid_f0) > 0)) if len(valid_f0) > 1 else 0,
                "falling_patterns": int(np.sum(np.diff(valid_f0) < 0)) if len(valid_f0) > 1 else 0,
                "variations_per_minute": float(len(valid_f0) / (len(audio) / sr / 60)) if len(audio) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise AudioProcessingError(f"Feature extraction failed: {str(e)}")


    def _process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single chunk of audio with improved pause detection"""
        # Calculate STFT
        D = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)
        
        # Calculate RMS energy in dB
        rms = librosa.feature.rms(S=S)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Detect pauses using silence threshold
        is_silence = rms_db < self.silence_threshold
        frame_time = self.hop_length / self.sr
        pause_analysis = self._analyze_pauses(is_silence, frame_time)
        
        # Calculate pitch features
        f0, voiced_flag, _ = librosa.pyin(
            chunk,
            sr=self.sr,
            fmin=70,
            fmax=400,
            frame_length=self.n_fft
        )
        
        return {
            "rms": rms,
            "f0": f0[voiced_flag == 1] if f0 is not None else np.array([]),
            "duration": len(chunk) / self.sr,
            "pause_count": pause_analysis['total_pauses'],
            "mean_pause_duration": pause_analysis['mean_pause_duration']
        }

    def _combine_features(self, features: List[Dict[str, Any]]) -> Dict[str, float]:
        """Combine features from multiple chunks"""
        all_f0 = np.concatenate([f["f0"] for f in features if len(f["f0"]) > 0])
        all_rms = np.concatenate([f["rms"] for f in features])
        
        pitch_mean = np.mean(all_f0) if len(all_f0) > 0 else 0
        pitch_std = np.std(all_f0) if len(all_f0) > 0 else 0
        
        return {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "mean_amplitude": float(np.mean(all_rms)),
            "amplitude_deviation": float(np.std(all_rms) / np.mean(all_rms)) if np.mean(all_rms) > 0 else 0,
            "rising_patterns": int(np.sum(np.diff(all_f0) > 0)) if len(all_f0) > 1 else 0,
            "falling_patterns": int(np.sum(np.diff(all_f0) < 0)) if len(all_f0) > 1 else 0,
            "variations_per_minute": float((np.sum(np.diff(all_f0) != 0) if len(all_f0) > 1 else 0) / 
                                        (sum(f["duration"] for f in features) / 60))
        }

class ContentAnalyzer:
    """Analyzes teaching content using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retry_count = 3
        self.retry_delay = 1
        
    def analyze_content(self, transcript: str, progress_callback=None) -> Dict[str, Any]:
        """Analyze teaching content with retry logic and robust JSON handling"""
        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing content analysis...")
                
                prompt = self._create_analysis_prompt(transcript)
                
                if progress_callback:
                    progress_callback(0.5, "Processing with AI model...")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a teaching expert providing a structured JSON analysis. Always respond with a valid JSON object containing scores and detailed citations."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content.strip()
                
                try:
                    result = json.loads(result_text)
                    # Ensure default structure with meaningful content
                    default_structure = {
                        "score": 0,
                        "citations": ["No specific citations available"]
                    }
                    
                    for category in ["subjectMatterAccuracy", "firstPrinciplesApproach", 
                                   "examplesAndContext", "cohesiveStorytelling", 
                                   "engagement", "professionalTone"]:
                        if category not in result:
                            result[category] = default_structure.copy()
                        elif not isinstance(result[category], dict):
                            result[category] = default_structure.copy()
                        else:
                            # Ensure required fields exist
                            if "score" not in result[category]:
                                result[category]["score"] = 0
                            if "citations" not in result[category]:
                                result[category]["citations"] = ["No specific citations available"]
                
                    return result
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response: {result_text}")
                    raise
                
            except Exception as e:
                logger.error(f"Content analysis attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_count - 1:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def _create_analysis_prompt(self, transcript: str) -> str:
        """Create the analysis prompt"""
        return f"""Analyze this teaching content and provide scores and citations:
Transcript: {transcript}
For each category below, provide:
1. Score (0 or 1)
2. Supporting citations with timestamps (if score is 0, cite problematic areas)
Concept Assessment:
1. Subject Matter Accuracy
2. First Principles Approach
3. Examples and Business Context
4. Cohesive Storytelling
5. Engagement and Interaction
6. Professional Tone
Code Assessment:
1. Depth of Explanation
2. Output Interpretation
3. Breaking down Complexity
Format as JSON."""

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                           progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = audio_features.get('duration', 0) / 60
            
            # Calculate words per minute with minimum threshold
            words_per_minute = max(words / duration_minutes if duration_minutes > 0 else 0, 1)
            
            # Improved filler word detection
            filler_words = re.findall(r'\b(um|uh|like|you\s+know|basically|actually|literally)\b', 
                                    transcript.lower())
            fillers_count = len(filler_words)
            
            # Calculate fillers per minute with minimum threshold
            fillers_per_minute = max(fillers_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Improved error detection
            repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
            incomplete_sentences = len(re.findall(r'[a-zA-Z]+\s*\.\.\.|\b[a-zA-Z]+\s*-\s+', transcript))
            errors_count = repeated_words + incomplete_sentences
            
            # Calculate errors per minute with minimum threshold
            errors_per_minute = max(errors_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Ensure mean amplitude is properly scaled and evaluated
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            
            # Updated acceptable ranges based on research and best practices
            return {
                "speed": {
                    "score": 1 if 130 <= words_per_minute <= 150 else 0,  # Narrowed optimal teaching range
                    "wpm": float(words_per_minute),
                    "total_words": int(words),
                    "duration_minutes": float(duration_minutes)
                },
                "fluency": {
                    "score": 1 if fillers_per_minute <= 3 and errors_per_minute <= 2 else 0,  # Stricter fluency standards
                    "fillersPerMin": float(fillers_per_minute),
                    "errorsPerMin": float(errors_per_minute)
                },
                "flow": {
                    "score": 1 if 8 <= float(audio_features.get("pauses_per_minute", 0)) <= 12 else 0,  # Adjusted for teaching pace
                    "pausesPerMin": float(audio_features.get("pauses_per_minute", 0))
                },
                "intonation": {
                    "pitch": float(audio_features.get("pitch_mean", 0)),
                    "pitchScore": 1 if 80 <= float(audio_features.get("pitch_std", 0)) <= 90 else 0,  # Refined pitch variation range
                    "pitchVariation": float(audio_features.get("pitch_std", 0)),
                    "patternScore": 1 if float(audio_features.get("variations_per_minute", 0)) >= 8 else 0,  # Increased minimum variations
                    "risingPatterns": int(audio_features.get("rising_patterns", 0)),
                    "fallingPatterns": int(audio_features.get("falling_patterns", 0)),
                    "variationsPerMin": float(audio_features.get("variations_per_minute", 0))
                },
                "energy": {
                    "score": 1 if 20 <= mean_amplitude <= 40 else 0,  # Adjusted amplitude range for clearer speech
                    "meanAmplitude": mean_amplitude,
                    "amplitudeDeviation": float(audio_features.get("amplitude_deviation", 0))
                }
            }

        except Exception as e:
            logger.error(f"Error in speech metrics evaluation: {e}")
            raise

class RecommendationGenerator:
    """Generates teaching recommendations using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retry_count = 3
        self.retry_delay = 1
        
    def generate_recommendations(self, 
                           metrics: Dict[str, Any], 
                           content_analysis: Dict[str, Any], 
                           progress_callback=None) -> Dict[str, Any]:
        """Generate recommendations with robust JSON handling"""
        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing recommendation analysis...")
                
                prompt = self._create_recommendation_prompt(metrics, content_analysis)
                
                if progress_callback:
                    progress_callback(0.5, "Generating recommendations...")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a teaching expert providing actionable recommendations. Always respond with a valid JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                if progress_callback:
                    progress_callback(0.8, "Formatting recommendations...")
                
                # Ensure we have valid JSON
                result_text = response.choices[0].message.content.strip()
                
                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError:
                    # Fallback to a default structure if JSON parsing fails
                    result = {
                        "geographyFit": "Unknown",
                        "improvements": [
                            "Unable to generate specific recommendations"
                        ],
                        "rigor": "Undetermined"
                    }
                
                if progress_callback:
                    progress_callback(1.0, "Recommendations complete!")
                
                return result
                
            except Exception as e:
                logger.error(f"Recommendation generation attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_count - 1:
                    # Return a default structure on final failure
                    return {
                        "geographyFit": "Unknown",
                        "improvements": [
                            "Unable to generate specific recommendations"
                        ],
                        "rigor": "Undetermined"
                    }
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def _create_recommendation_prompt(self, metrics: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """Create the recommendation prompt"""
        return f"""Based on the following metrics and analysis, provide recommendations:
Metrics: {json.dumps(metrics)}
Content Analysis: {json.dumps(content_analysis)}
Provide:
1. Specific improvements needed
2. Rigor assessment considering technical and teaching abilities
Format as JSON with keys: geographyFit, improvements (array), rigor"""

class MentorEvaluator:
    """Main class for video evaluation"""
    def __init__(self, model_cache_dir: Optional[str] = None):
        """Initialize with proper model caching"""
        self.api_key = st.secrets["OPENAI_API_KEY"]
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
    
        # Create a specific directory for the whisper model
        if model_cache_dir:
            self.model_cache_dir = model_cache_dir
        else:
            # Create a persistent directory in the user's home directory
            self.model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            os.makedirs(self.model_cache_dir, exist_ok=True)
        
        self._whisper_model = None
        self._feature_extractor = None
        self._content_analyzer = None
        self._recommendation_generator = None

    @property
    def whisper_model(self):
        """Lazy loading of whisper model with proper cache directory handling"""
        if self._whisper_model is None:
            try:
                # First try to load from cache
                self._whisper_model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                    download_root=self.model_cache_dir,
                    local_files_only=True
                )
            except Exception as e:
                logger.info(f"Could not load model from cache, downloading: {e}")
                # If loading from cache fails, download the model
                self._whisper_model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                    download_root=self.model_cache_dir,
                    local_files_only=False
                )
                logger.info("Model downloaded successfully")
        return self._whisper_model

    @property
    def feature_extractor(self):
        """Lazy loading of feature extractor"""
        if self._feature_extractor is None:
            self._feature_extractor = AudioFeatureExtractor()
        return self._feature_extractor

    @property
    def content_analyzer(self):
        """Lazy loading of content analyzer"""
        if self._content_analyzer is None:
            self._content_analyzer = ContentAnalyzer(api_key=self.api_key)
        return self._content_analyzer

    @property
    def recommendation_generator(self):
        """Lazy loading of recommendation generator"""
        if self._recommendation_generator is None:
            self._recommendation_generator = RecommendationGenerator(api_key=self.api_key)
        return self._recommendation_generator

    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        """Evaluate video with proper resource management"""
        with temporary_file(suffix=".wav") as temp_audio:
            try:
                # Extract audio
                with st.status("Extracting audio...") as status:
                    progress_bar = st.progress(0)
                    tracker = ProgressTracker(status, progress_bar)
                    self._extract_audio(video_path, temp_audio, tracker.update)

                # Extract features
                with st.status("Extracting audio features...") as status:
                    progress_bar = st.progress(0)
                    tracker = ProgressTracker(status, progress_bar)
                    audio_features = self.feature_extractor.extract_features(
                        temp_audio,
                        tracker.update
                    )

                # Transcribe
                with st.status("Transcribing audio...") as status:
                    progress_bar = st.progress(0)
                    tracker = ProgressTracker(status, progress_bar)
                    transcript = self._transcribe_audio(temp_audio, tracker.update)

                # Analyze content
                with st.status("Analyzing content...") as status:
                    progress_bar = st.progress(0)
                    tracker = ProgressTracker(status, progress_bar)
                    content_analysis = self.content_analyzer.analyze_content(
                        transcript,
                        tracker.update
                    )

                # Evaluate speech
                with st.status("Evaluating speech metrics...") as status:
                    progress_bar = st.progress(0)
                    tracker = ProgressTracker(status, progress_bar)
                    speech_metrics = self._evaluate_speech_metrics(
                        transcript,
                        audio_features,
                        tracker.update
                    )

                # Generate recommendations
                with st.status("Generating recommendations...") as status:
                    progress_bar = st.progress(0)
                    tracker = ProgressTracker(status, progress_bar)
                    recommendations = self.recommendation_generator.generate_recommendations(
                        speech_metrics,
                        content_analysis,
                        tracker.update
                    )

                return {
                    "communication": speech_metrics,
                    "teaching": content_analysis,
                    "recommendations": recommendations,
                    "transcript": transcript
                }

            except Exception as e:
                logger.error(f"Error in video evaluation: {e}")
                raise

    def _extract_audio(self, video_path: str, output_path: str, progress_callback=None) -> str:
        """Extract audio from video"""
        try:
            if progress_callback:
                progress_callback(0.1, "Checking dependencies...")

            if not shutil.which('ffmpeg'):
                raise AudioProcessingError("FFmpeg is not installed")

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            if not os.access(os.path.dirname(output_path), os.W_OK):
                raise AudioProcessingError(f"No write permission for output directory: {os.path.dirname(output_path)}")

            if progress_callback:
                progress_callback(0.3, "Configuring audio extraction...")

            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                '-v', 'warning',
                '-y',
                output_path
            ]

            if progress_callback:
                progress_callback(0.5, "Extracting audio...")

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise AudioProcessingError(f"FFmpeg Error: {result.stderr}")

            if progress_callback:
                progress_callback(1.0, "Audio extraction complete!")

            return output_path

        except Exception as e:
            logger.error(f"Error in audio extraction: {e}")
            raise AudioProcessingError(f"Audio extraction failed: {str(e)}")

    def _transcribe_audio(self, audio_path: str, progress_callback=None) -> str:
        """Transcribe audio with improved memory management"""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading transcription model...")
    
            audio_info = sf.info(audio_path)
            total_duration = audio_info.duration
            chunk_duration = 5 * 60  # 5-minute chunks
            overlap_duration = 10  # 10-second overlap
    
            transcripts = []
            total_chunks = int(np.ceil(total_duration / (chunk_duration - overlap_duration)))
    
            with sf.SoundFile(audio_path) as f:
                for i in range(total_chunks):
                    if progress_callback:
                        progress_callback(0.4 + (i / total_chunks) * 0.4,
                                       f"Transcribing chunk {i + 1}/{total_chunks}...")
    
                    # Calculate positions in samples
                    start_sample = int(i * (chunk_duration - overlap_duration) * f.samplerate)
                    f.seek(start_sample)
                    chunk = f.read(frames=int(chunk_duration * f.samplerate))
    
                    with temporary_file(suffix=".wav") as chunk_path:
                        sf.write(chunk_path, chunk, f.samplerate)
                        # The fix: properly handle the segments from faster-whisper
                        segments, _ = self.whisper_model.transcribe(chunk_path)
                        # Combine all segment texts
                        chunk_text = ' '.join(segment.text for segment in segments)
                        transcripts.append(chunk_text)
    
            if progress_callback:
                progress_callback(1.0, "Transcription complete!")
    
            return " ".join(transcripts)
    
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise

    def calculate_speech_metrics(self, transcript: str, audio_duration: float) -> Dict[str, float]:
        """Calculate words per minute and other speech metrics."""
        words = len(transcript.split())
        minutes = audio_duration / 60
        return {
            'words_per_minute': words / minutes if minutes > 0 else 0,
            'total_words': words,
            'duration_minutes': minutes
        }

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                               progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = audio_features.get('duration', 0) / 60
            
            # Calculate words per minute with minimum threshold
            words_per_minute = max(words / duration_minutes if duration_minutes > 0 else 0, 1)
            
            # Improved filler word detection
            filler_words = re.findall(r'\b(um|uh|like|you\s+know|basically|actually|literally)\b', 
                                    transcript.lower())
            fillers_count = len(filler_words)
            
            # Calculate fillers per minute with minimum threshold
            fillers_per_minute = max(fillers_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Improved error detection
            repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
            incomplete_sentences = len(re.findall(r'[a-zA-Z]+\s*\.\.\.|\b[a-zA-Z]+\s*-\s+', transcript))
            errors_count = repeated_words + incomplete_sentences
            
            # Calculate errors per minute with minimum threshold
            errors_per_minute = max(errors_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Ensure mean amplitude is properly scaled and evaluated
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            
            # Updated acceptable ranges based on research and best practices
            return {
                "speed": {
                    "score": 1 if 130 <= words_per_minute <= 150 else 0,  # Narrowed optimal teaching range
                    "wpm": float(words_per_minute),
                    "total_words": int(words),
                    "duration_minutes": float(duration_minutes)
                },
                "fluency": {
                    "score": 1 if fillers_per_minute <= 3 and errors_per_minute <= 2 else 0,  # Stricter fluency standards
                    "fillersPerMin": float(fillers_per_minute),
                    "errorsPerMin": float(errors_per_minute)
                },
                "flow": {
                    "score": 1 if 8 <= float(audio_features.get("pauses_per_minute", 0)) <= 12 else 0,  # Adjusted for teaching pace
                    "pausesPerMin": float(audio_features.get("pauses_per_minute", 0))
                },
                "intonation": {
                    "pitch": float(audio_features.get("pitch_mean", 0)),
                    "pitchScore": 1 if 80 <= float(audio_features.get("pitch_std", 0)) <= 90 else 0,  # Refined pitch variation range
                    "pitchVariation": float(audio_features.get("pitch_std", 0)),
                    "patternScore": 1 if float(audio_features.get("variations_per_minute", 0)) >= 8 else 0,  # Increased minimum variations
                    "risingPatterns": int(audio_features.get("rising_patterns", 0)),
                    "fallingPatterns": int(audio_features.get("falling_patterns", 0)),
                    "variationsPerMin": float(audio_features.get("variations_per_minute", 0))
                },
                "energy": {
                    "score": 1 if 20 <= mean_amplitude <= 40 else 0,  # Adjusted amplitude range for clearer speech
                    "meanAmplitude": mean_amplitude,
                    "amplitudeDeviation": float(audio_features.get("amplitude_deviation", 0))
                }
            }

        except Exception as e:
            logger.error(f"Error in speech metrics evaluation: {e}")
            raise

def validate_video_file(file_path: str):
    """Validate video file before processing"""
    valid_extensions = {'.mp4', '.avi', '.mov'}
    
    if not os.path.exists(file_path):
        raise ValueError("Video file does not exist")
        
    if os.path.splitext(file_path)[1].lower() not in valid_extensions:
        raise ValueError("Unsupported video format")
        
    if os.path.getsize(file_path) > 2 * 1024 * 1024 * 1024:  # 2GB
        raise ValueError("File size exceeds 2GB limit")
        
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', file_path],
            capture_output=True,
            text=True
        )
        if probe.returncode != 0:
            raise ValueError("Invalid video file")
    except subprocess.SubprocessError:
        raise ValueError("Unable to validate video file")

def display_evaluation(evaluation: Dict[str, Any]):
    """Display evaluation results with improved metrics visualization"""
    try:
        tabs = st.tabs(["Communication", "Teaching", "Recommendations", "Transcript"])

        with tabs[0]:
            with st.status("Loading communication metrics...") as status:
                progress_bar = st.progress(0)
                progress_bar.progress(0.2)
                st.header("Communication")

                # Speed metrics
                st.subheader("Speed")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "Pass" if evaluation["communication"]["speed"]["score"] == 1 
                             else "Need Improvement")
                with col2:
                    st.metric("Words per Minute", 
                             f"{evaluation['communication']['speed']['wpm']:.1f}")
                st.caption("Acceptable Range: 130-150 WPM")
                progress_bar.progress(0.4)

                # Fluency metrics
                st.subheader("Fluency")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", "Pass" if evaluation["communication"]["fluency"]["score"] == 1 
                             else "Need Improvement")
                with col2:
                    st.metric("Fillers/Min", 
                             f"{evaluation['communication']['fluency']['fillersPerMin']:.1f}")
                with col3:
                    st.metric("Errors/Min", 
                             f"{evaluation['communication']['fluency']['errorsPerMin']:.1f}")
                progress_bar.progress(0.6)

                # Flow metrics
                st.subheader("Flow")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "Pass" if evaluation["communication"]["flow"]["score"] == 1 
                             else "Need Improvement")
                with col2:
                    st.metric("Pauses/Min", 
                             f"{evaluation['communication']['flow']['pausesPerMin']:.1f}")
                st.caption("Acceptable Range: 8-12 pauses/min")

                # Intonation metrics
                st.subheader("Intonation")
                
                # Frequency/Pitch section
                st.write("**Frequency/Pitch Metrics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Base Frequency", 
                             f"{evaluation['communication']['intonation']['pitch']:.1f} Hz")
                with col2:
                    st.metric("Pitch Score", 
                             "Pass" if evaluation["communication"]["intonation"]["pitchScore"] == 1 
                             else "Need Improvement")
                with col3:
                    st.metric("Pitch Variation (σ)", 
                             f"{evaluation['communication']['intonation']['pitchVariation']:.1f} Hz")
                st.caption("Acceptable Pitch Variation Range: 80-90 Hz")
                
                # Patterns section
                st.write("**Pattern Analysis:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pattern Score",
                             "Pass" if evaluation["communication"]["intonation"]["patternScore"] == 1 
                             else "Need Improvement")
                with col2:
                    st.metric("Rising Patterns",
                             evaluation["communication"]["intonation"]["risingPatterns"])
                with col3:
                    st.metric("Falling Patterns",
                             evaluation["communication"]["intonation"]["fallingPatterns"])
                with col4:
                    st.metric("Variations/Min",
                             f"{evaluation['communication']['intonation']['variationsPerMin']:.1f}")
                st.caption("Acceptable Variations per Minute: >=8")
                progress_bar.progress(0.8)

                # Energy metrics
                st.subheader("Energy")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", 
                             "Pass" if evaluation["communication"]["energy"]["score"] == 1 
                             else "Need Improvement")
                with col2:
                    st.metric("Mean Amplitude",
                             f"{evaluation['communication']['energy']['meanAmplitude']:.1f}")
                with col3:
                    st.metric("Amplitude Deviation (σ/μ)",
                             f"{evaluation['communication']['energy']['amplitudeDeviation']:.2f}")
                
                st.caption("""
                Acceptable Ranges:
                - Mean Amplitude: 20 - 40 (normalized RMS value)
                - Amplitude Deviation (σ/μ): >0.2
                """)
                
                progress_bar.progress(1.0)
                status.update(label="Communication metrics loaded!", state="complete")

        # Teaching tab with improved error handling
        with tabs[1]:
            st.header("Teaching Analysis")
            
            # Access the teaching data directly from the evaluation
            teaching_data = evaluation.get("teaching", {})
            
            # Debug prints
            print("\nDEBUG INFO:")
            print("Full teaching_data:", json.dumps(teaching_data, indent=2))
            
            # Get the concept assessment data
            concept_data = teaching_data.get("ConceptAssessment", {})
            code_data = teaching_data.get("CodeAssessment", {})
            
            print("Concept Data:", json.dumps(concept_data, indent=2))
            print("Code Data:", json.dumps(code_data, indent=2))
            
            # Define categories to display
            concept_categories = {
                "Subject Matter Accuracy": "SubjectMatterAccuracy",
                "First Principles Approach": "FirstPrinciplesApproach",
                "Examples and Context": "ExamplesAndBusinessContext",
                "Cohesive Storytelling": "CohesiveStorytelling",
                "Engagement": "EngagementAndInteraction",
                "Professional Tone": "ProfessionalTone"
            }
            
            code_categories = {
                "Depth of Explanation": "DepthOfExplanation",
                "Output Interpretation": "OutputInterpretation",
                "Breaking Down Complexity": "BreakingDownComplexity"
            }
            
            # Display Concept Assessment
            st.subheader("Concept Assessment")
            
            # Debug print for first category
            first_category = list(concept_categories.values())[0]
            print(f"First category data:", json.dumps(concept_data.get(first_category, {}), indent=2))
            
            col1, col2 = st.columns(2)
            
            # Display each concept category
            for i, (display_name, key) in enumerate(concept_categories.items()):
                with col1 if i % 2 == 0 else col2:
                    st.write(f"### {display_name}")
                    category_data = concept_data.get(key, {"score": 0, "citations": []})
                    
                    # Debug print for category data
                    print(f"Category {key} data:", json.dumps(category_data, indent=2))
                    
                    # Display score with color-coded metric
                    score = category_data.get("score", 0)
                    st.metric(
                        "Score", 
                        "Pass" if score == 1 else "Needs Improvement",
                        delta="✓" if score == 1 else "✗",
                        delta_color="normal" if score == 1 else "inverse"
                    )
                    
                    # Display citations in an expandable section
                    citations = category_data.get("citations", [])
                    with st.expander("View Evidence"):
                        if citations:  # Only create expander if there are citations
                            if isinstance(citations, list):
                                for citation in citations:
                                    if isinstance(citation, dict):
                                        st.write(f"• [{citation.get('timestamp')}] {citation.get('description')}")
                                    else:
                                        st.write(f"• {citation}")
                            else:
                                st.write(f"• {citations}")
                        else:
                            st.write("No evidence available")
                    
                    st.markdown("---")
            
            # Display Code Assessment
            st.subheader("Code Assessment")
            col1, col2 = st.columns(2)
            
            # Display each code category
            for i, (display_name, key) in enumerate(code_categories.items()):
                with col1 if i % 2 == 0 else col2:
                    st.write(f"### {display_name}")
                    category_data = code_data.get(key, {"score": 0, "citations": []})
                    
                    # Debug print for code category data
                    print(f"Code category {key} data:", json.dumps(category_data, indent=2))
                    
                    # Display score with color-coded metric
                    score = category_data.get("score", 0)
                    st.metric(
                        "Score", 
                        "Pass" if score == 1 else "Needs Improvement",
                        delta="✓" if score == 1 else "✗",
                        delta_color="normal" if score == 1 else "inverse"
                    )
                    
                    # Display citations in an expandable section
                    citations = category_data.get("citations", [])
                    with st.expander("View Evidence"):
                        if citations:  # Only create expander if there are citations
                            if isinstance(citations, list):
                                for citation in citations:
                                    if isinstance(citation, dict):
                                        st.write(f"• [{citation.get('timestamp')}] {citation.get('description')}")
                                    else:
                                        st.write(f"• {citation}")
                            else:
                                st.write(f"• {citations}")
                        else:
                            st.write("No evidence available")
                    
                    st.markdown("---")

            # Add a debug section to display raw data
            with st.expander("Debug Raw Data"):
                st.json(teaching_data)

        # Recommendations tab with improved formatting
        with tabs[2]:
            st.header("Recommendations")
            
            recommendations = evaluation.get("recommendations", {})
            
            # Geography Fit with improved formatting
            st.subheader("🌍 Geography Fit")
            geography_fit = recommendations.get("geographyFit", "Not specified")
            if isinstance(geography_fit, dict):
                for region, fit in geography_fit.items():
                    st.write(f"**{region}:** {fit}")
            else:
                st.write(geography_fit)
            
            # Improvements Needed with better formatting
            st.subheader("💡 Suggested Improvements")
            improvements = recommendations.get("improvements", ["No specific improvements listed"])
            if isinstance(improvements, list):
                for i, improvement in enumerate(improvements, 1):
                    st.write(f"{i}. {improvement}")
            else:
                st.write(improvements)
            
            # Rigor Assessment with enhanced display
            st.subheader("📊 Rigor Assessment")
            rigor = recommendations.get("rigor", "Not specified")
            if isinstance(rigor, dict):
                for category, assessment in rigor.items():
                    st.write(f"**{category}:** {assessment}")
            else:
                st.write(rigor)
            
            # Add visual separator
            st.markdown("---")
            
            # Summary metrics with error handling
            st.subheader("📈 Overall Scores")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Calculate teaching score
                teaching_data = evaluation.get("teaching", {})
                concept_scores = [
                    concept_data.get(cat, {}).get("score", 0)
                    for cat in ["SubjectMatterAccuracy", "FirstPrinciplesApproach", 
                               "ExamplesAndBusinessContext", "CohesiveStorytelling",
                               "EngagementAndInteraction", "ProfessionalTone"]
                ]
                code_scores = [
                    code_data.get(cat, {}).get("score", 0)
                    for cat in ["DepthOfExplanation", "OutputInterpretation", "BreakingDownComplexity"]
                ]
                
                all_scores = concept_scores + code_scores
                teaching_score = (sum(all_scores) / len(all_scores)) * 100 if all_scores else 0
                st.metric("Teaching Score", f"{teaching_score:.1f}%")
            
            with col2:
                # Calculate communication score
                communication = evaluation.get("communication", {})
                comm_categories = ["speed", "fluency", "flow", "intonation", "energy"]
                comm_scores = [
                    communication.get(cat, {}).get("score", 0)
                    for cat in comm_categories
                ]
                comm_score = (sum(comm_scores) / len(comm_scores)) * 100 if comm_scores else 0
                st.metric("Communication Score", f"{comm_score:.1f}%")
            
            with col3:
                # Calculate overall score (average of teaching and communication)
                overall_score = (teaching_score + comm_score) / 2
                st.metric("Overall Score", f"{overall_score:.1f}%")
            
            # Additional metrics row
            col1, col2 = st.columns(2)
            with col1:
                # Count total improvement points
                improvement_count = len(improvements) if isinstance(improvements, list) else 1
                st.metric("Improvement Points", improvement_count)
            
            with col2:
                # Display words per minute if available
                wpm = communication.get("speed", {}).get("wpm", 0)
                st.metric("Speaking Speed", f"{wpm:.1f} WPM")

        # Transcript tab with error handling
        with tabs[3]:
            st.header("Transcript")
            st.text(evaluation.get("transcript", "Transcript not available"))

    except Exception as e:
        logger.error(f"Error displaying evaluation: {e}")
        st.error(f"Error displaying results: {str(e)}")
        st.error("Please check the evaluation data structure and try again.")

def check_dependencies() -> List[str]:
    """Check if required dependencies are installed"""
    missing = []
    
    if not shutil.which('ffmpeg'):
        missing.append("FFmpeg")
    
    return missing

def main():
    try:
        st.set_page_config(page_title="🎓 Mentor Demo Review System", layout="wide")
        
        st.title("🎓 Mentor Demo Review System")
        
        # Check dependencies with progress
        with st.status("Checking system requirements...") as status:
            progress_bar = st.progress(0)
            
            status.update(label="Checking FFmpeg installation...")
            progress_bar.progress(0.3)
            missing_deps = check_dependencies()
            
            progress_bar.progress(0.6)
            if missing_deps:
                status.update(label="Missing dependencies detected!", state="error")
                st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
                st.markdown("""
                Please install the missing dependencies:
                ```bash
                sudo apt-get update
                sudo apt-get install ffmpeg
                ```
                """)
                return
            
            progress_bar.progress(1.0)
            status.update(label="System requirements satisfied!", state="complete")
        
        with st.sidebar:
            st.header("Instructions")
            st.markdown("""
            1. Upload your teaching video
            2. Wait for the analysis 
            3. Review the detailed feedback
            4. Download the report
            
            **Supported formats:** MP4, AVI, MOV
            **Maximum file size:** 500mb
            """)
            
            st.header("Processing Status")
            st.info("Upload a video to begin analysis")
        
        uploaded_file = st.file_uploader(
            "Upload Teaching Video",
            type=['mp4', 'avi', 'mov'],
            help="Upload your teaching video in MP4, AVI, or MOV format"
        )
        
        if uploaded_file:
            # Create temp directory for processing
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, uploaded_file.name)
            
            try:
                # Save uploaded file with progress
                with st.status("Saving uploaded file...") as status:
                    progress_bar = st.progress(0)
                    
                    # Save in chunks to show progress
                    chunk_size = 1024 * 1024  # 1MB chunks
                    file_size = len(uploaded_file.getbuffer())
                    chunks = file_size // chunk_size + 1
                    
                    with open(video_path, 'wb') as f:
                        for i in range(chunks):
                            start = i * chunk_size
                            end = min(start + chunk_size, file_size)
                            f.write(uploaded_file.getbuffer()[start:end])
                            progress = (i + 1) / chunks
                            status.update(label=f"Saving file: {progress:.1%}")
                            progress_bar.progress(progress)
                    
                    status.update(label="File saved successfully!", state="complete")
                
                # Validate file size
                file_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)  # Size in GB
                if file_size > 2:
                    st.error("File size exceeds 2GB limit. Please upload a smaller file.")
                    return
                
                # Store evaluation results in session state
                if 'evaluation_results' not in st.session_state:
                    # Process video only if results aren't already in session state
                    with st.spinner("Processing video"):
                        evaluator = MentorEvaluator()
                        st.session_state.evaluation_results = evaluator.evaluate_video(video_path)
                
                # Display results using stored evaluation
                st.success("Analysis complete!")
                display_evaluation(st.session_state.evaluation_results)
                
                # Add download button using stored results
                if st.download_button(
                    "📥 Download Full Report",
                    json.dumps(st.session_state.evaluation_results, indent=2),
                    "evaluation_report.json",
                    "application/json",
                    help="Download the complete evaluation report in JSON format"
                ):
                    st.success("Report downloaded successfully!")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                
            finally:
                # Clean up temp files
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
