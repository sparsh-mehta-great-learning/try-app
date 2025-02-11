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
        """Analyze teaching content with more lenient validation and robust JSON handling"""
        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing content analysis...")
                
                prompt = self._create_analysis_prompt(transcript)
                logger.info(f"Attempt {attempt + 1}: Sending analysis request")
                logger.info(f"Transcript length: {len(transcript)} characters")
                
                if progress_callback:
                    progress_callback(0.5, "Processing with AI model...")
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Keeping original model
                        messages=[
                            {"role": "system", "content": """You are a balanced teaching evaluator providing constructive analysis.
                             Score of 1 should be given when MOST criteria below are met with reasonable evidence. 
                             Only default to 0 if MULTIPLE major criteria are missing or severely lacking.
                             
                             Concept Assessment Scoring Criteria:
                             - Subject Matter Accuracy (Score 1 requires: Generally accurate information, minor errors acceptable)
                             - First Principles Approach (Score 1 requires: Some explanation of fundamentals)
                             - Examples and Business Context (Score 1 requires: At least one relevant example)
                             - Cohesive Storytelling (Score 1 requires: Generally logical flow, minor jumps acceptable)
                             - Engagement and Interaction (Score 1 requires: Some attempt at learner engagement)
                             - Professional Tone (Score 1 requires: Generally professional language, occasional casual expressions acceptable)
                             
                             Code Assessment Scoring Criteria:
                             - Depth of Explanation (Score 1 requires: Basic explanation of implementation)
                             - Output Interpretation (Score 1 requires: Some connection between code and business impact)
                             - Breaking down Complexity (Score 1 requires: Attempt to break down concepts)
                             
                             Citations Requirements:
                             - Include timestamps [MM:SS] where possible
                             - Provide at least one example for each category
                             - Focus on constructive feedback
                             
                             Remember that transcripts may contain errors, so focus on the overall teaching effectiveness 
                             rather than minor mistakes or typos.
                             
                             Always respond with valid JSON containing these exact categories."""},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.4  # Slightly higher temperature for more lenient evaluation
                    )
                    logger.info("API call successful")
                except Exception as api_error:
                    logger.error(f"API call failed: {str(api_error)}")
                    raise
                
                result_text = response.choices[0].message.content.strip()
                logger.info(f"Raw API response: {result_text[:500]}...")
                
                try:
                    result = json.loads(result_text)
                    logger.info("Successfully parsed JSON response")
                    
                    # Validate the response structure
                    required_categories = {
                        "Concept Assessment": [
                            "Subject Matter Accuracy",
                            "First Principles Approach",
                            "Examples and Business Context",
                            "Cohesive Storytelling",
                            "Engagement and Interaction",
                            "Professional Tone"
                        ],
                        "Code Assessment": [
                            "Depth of Explanation",
                            "Output Interpretation",
                            "Breaking down Complexity"
                        ]
                    }
                    
                    # Check if response has required structure
                    for category, subcategories in required_categories.items():
                        if category not in result:
                            logger.error(f"Missing category: {category}")
                            raise ValueError(f"Response missing required category: {category}")
                        
                        for subcategory in subcategories:
                            if subcategory not in result[category]:
                                logger.error(f"Missing subcategory: {subcategory} in {category}")
                                raise ValueError(f"Response missing required subcategory: {subcategory}")
                            
                            subcat_data = result[category][subcategory]
                            if not isinstance(subcat_data, dict):
                                logger.error(f"Invalid format for {category}.{subcategory}")
                                raise ValueError(f"Invalid format for {category}.{subcategory}")
                            
                            if "Score" not in subcat_data or "Citations" not in subcat_data:
                                logger.error(f"Missing Score or Citations in {category}.{subcategory}")
                                raise ValueError(f"Missing Score or Citations in {category}.{subcategory}")
                    
                    return result
                    
                except json.JSONDecodeError as json_error:
                    logger.error(f"JSON parsing error: {str(json_error)}")
                    logger.error(f"Invalid JSON response: {result_text}")
                    raise
                except ValueError as val_error:
                    logger.error(f"Validation error: {str(val_error)}")
                    raise
                
            except Exception as e:
                logger.error(f"Content analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_count - 1:
                    logger.error("All attempts failed, returning default structure")
                    return {
                        "Concept Assessment": {
                            "Subject Matter Accuracy": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "First Principles Approach": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Examples and Business Context": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Cohesive Storytelling": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Engagement and Interaction": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Professional Tone": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]}
                        },
                        "Code Assessment": {
                            "Depth of Explanation": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Output Interpretation": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Breaking down Complexity": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]}
                        }
                    }
                time.sleep(self.retry_delay * (2 ** attempt))

    def _create_analysis_prompt(self, transcript: str) -> str:
        """Create the analysis prompt"""
        prompt_template = """Analyze this teaching content and provide detailed assessment with timestamps:

Transcript: {transcript}

Provide a detailed assessment in JSON format with scores (0 or 1) and timestamped citations for each category.
If score is 0, citations should point out problems. If score is 1, citations should highlight good examples.

Required JSON structure:
{{
    "Concept Assessment": {{
        "Subject Matter Accuracy": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "First Principles Approach": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "Examples and Business Context": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "Cohesive Storytelling": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "Engagement and Interaction": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "Professional Tone": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }}
    }},
    "Code Assessment": {{
        "Depth of Explanation": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "Output Interpretation": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }},
        "Breaking down Complexity": {{
            "Score": 1,
            "Citations": ["[MM:SS] Example citation"]
        }}
    }}
}}

Evaluation Criteria:
- Subject Matter Accuracy: Check for factual errors or incorrect correlations
- First Principles Approach: Evaluate if fundamentals are explained before technical terms
- Examples and Business Context: Look for real-world examples
- Cohesive Storytelling: Check for logical flow between topics
- Engagement and Interaction: Evaluate use of questions and engagement techniques
- Professional Tone: Assess language and delivery professionalism
- Depth of Explanation: Evaluate technical explanations
- Output Interpretation: Check if code outputs are explained clearly
- Breaking down Complexity: Assess ability to simplify complex concepts

Important:
- Include timestamps in [MM:SS] format
- Provide specific citations from the transcript
- Use only Score values of 0 or 1
- Include at least one citation for each category"""

        return prompt_template.format(transcript=transcript)

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                           progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = audio_features.get('duration', 0) / 60
            
            # Calculate words per minute with updated range (130-160 WPM is ideal for teaching)
            words_per_minute = max(words / duration_minutes if duration_minutes > 0 else 0, 1)
            
            # Improved filler word detection (2-3 per minute is acceptable)
            filler_words = re.findall(r'\b(um|uh|like|you\s+know|basically|actually|literally)\b', 
                                    transcript.lower())
            fillers_count = len(filler_words)
            fillers_per_minute = max(fillers_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Improved error detection (1-2 per minute is acceptable)
            repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
            incomplete_sentences = len(re.findall(r'[a-zA-Z]+\s*\.\.\.|\b[a-zA-Z]+\s*-\s+', transcript))
            errors_count = repeated_words + incomplete_sentences
            errors_per_minute = max(errors_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Ensure mean amplitude is properly scaled (60-75 dB is ideal for teaching)
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,  # Updated WPM range for teaching
                    "wpm": float(words_per_minute),
                    "total_words": int(words),
                    "duration_minutes": float(duration_minutes)
                },
                "fluency": {
                    "score": 1 if errors_per_minute <= 2 and fillers_per_minute <= 3 else 0,  # Updated thresholds
                    "fillersPerMin": float(fillers_per_minute),
                    "errorsPerMin": float(errors_per_minute)
                },
                "flow": {
                    "score": 1 if float(audio_features.get("pauses_per_minute", 0)) <= 12 else 0,  # Updated pause threshold
                    "pausesPerMin": float(audio_features.get("pauses_per_minute", 0))
                },
                "intonation": {
                    "pitch": float(audio_features.get("pitch_mean", 0)),
                    "pitchScore": 1 if 20 <= float(audio_features.get("pitch_std", 0)) / float(audio_features.get("pitch_mean", 1)) * 100 <= 40 else 0,  # Updated pitch variation range
                    "pitchVariation": float(audio_features.get("pitch_std", 0)),
                    "patternScore": 1 if float(audio_features.get("variations_per_minute", 0)) >= 8 else 0,  # Updated minimum variations
                    "risingPatterns": int(audio_features.get("rising_patterns", 0)),
                    "fallingPatterns": int(audio_features.get("falling_patterns", 0)),
                    "variationsPerMin": float(audio_features.get("variations_per_minute", 0))
                },
                "energy": {
                    "score": 1 if 60 <= mean_amplitude <= 75 else 0,  # Updated amplitude range for teaching
                    "meanAmplitude": mean_amplitude,
                    "amplitudeDeviation": float(audio_features.get("amplitude_deviation", 0)),
                    "variationScore": 1 if 0.05 <= float(audio_features.get("amplitude_deviation", 0)) <= 0.15 else 0  # Updated amplitude variation range
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
                        "rigor": "Undetermined",
                        "profileMatches": []
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
                        "rigor": "Undetermined",
                        "profileMatches": []
                    }
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def _create_recommendation_prompt(self, metrics: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """Create the recommendation prompt"""
        return f"""Based on the following metrics and analysis, provide recommendations:
Metrics: {json.dumps(metrics)}
Content Analysis: {json.dumps(content_analysis)}

Analyze the teaching style and provide:
1. Geography fit assessment
2. Specific improvements needed
3. Profile matching for different learner types (choose ONLY ONE best match)
4. Overall teaching rigor assessment

Required JSON structure:
{{
    "geographyFit": "String describing geographical market fit",
    "improvements": [
        "Array of specific improvement recommendations"
    ],
    "rigor": "Assessment of teaching rigor",
    "profileMatches": [
        {{
            "profile": "junior_technical",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }},
        {{
            "profile": "senior_non_technical",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }},
        {{
            "profile": "junior_expert",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }},
        {{
            "profile": "senior_expert",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }}
    ]
}}

IMPORTANT: Set match=true for ONLY ONE profile that best matches the teaching style. 
All other profiles should have match=false with explanations of why they're not the best fit.

Profile Definitions:
- junior_technical: Low Programming Ex + Low Work Ex
- senior_non_technical: Low Programming Ex + High Work Ex
- junior_expert: High Programming Ex + Low Work Ex
- senior_expert: High Programming Ex + High Work Ex

Evaluation Criteria for Best Match:
1. Teaching Pace:
   - junior_technical/senior_non_technical: Needs slower, detailed explanations
   - junior_expert/senior_expert: Can handle faster, more concise delivery

2. Technical Depth:
   - junior_technical: Basic concepts with lots of examples
   - senior_non_technical: Business context with technical foundations
   - junior_expert: Advanced concepts with implementation details
   - senior_expert: Complex systems and architectural considerations

3. Business Context:
   - junior_technical/junior_expert: Less emphasis needed
   - senior_non_technical/senior_expert: Strong business context required

4. Code Explanation:
   - junior_technical: Step-by-step, basic syntax
   - senior_non_technical: High-level overview with business impact
   - junior_expert: Implementation details and best practices
   - senior_expert: Architecture patterns and system design

Consider:
- Teaching pace and complexity level
- Balance of technical vs business context
- Depth of code explanations
- Use of examples and analogies
- Engagement style"""

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
                logger.info("Attempting to initialize Whisper model...")
                # First try to initialize model with downloading allowed
                self._whisper_model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                    download_root=self.model_cache_dir,
                    local_files_only=False  # Allow downloading if needed
                )
                logger.info("Whisper model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Whisper model: {e}")
                # If first attempt fails, try with local files
                try:
                    logger.info("Attempting to load model from local cache...")
                    self._whisper_model = WhisperModel(
                        "small",
                        device="cpu",
                        compute_type="int8",
                        download_root=self.model_cache_dir,
                        local_files_only=True
                    )
                    logger.info("Model loaded from cache successfully")
                except Exception as cache_error:
                    logger.error(f"Error loading from cache: {cache_error}")
                    raise RuntimeError(
                        "Failed to initialize Whisper model. Please ensure you have "
                        "internet connectivity for the first run to download the model."
                    ) from cache_error
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
        """Transcribe audio with optimized performance using batching and parallel processing"""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading transcription model...")

            # Check if GPU is available and set device accordingly
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            # Generate cache key based on file content
            cache_key = f"transcript_{hash(open(audio_path, 'rb').read())}"
            
            # Check cache first
            if cache_key in st.session_state:
                logger.info("Using cached transcription")
                return st.session_state[cache_key]

            # Initialize model with optimized settings
            model = WhisperModel(
                "small",
                device=device,
                compute_type=compute_type,
                download_root=self.model_cache_dir,
                local_files_only=False,
                cpu_threads=4,  # Adjust based on system capabilities
                num_workers=2   # Adjust based on system capabilities
            )

            # Transcribe with correct parameters for faster-whisper
            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=100
                )
            )

            # Combine segments into final transcript
            transcript = ' '.join(segment.text for segment in segments)
            
            # Cache the result
            st.session_state[cache_key] = transcript

            if progress_callback:
                progress_callback(1.0, "Transcription complete!")

            return transcript

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise

    def _merge_transcripts(self, transcripts: List[str]) -> str:
        """Merge transcripts with overlap deduplication"""
        if not transcripts:
            return ""
        
        def clean_text(text):
            # Remove extra spaces and normalize punctuation
            return ' '.join(text.split())
        
        def find_overlap(text1, text2):
            # Find overlapping text between consecutive chunks
            words1 = text1.split()
            words2 = text2.split()
            
            for i in range(min(len(words1), 20), 0, -1):  # Check up to 20 words
                if ' '.join(words1[-i:]) == ' '.join(words2[:i]):
                    return i
            return 0

        merged = clean_text(transcripts[0])
        
        for i in range(1, len(transcripts)):
            current = clean_text(transcripts[i])
            overlap_size = find_overlap(merged, current)
            merged += ' ' + current.split(' ', overlap_size)[-1]
        
        return merged

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
            
            # Calculate words per minute with updated range (130-160 WPM is ideal for teaching)
            words_per_minute = max(words / duration_minutes if duration_minutes > 0 else 0, 1)
            
            # Improved filler word detection (2-3 per minute is acceptable)
            filler_words = re.findall(r'\b(um|uh|like|you\s+know|basically|actually|literally)\b', 
                                    transcript.lower())
            fillers_count = len(filler_words)
            fillers_per_minute = max(fillers_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Improved error detection (1-2 per minute is acceptable)
            repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
            incomplete_sentences = len(re.findall(r'[a-zA-Z]+\s*\.\.\.|\b[a-zA-Z]+\s*-\s+', transcript))
            errors_count = repeated_words + incomplete_sentences
            errors_per_minute = max(errors_count / duration_minutes if duration_minutes > 0 else 0, 0.1)
            
            # Ensure mean amplitude is properly scaled (60-75 dB is ideal for teaching)
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,  # Updated WPM range for teaching
                    "wpm": float(words_per_minute),
                    "total_words": int(words),
                    "duration_minutes": float(duration_minutes)
                },
                "fluency": {
                    "score": 1 if errors_per_minute <= 2 and fillers_per_minute <= 3 else 0,  # Updated thresholds
                    "fillersPerMin": float(fillers_per_minute),
                    "errorsPerMin": float(errors_per_minute)
                },
                "flow": {
                    "score": 1 if float(audio_features.get("pauses_per_minute", 0)) <= 12 else 0,  # Updated pause threshold
                    "pausesPerMin": float(audio_features.get("pauses_per_minute", 0))
                },
                "intonation": {
                    "pitch": float(audio_features.get("pitch_mean", 0)),
                    "pitchScore": 1 if 20 <= float(audio_features.get("pitch_std", 0)) / float(audio_features.get("pitch_mean", 1)) * 100 <= 40 else 0,  # Updated pitch variation range
                    "pitchVariation": float(audio_features.get("pitch_std", 0)),
                    "patternScore": 1 if float(audio_features.get("variations_per_minute", 0)) >= 8 else 0,  # Updated minimum variations
                    "risingPatterns": int(audio_features.get("rising_patterns", 0)),
                    "fallingPatterns": int(audio_features.get("falling_patterns", 0)),
                    "variationsPerMin": float(audio_features.get("variations_per_minute", 0))
                },
                "energy": {
                    "score": 1 if 60 <= mean_amplitude <= 75 else 0,  # Updated amplitude range for teaching
                    "meanAmplitude": mean_amplitude,
                    "amplitudeDeviation": float(audio_features.get("amplitude_deviation", 0)),
                    "variationScore": 1 if 0.05 <= float(audio_features.get("amplitude_deviation", 0)) <= 0.15 else 0  # Updated amplitude variation range
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
            st.header("Communication Metrics")
            
            metrics = evaluation.get("communication", {})
            
            # Speed Metrics
            with st.expander("🏃 Speed", expanded=True):
                speed = metrics.get("speed", {})
                score = speed.get("score", 0)
                wpm = speed.get("wpm", 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if score == 1 else "❌ Needs Improvement")
                    st.metric("Words per Minute", f"{wpm:.1f}")
                with col2:
                    st.info("**Acceptable Range:** 120-180 WPM")
            
            # Fluency Metrics
            with st.expander("🗣️ Fluency", expanded=True):
                fluency = metrics.get("fluency", {})
                score = fluency.get("score", 0)
                fpm = fluency.get("fillersPerMin", 0)
                epm = fluency.get("errorsPerMin", 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if score == 1 else "❌ Needs Improvement")
                    st.metric("Fillers per Minute", f"{fpm:.1f}")
                    st.metric("Errors per Minute", f"{epm:.1f}")
                with col2:
                    st.info("""
                    **Acceptable Ranges:**
                    - Fillers: <=3 FPM
                    - Errors: <=2 EPM
                    """)
            
            # Flow Metrics
            with st.expander("🌊 Flow", expanded=True):
                flow = metrics.get("flow", {})
                score = flow.get("score", 0)
                ppm = flow.get("pausesPerMin", 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if score == 1 else "❌ Needs Improvement")
                    st.metric("Pauses per Minute", f"{ppm:.1f}")
                with col2:
                    st.info("**Acceptable Range:** < 12 PPM")
            
            # Intonation Metrics
            with st.expander("🎵 Intonation", expanded=True):
                intonation = metrics.get("intonation", {})
                pitch_score = intonation.get("pitchScore", 0)
                pattern_score = intonation.get("patternScore", 0)
                pitch = intonation.get("pitch", 0)
                pitch_variation = intonation.get("pitchVariation", 0)
                rising = intonation.get("risingPatterns", 0)
                falling = intonation.get("fallingPatterns", 0)
                variations = intonation.get("variationsPerMin", 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pitch Score", "✅ Pass" if pitch_score == 1 else "❌ Needs Improvement")
                    st.metric("Pattern Score", "✅ Pass" if pattern_score == 1 else "❌ Needs Improvement")
                    st.metric("Frequency / Pitch", f"{pitch:.1f} Hz")
                    st.metric("Pitch Variation (σ)", f"{pitch_variation:.1f} Hz")
                    st.metric("Rising Patterns", rising)
                    st.metric("Falling Patterns", falling)
                    st.metric("Variations per Minute", f"{variations:.1f}")
                with col2:
                    st.info("""
                    **Acceptable Ranges:**
                    - Pitch Variation: 20-40% from baseline
                    - Variations per Minute: >8
                    """)
            
            # Energy Metrics
            with st.expander("⚡ Energy", expanded=True):
                energy = metrics.get("energy", {})
                score = energy.get("score", 0)
                amplitude = energy.get("meanAmplitude", 0)
                deviation = energy.get("amplitudeDeviation", 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if score == 1 else "❌ Needs Improvement")
                    st.metric("Mean Amplitude", f"{amplitude:.1f}")
                    st.metric("Amplitude Deviation (σ/μ)", f"{deviation:.2f}")
                with col2:
                    st.info("""
                    **Acceptable Ranges:**
                    - Mean Amplitude: 60-75 dB
                    - Amplitude Deviation: 0.05-0.15
                    """)

        with tabs[1]:
            st.header("Teaching Analysis")
            
            # Get teaching data
            teaching_data = evaluation.get("teaching", {})
            
            # Concept Assessment Section
            st.subheader("Concept Assessment")
            
            concept_assessment = teaching_data.get("Concept Assessment", {})
            
            # Define concept categories with descriptions
            concept_categories = [
                ("Subject Matter Accuracy", "Is the mentor making any factual errors when teaching / making any wrong assumptions or wrong conclusions / correlations - is he talking in the air"),
                ("First Principles Approach", "You talk about the WHY first and build an understanding / intuition of the topic before you introduce technical jargons / terminologies"),
                ("Examples and Business Context", "In relevant areas, you support your teaching with business examples and context"),
                ("Cohesive Storytelling", "Is the mentor linking between topics or jumping around"),
                ("Engagement and Interaction", "Is the mentor speaking to himself or is he asking learners thought provoking questions to inspire understanding"),
                ("Professional Tone", "Is the mentor speaking in a casual or professional tone / is he using words that are unprofessional")
            ]
            
            # Display each concept category
            for category, description in concept_categories:
                with st.expander(f"{category} - Score: {'Pass' if concept_assessment.get(category, {}).get('Score', 0) == 1 else 'Needs Improvement'}", expanded=True):
                    st.caption(description)
                    
                    assessment = concept_assessment.get(category, {"Score": 0, "Citations": []})
                    score = assessment.get("Score", 0)
                    citations = assessment.get("Citations", [])
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        # Display score with color and icon
                        score_color = "green" if score == 1 else "red"
                        score_icon = "✅" if score == 1 else "❌"
                        st.markdown(f"**Score:** :{score_color}[{score_icon} {'Pass' if score == 1 else 'Needs Improvement'}]")
                    
                    with col2:
                        # Display citations in a formatted box
                        if citations:
                            st.markdown("**Key Observations:**")
                            for citation in citations:
                                st.markdown(f"""
                                <div class="citation-box">
                                    🔍 {citation}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No observations available")
            
            # Code Assessment Section
            st.subheader("Code Assessment")
            
            code_assessment = teaching_data.get("Code Assessment", {})
            
            # Define code categories with descriptions
            code_categories = [
                ("Depth of Explanation", "Is code just being read out or is the mentor talking about syntaxes, usage of libraries and why, functions, parameters and method (and what they do)"),
                ("Output Interpretation", "Are the outputs of the code being interpreted in the light of a business context and outcome"),
                ("Breaking down Complexity", "Is the mentor able to break down a code into simpler sections of code blocks / modules and explain their purpose and logical flow")
            ]
            
            # Display each code category
            for category, description in code_categories:
                with st.expander(f"{category} - Score: {'Pass' if code_assessment.get(category, {}).get('Score', 0) == 1 else 'Needs Improvement'}", expanded=True):
                    st.caption(description)
                    
                    assessment = code_assessment.get(category, {"Score": 0, "Citations": []})
                    score = assessment.get("Score", 0)
                    citations = assessment.get("Citations", [])
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        score_color = "green" if score == 1 else "red"
                        score_icon = "✅" if score == 1 else "❌"
                        st.markdown(f"**Score:** :{score_color}[{score_icon} {'Pass' if score == 1 else 'Needs Improvement'}]")
                    
                    with col2:
                        if citations:
                            st.markdown("**Key Observations:**")
                            for citation in citations:
                                st.markdown(f"""
                                <div class="citation-box">
                                    💻 {citation}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No observations available")
            
            # Teaching Summary Section with metrics
            st.subheader("Teaching Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                concept_scores = [assessment.get("Score", 0) for category in concept_assessment.values() for assessment in [category]]
                concept_score = (sum(concept_scores) / len(concept_scores) * 100) if concept_scores else 0
                st.metric("Concept Score", f"{concept_score:.1f}%", 
                         delta="Pass" if concept_score >= 70 else "Needs Improvement",
                         delta_color="normal" if concept_score >= 70 else "inverse")
            
            with col2:
                code_scores = [assessment.get("Score", 0) for category in code_assessment.values() for assessment in [category]]
                code_score = (sum(code_scores) / len(code_scores) * 100) if code_scores else 0
                st.metric("Code Score", f"{code_score:.1f}%",
                         delta="Pass" if code_score >= 70 else "Needs Improvement",
                         delta_color="normal" if code_score >= 70 else "inverse")
            
            with col3:
                total_scores = concept_scores + code_scores
                overall_score = (sum(total_scores) / len(total_scores) * 100) if total_scores else 0
                st.metric("Overall Teaching Score", f"{overall_score:.1f}%",
                         delta="Pass" if overall_score >= 70 else "Needs Improvement",
                         delta_color="normal" if overall_score >= 70 else "inverse")

        with tabs[2]:
            st.header("Recommendations")
            
            recommendations = evaluation.get("recommendations", {})
            
            # Calculate Overall Score
            communication_metrics = evaluation.get("communication", {})
            teaching_data = evaluation.get("teaching", {})
            
            # Calculate Communication Score
            comm_scores = []
            for category in ["speed", "fluency", "flow", "intonation", "energy"]:
                if category in communication_metrics:
                    if "score" in communication_metrics[category]:
                        comm_scores.append(communication_metrics[category]["score"])
            
            communication_score = (sum(comm_scores) / len(comm_scores) * 100) if comm_scores else 0
            
            # Calculate Teaching Score (combining concept and code assessment)
            concept_assessment = teaching_data.get("Concept Assessment", {})
            code_assessment = teaching_data.get("Code Assessment", {})
            
            teaching_scores = []
            # Add concept scores
            for category in concept_assessment.values():
                if isinstance(category, dict) and "Score" in category:
                    teaching_scores.append(category["Score"])
            
            # Add code scores
            for category in code_assessment.values():
                if isinstance(category, dict) and "Score" in category:
                    teaching_scores.append(category["Score"])
            
            teaching_score = (sum(teaching_scores) / len(teaching_scores) * 100) if teaching_scores else 0
            
            # Calculate Overall Score (50-50 weight between communication and teaching)
            overall_score = (communication_score + teaching_score) / 2
            
            # Display Overall Scores at the top of recommendations
            st.markdown("### 📊 Overall Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Communication Score",
                    f"{communication_score:.1f}%",
                    delta="Pass" if communication_score >= 70 else "Needs Improvement",
                    delta_color="normal" if communication_score >= 70 else "inverse"
                )
            
            with col2:
                st.metric(
                    "Teaching Score",
                    f"{teaching_score:.1f}%",
                    delta="Pass" if teaching_score >= 70 else "Needs Improvement",
                    delta_color="normal" if teaching_score >= 70 else "inverse"
                )
            
            with col3:
                st.metric(
                    "Overall Score",
                    f"{overall_score:.1f}%",
                    delta="Pass" if overall_score >= 70 else "Needs Improvement",
                    delta_color="normal" if overall_score >= 70 else "inverse"
                )
            
            # Continue with existing recommendations display
            with st.expander("💡 Areas for Improvement", expanded=True):
                improvements = recommendations.get("improvements", [])
                if isinstance(improvements, list):
                    for i, improvement in enumerate(improvements, 1):
                        if isinstance(improvement, str):
                            category = "General"
                            icon = "🎯"
                            if any(keyword in improvement.lower() for keyword in ["voice", "volume", "pitch", "pace"]):
                                category = "Voice and Delivery"
                                icon = "🗣️"
                            elif any(keyword in improvement.lower() for keyword in ["explain", "concept", "understanding"]):
                                category = "Teaching Approach"
                                icon = "📚"
                            elif any(keyword in improvement.lower() for keyword in ["engage", "interact", "question"]):
                                category = "Engagement"
                                icon = "🤝"
                            elif any(keyword in improvement.lower() for keyword in ["code", "technical", "implementation"]):
                                category = "Technical Content"
                                icon = "💻"
                            
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{icon} {i}. {category}</h4>
                                <p>{improvement}</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Profile Matching with enhanced formatting
            with st.expander("👥 Profile Matching", expanded=True):
                st.markdown("""
                    <div class="profile-guide">
                        <h4>🎯 Profile Matching Guide</h4>
                        <p>Analyzing mentor's teaching style compatibility with different learner profiles</p>
                    </div>
                """, unsafe_allow_html=True)
                
                profiles = {
                    "junior_technical": {
                        "title": "Junior Technical",
                        "description": "Low Programming Ex + Low Work Ex",
                        "icon": "👨‍💻",
                        "characteristics": [
                            "Young professionals starting their technical career",
                            "Need engaging, fast-paced content",
                            "Prefer hands-on examples"
                        ]
                    },
                    "senior_non_technical": {
                        "title": "Senior Non-Technical",
                        "description": "Low Programming Ex + High Work Ex",
                        "icon": "👨‍💼",
                        "characteristics": [
                            "Experienced professionals transitioning to technical roles",
                            "Need slower pace with detailed explanations",
                            "Appreciate business context"
                        ]
                    },
                    "junior_expert": {
                        "title": "Junior Expert",
                        "description": "High Programming Ex + Low Work Ex",
                        "icon": "🚀",
                        "characteristics": [
                            "Technical experts early in their career",
                            "Prefer fast-paced, advanced content",
                            "Focus on technical depth"
                        ]
                    },
                    "senior_expert": {
                        "title": "Senior Expert",
                        "description": "High Programming Ex + High Work Ex",
                        "icon": "🎯",
                        "characteristics": [
                            "Seasoned technical professionals",
                            "Expect advanced concepts with business context",
                            "Value efficient, precise delivery"
                        ]
                    }
                }
                
                profile_matches = recommendations.get("profileMatches", [])
                
                # Find the recommended profile
                recommended_profile = next(
                    (match["profile"] for match in profile_matches if match.get("match", False)),
                    None
                )
                
                for profile_key, profile_data in profiles.items():
                    match_info = next(
                        (match for match in profile_matches if match["profile"] == profile_key),
                        None
                    )
                    # Only mark as recommended if it matches the single recommended profile
                    is_recommended = profile_key == recommended_profile
                    
                    st.markdown(f"""
                    <div class="profile-card {'recommended' if is_recommended else ''}">
                        <div class="profile-header">
                            <h5>{profile_data['icon']} {profile_data['title']}</h5>
                            <span class="profile-badge {profile_key}">
                                {profile_data['description']}
                            </span>
                        </div>
                        <div class="profile-content">
                            <p><strong>Characteristics:</strong></p>
                            <ul>
                                {''.join(f'<li>{char}</li>' for char in profile_data['characteristics'])}
                            </ul>
                            <div class="recommendation-status {'recommended' if is_recommended else ''}">
                                {('✅ Recommended' if is_recommended else '⚠️ Not Optimal')}<br>
                                <small>{match_info['reason'] if match_info else 'Profile not matched based on teaching style'}</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Transcript tab with error handling
        with tabs[3]:
            st.header("Transcript")
            st.text(evaluation.get("transcript", "Transcript not available"))

    except Exception as e:
        logger.error(f"Error displaying evaluation: {e}")
        st.error(f"Error displaying results: {str(e)}")
        st.error("Please check the evaluation data structure and try again.")

    # Add these styles to the existing CSS in the main function
    st.markdown("""
        <style>
        /* ... existing styles ... */
        
        .citation-box {
            background-color: #f8f9fa;
            border-left: 3px solid #6c757d;
            padding: 10px;
            margin: 5px 0;
            border-radius: 0 4px 4px 0;
        }
        
        .recommendation-card {
            background-color: #ffffff;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .recommendation-card h4 {
            color: #1f77b4;
            margin: 0 0 10px 0;
        }
        
        .rigor-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .score-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .green-score {
            background-color: #28a745;
            color: white;
        }
        
        .orange-score {
            background-color: #fd7e14;
            color: white;
        }
        
        .metric-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .profile-guide {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #1f77b4;
        }
        
        .profile-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .profile-card.recommended {
            border-left: 4px solid #28a745;
        }
        
        .profile-header {
            margin-bottom: 15px;
        }
        
        .profile-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            margin-top: 5px;
            background-color: #f8f9fa;
        }
        
        .profile-content ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .recommendation-status {
            margin-top: 15px;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .recommendation-status small {
            display: block;
            margin-top: 5px;
            font-weight: normal;
            color: #666;
        }
        
        .recommendation-status.recommended {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        
        .recommendation-status:not(.recommended) {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404;
        }
        
        .profile-card.recommended {
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.1);
        }
        
        .profile-card:not(.recommended) {
            border-left: 4px solid #ffc107;
            opacity: 0.8;
        }
        
        .profile-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def check_dependencies() -> List[str]:
    """Check if required dependencies are installed"""
    missing = []
    
    if not shutil.which('ffmpeg'):
        missing.append("FFmpeg")
    
    return missing

def generate_pdf_report(evaluation_data: Dict[str, Any]) -> bytes:
    """Generate a formatted PDF report from evaluation data"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Mentor Demo Evaluation Report", title_style))
        story.append(Spacer(1, 20))
        
        # Communication Metrics Section
        story.append(Paragraph("Communication Metrics", styles['Heading2']))
        comm_metrics = evaluation_data.get("communication", {})
        
        # Create tables for each metric category
        for category in ["speed", "fluency", "flow", "intonation", "energy"]:
            if category in comm_metrics:
                metrics = comm_metrics[category]
                story.append(Paragraph(category.title(), styles['Heading3']))
                
                data = [[k.replace('_', ' ').title(), str(v)] for k, v in metrics.items()]
                t = Table(data, colWidths=[200, 200])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
                story.append(Spacer(1, 20))
        
        # Teaching Analysis Section
        story.append(Paragraph("Teaching Analysis", styles['Heading2']))
        teaching_data = evaluation_data.get("teaching", {})
        
        for assessment_type in ["Concept Assessment", "Code Assessment"]:
            if assessment_type in teaching_data:
                story.append(Paragraph(assessment_type, styles['Heading3']))
                categories = teaching_data[assessment_type]
                
                for category, details in categories.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    data = [
                        [category, "Score: " + ("Pass" if score == 1 else "Needs Improvement")],
                        ["Citations:", ""]
                    ] + [["-", citation] for citation in citations]
                    
                    t = Table(data, colWidths=[200, 300])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = evaluation_data.get("recommendations", {})
        
        if "improvements" in recommendations:
            story.append(Paragraph("Areas for Improvement:", styles['Heading3']))
            for improvement in recommendations["improvements"]:
                story.append(Paragraph("• " + improvement, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise RuntimeError(f"Failed to generate PDF report: {str(e)}")

def main():
    try:
        # Set page config must be the first Streamlit command
        st.set_page_config(page_title="🎓 Mentor Demo Review System", layout="wide")
        
        # Add custom CSS for animations and styling
        st.markdown("""
            <style>
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                @keyframes slideIn {
                    from { transform: translateX(-100%); }
                    to { transform: translateX(0); }
                }
                
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                
                .fade-in {
                    animation: fadeIn 1s ease-in;
                }
                
                .slide-in {
                    animation: slideIn 0.5s ease-out;
                }
                
                .pulse {
                    animation: pulse 2s infinite;
                }
                
                .metric-card {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }
                
                .metric-card:hover {
                    transform: translateY(-5px);
                }
                
                .stButton>button {
                    transition: all 0.3s ease;
                }
                
                .stButton>button:hover {
                    transform: scale(1.05);
                }
                
                .category-header {
                    background: linear-gradient(90deg, #1f77b4, #2c3e50);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                
                .score-badge {
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-weight: bold;
                }
                
                .score-pass {
                    background-color: #28a745;
                    color: white;
                }
                
                .score-fail {
                    background-color: #dc3545;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)

        # Sidebar with instructions and status
        with st.sidebar:
            st.markdown("""
                <div class="slide-in">
                    <h2>Instructions</h2>
                    <ol>
                        <li>Upload your teaching video</li>
                        <li>Wait for the analysis</li>
                        <li>Review the detailed feedback</li>
                        <li>Download the report</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            
            # Add file format information separately
            st.markdown("**Supported formats:** MP4, AVI, MOV")
            st.markdown("**Maximum file size:** 500MB")
            
            # Create a placeholder for status updates in the sidebar
            status_placeholder = st.empty()
            status_placeholder.info("Upload a video to begin analysis")

        # Main title (only once)
        st.markdown("""
            <div class="fade-in">
                <h1 style='text-align: center; color: #1f77b4;'>
                    🎓 Mentor Demo Review System
                </h1>
            </div>
        """, unsafe_allow_html=True)

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

        # Temporary: Add radio button for input type selection
        input_type = st.radio(
            "Select Input Type (Temporary Feature)",
            ["Video Only", "Video + Transcript"],
            help="Temporary feature: Choose to upload video only or video with transcript"
        )

        uploaded_file = st.file_uploader(
            "Upload Teaching Video",
            type=['mp4', 'avi', 'mov'],
            help="Upload your teaching video in MP4, AVI, or MOV format"
        )

        # Temporary: Add transcript uploader if Video + Transcript is selected
        uploaded_transcript = None
        if input_type == "Video + Transcript":
            uploaded_transcript = st.file_uploader(
                "Upload Transcript (Optional)",
                type=['txt'],
                help="Upload your transcript in TXT format"
            )
        
        if uploaded_file:
            # Update status in sidebar
            status_placeholder.info("Video uploaded, beginning processing...")
            
            # Add a pulsing animation while processing
            st.markdown("""
                <div class="pulse" style="text-align: center;">
                    <h3>Processing your video...</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Create temp directory for processing
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, uploaded_file.name)
            
            try:
                # Save uploaded file with progress
                with st.status("Saving uploaded file...") as status:
                    # Update sidebar status
                    status_placeholder.info("Saving uploaded file...")
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
                    # Update sidebar status
                    status_placeholder.info("Processing video and generating analysis...")
                    
                    # Process video only if results aren't already in session state
                    with st.spinner("Processing video"):
                        evaluator = MentorEvaluator()
                        
                        # Temporary: Handle transcript if provided
                        if uploaded_transcript:
                            transcript_text = uploaded_transcript.getvalue().decode('utf-8')
                            # Extract audio features but skip transcription
                            audio_features = evaluator.feature_extractor.extract_features(video_path)
                            
                            # Evaluate speech metrics
                            speech_metrics = evaluator._evaluate_speech_metrics(
                                transcript_text,
                                audio_features
                            )
                            
                            # Analyze content
                            content_analysis = evaluator.content_analyzer.analyze_content(transcript_text)
                            
                            # Generate recommendations
                            recommendations = evaluator.recommendation_generator.generate_recommendations(
                                speech_metrics,
                                content_analysis
                            )
                            
                            # Combine results
                            st.session_state.evaluation_results = {
                                "communication": speech_metrics,
                                "teaching": content_analysis,
                                "recommendations": recommendations,
                                "transcript": transcript_text
                            }
                        else:
                            # Original flow: full video evaluation
                            st.session_state.evaluation_results = evaluator.evaluate_video(video_path)
                
                # Update sidebar status for completion
                status_placeholder.success("Analysis complete! Review results below.")
                
                # Display results using stored evaluation
                st.success("Analysis complete!")
                display_evaluation(st.session_state.evaluation_results)
                
                # Add download options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.download_button(
                        "📥 Download JSON Report",
                        json.dumps(st.session_state.evaluation_results, indent=2),
                        "evaluation_report.json",
                        "application/json",
                        help="Download the raw evaluation data in JSON format"
                    ):
                        st.success("JSON report downloaded successfully!")
                
                with col2:
                    if st.download_button(
                        "📄 Download Full Report (PDF)",
                        generate_pdf_report(st.session_state.evaluation_results),
                        "evaluation_report.pdf",
                        "application/pdf",
                        help="Download a formatted PDF report with detailed analysis"
                    ):
                        st.success("PDF report downloaded successfully!")
                
            except Exception as e:
                # Update sidebar status for error
                status_placeholder.error(f"Error during processing: {str(e)}")
                st.error(f"Error during evaluation: {str(e)}")
                
            finally:
                # Clean up temp files
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
