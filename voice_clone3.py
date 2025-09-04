import os
import re
import time
import tempfile
import numpy as np
import soundfile as sf
import librosa
import warnings
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from pydub import AudioSegment
import gc
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class CoquiXTTSVoiceCloner:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
        """
        Initialize Coqui XTTS-v2 for voice cloning
        
        Args:
            model_name: XTTS model to use
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize TTS
        self.tts = TTS(model_name=model_name)
        
        # Supported languages for XTTS-v2
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", 
            "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
        ]
        print(f"Supported languages: {', '.join(self.supported_languages)}")
    
    def preprocess_reference_audio(self, audio_path: str, output_path: Optional[str] = None, 
                                 duration_limit: int = 30) -> str:
        """
        Preprocess reference audio for XTTS voice cloning
        
        Args:
            audio_path: Path to reference audio
            output_path: Output path for preprocessed audio
            duration_limit: Maximum duration in seconds (XTTS works best with 6-30 seconds)
        """
        print(f"Preprocessing reference audio: {audio_path}")
        
        # Load audio using pydub for better format support
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Convert to 22kHz (XTTS optimal sample rate)
            audio_segment = audio_segment.set_frame_rate(22050)
            
            # Limit duration (XTTS works best with shorter clips)
            max_duration_ms = duration_limit * 1000
            if len(audio_segment) > max_duration_ms:
                # Take middle portion for better voice characteristics
                start_ms = (len(audio_segment) - max_duration_ms) // 2
                audio_segment = audio_segment[start_ms:start_ms + max_duration_ms]
            
            # Ensure minimum duration (at least 3 seconds)
            if len(audio_segment) < 3000:
                print("Warning: Reference audio is very short. This may affect cloning quality.")
            
            # Normalize audio
            audio_segment = audio_segment.normalize()
            
            # Save preprocessed audio
            if output_path is None:
                base_name = os.path.splitext(audio_path)[0]
                output_path = f"{base_name}_xtts_preprocessed.wav"
            
            audio_segment.export(output_path, format="wav")
            
            print(f"Preprocessed audio saved to: {output_path}")
            print(f"Duration: {len(audio_segment) / 1000:.2f} seconds")
            print(f"Sample rate: {audio_segment.frame_rate} Hz")
            
            return output_path
            
        except Exception as e:
            print(f"Error preprocessing audio with pydub: {e}")
            # Fallback to librosa/soundfile
            return self._preprocess_with_librosa(audio_path, output_path, duration_limit)
    
    def _preprocess_with_librosa(self, audio_path: str, output_path: str, duration_limit: int) -> str:
        """Fallback preprocessing with librosa"""
        try:
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Limit duration
            max_samples = duration_limit * sr
            if len(audio) > max_samples:
                start_idx = (len(audio) - max_samples) // 2
                audio = audio[start_idx:start_idx + max_samples]
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            if output_path is None:
                base_name = os.path.splitext(audio_path)[0]
                output_path = f"{base_name}_xtts_preprocessed.wav"
            
            sf.write(output_path, audio, sr)
            print(f"Fallback preprocessing completed: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in fallback preprocessing: {e}")
            return audio_path
    
    def analyze_reference_audio(self, audio_path: str):
        """Analyze reference audio characteristics"""
        try:
            audio, sr = librosa.load(audio_path, sr=22050)
            duration = len(audio) / sr
            
            # Basic audio analysis
            rms_energy = librosa.feature.rms(y=audio)[0]
            energy_mean = np.mean(rms_energy)
            energy_std = np.std(rms_energy)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_mean = np.mean(spectral_centroid)
            
            print(f"Reference Audio Analysis:")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Energy (mean): {energy_mean:.4f}")
            print(f"  Energy (std): {energy_std:.4f}")
            print(f"  Spectral Centroid (mean): {spectral_mean:.2f} Hz")
            
            # Quality recommendations
            if duration < 6:
                print("  ⚠️ Warning: Audio is shorter than 6 seconds. Consider using a longer clip.")
            elif duration > 30:
                print("  ⚠️ Info: Audio is longer than 30 seconds. It will be trimmed.")
            else:
                print("  ✅ Audio duration is optimal for XTTS.")
            
            if energy_std / energy_mean < 0.1:
                print("  ⚠️ Info: Audio has low variation. Consider using a more expressive clip.")
            else:
                print("  ✅ Audio has good variation for voice cloning.")
                
        except Exception as e:
            print(f"Error analyzing audio: {e}")
    
    def generate_speech(self, text: str, reference_audio_path: str, 
                       target_language: str = "en", output_path: Optional[str] = None,
                       speed: float = 1.0, temperature: float = 0.7):
        """
        Generate speech using XTTS with voice cloning
        """
        if target_language not in self.supported_languages:
            raise ValueError(f"Language '{target_language}' not supported. Supported: {self.supported_languages}")

        # Check if XTTS model is loaded
        if not hasattr(self.tts, "tts") or "xtts" not in getattr(self.tts, "model_name", "").lower():
            print("❌ Error: XTTS model is not loaded. Please ensure you are using 'tts_models/multilingual/multi-dataset/xtts_v2'.")
            return None, None

        try:
            print(f"Generating speech for: {text[:50]}...")

            # Generate audio
            wav = self.tts.tts(
                text=text,
                speaker_wav=reference_audio_path,
                language=target_language,
                speed=speed
            )

            # Convert to numpy array if it isn't already
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Save audio
            if output_path is None:
                output_path = f"xtts_output_{int(time.time())}.wav"

            # XTTS outputs at 24kHz by default
            sf.write(output_path, wav, 24000)

            return output_path, wav

        except RuntimeError as e:
            if "device-side assert" in str(e):
                print(f"? A CUDA error occurred: {e}")
                print("This can be due to memory issues. Try reducing batch size or sequence length.")
            else:
                print(f"? A runtime error occurred: {e}")
            return None, None
        except Exception as e:
            print(f"? Error generating speech: {e}")
            print("If you see an error about 'generate', your TTS library may be using the wrong backend. Please update Coqui TTS and ensure XTTS is installed.")
            return None, None
    
    def parse_srt_blocks(self, srt_path: str):
        """Parse SRT subtitle file"""
        blocks = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        srt_entries = re.split(r"\n\n", content.strip())
        for entry in srt_entries:
            lines = entry.strip().split("\n")
            if len(lines) >= 3:
                index = lines[0]
                times = lines[1]
                text = " ".join(lines[2:]).strip()
                start, end = times.split(" --> ")
                blocks.append((index, start, end, text))
        return blocks
    
    def srt_time_to_ms(self, time_str):
        """Convert SRT time format to milliseconds"""
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split(',')
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)
    
    def generate_tts_for_srt(self, srt_path: str, reference_audio_path: str,
                            target_language: str = "en", max_duration_sec: Optional[float] = None,
                            speed: float = 1.0, temperature: float = 0.7,
                            preprocess_reference: bool = True):
        """
        Generate TTS for entire SRT file using XTTS voice cloning
        """
        
        # Preprocess reference audio if requested
        if preprocess_reference:
            processed_ref_path = self.preprocess_reference_audio(reference_audio_path)
            self.analyze_reference_audio(processed_ref_path)
        else:
            processed_ref_path = reference_audio_path
        
        # Parse SRT
        blocks = self.parse_srt_blocks(srt_path)
        if not blocks:
            raise ValueError("No subtitle blocks found in SRT file")
        
        print(f"Processing {len(blocks)} subtitle blocks...")
        
        # Generate audio for each block
        audio_segments = []
        sample_rate = 24000  # XTTS default
        total_duration_sec = 0.0
        
        for idx, start, end, text in blocks:
            if not text.strip():
                continue

            start_time_ms = self.srt_time_to_ms(start)
            end_time_ms = self.srt_time_to_ms(end)
            seg_duration_sec = (end_time_ms - start_time_ms) / 1000.0

            # Check duration limit
            if max_duration_sec and total_duration_sec + seg_duration_sec > max_duration_sec:
                print(f"Reached max duration ({max_duration_sec}s). Stopping.")
                break

            print(f"Processing block {idx}: {text[:50]}... [{start} - {end}]")

            try:
                # Generate speech for this segment
                temp_output = f"temp_segment_{idx}.wav"
                output_path, wav_data = self.generate_speech(
                    text=text,
                    reference_audio_path=processed_ref_path,
                    target_language=target_language,
                    output_path=temp_output,
                    speed=speed,
                    temperature=temperature
                )

                if wav_data is not None:
                    audio_segments.append({
                        'start_ms': start_time_ms,
                        'end_ms': end_time_ms,
                        'audio_data': wav_data,
                        'temp_file': temp_output
                    })
                    total_duration_sec += seg_duration_sec
                else:
                    print(f"Failed to generate audio for block {idx}")

            except Exception as e:
                print(f"Error processing block {idx}: {e}")
                continue

            # Clear GPU memory periodically
            try:
                idx_int = int(idx)
            except Exception:
                idx_int = 0
            if idx_int % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        if not audio_segments:
            raise ValueError("No audio segments were generated successfully")
        
        # Combine all segments into final audio
        print("Combining audio segments...")
        
        # Calculate total duration needed
        total_duration_ms = max(seg['end_ms'] for seg in audio_segments)
        total_samples = int((total_duration_ms / 1000.0) * sample_rate)
        
        # Create final audio array
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Place each segment at correct position
        for segment in audio_segments:
            start_sample = int((segment['start_ms'] / 1000.0) * sample_rate)
            audio_data = segment['audio_data']
            
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            end_sample = start_sample + len(audio_data)
            
            # Ensure we don't exceed bounds
            if end_sample > len(final_audio):
                audio_data = audio_data[:len(final_audio) - start_sample]
                end_sample = len(final_audio)
            
            if start_sample < len(final_audio):
                final_audio[start_sample:end_sample] = audio_data
            
            # Clean up temp file
            try:
                os.remove(segment['temp_file'])
            except:
                pass
        
        # Save final combined audio
        output_dir = os.path.dirname(srt_path)
        srt_filename = os.path.splitext(os.path.basename(srt_path))[0]
        final_output_path = os.path.join(output_dir, f"{srt_filename}_xtts_cloned.wav")
        
        sf.write(final_output_path, final_audio, sample_rate)
        
        print(f"✅ Successfully generated XTTS cloned audio!")
        print(f"Output file: {final_output_path}")
        print(f"Total duration: {len(final_audio) / sample_rate:.2f} seconds")
        
        # Clean up preprocessed reference if it was created
        if preprocess_reference and processed_ref_path != reference_audio_path:
            try:
                os.remove(processed_ref_path)
                print("Cleaned up preprocessed reference audio")
            except:
                pass
        
        return final_output_path
    
    def test_installation(self):
        """Test XTTS installation and basic functionality"""
        try:
            print("Testing XTTS installation...")
            print(f"Device: {self.device}")
            print(f"Model loaded: {self.tts.model_name}")
            
            # Test simple synthesis
            test_text = "Hello, this is a test of XTTS voice synthesis."
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                test_output_path = fp.name
            self.tts.tts_to_file(text=test_text, file_path=test_output_path, speaker_wav=reference_audio_path, language=target_language)
            if os.path.exists(test_output_path):
                os.remove(test_output_path)
                print("XTTS installation test completed successfully!")
                return True
            else:
                print("XTTS installation test failed: Could not generate test audio file.")
                return False
            
        except Exception as e:
            print(f"XTTS installation test failed: {e}")
            return False

def main():
    """Main execution function"""
    
    # Configuration
    REFERENCE_AUDIO = os.path.join("outputs", "Live_Bank_", "Live_Bank_Nifty_Option_Trading_____Intra.mp4")
    SRT_FILE = os.path.join("outputs", "Live_Bank_", "Live_Bank_Nifty_Option_Trading_____Intra.srt")
    TARGET_LANGUAGE = "en"  # English output
    MAX_DURATION = 30.0  # Process first 30 seconds

    if not os.path.exists(REFERENCE_AUDIO):
        print(f"? File not found: {REFERENCE_AUDIO}")
        print("Please make sure the reference audio file exists.")
        return

    if not os.path.exists(SRT_FILE):
        print(f"? File not found: {SRT_FILE}")
        print("Please make sure the SRT file exists.")
        return
    
    try:
        # Initialize XTTS
        print("Initializing Coqui XTTS-v2...")
        xtts_cloner = CoquiXTTSVoiceCloner()
        
        # Test installation
        if not xtts_cloner.test_installation():
            print("XTTS installation test failed. Please check your setup.")
            return
        
        print("\n" + "="*60)
        print("CROSS-LANGUAGE VOICE CLONING: Hindi Reference → English Speech")
        print("="*60)
        print(f"Reference audio: {REFERENCE_AUDIO}")
        print(f"SRT file: {SRT_FILE}")
        print(f"Target language: {TARGET_LANGUAGE}")
        print(f"Max duration: {MAX_DURATION}s")
        print()
        
        # Generate cloned voice audio
        output_path = xtts_cloner.generate_tts_for_srt(
            srt_path=SRT_FILE,
            reference_audio_path=REFERENCE_AUDIO,
            target_language=TARGET_LANGUAGE,
            max_duration_sec=MAX_DURATION,
            speed=1.0,  # Normal speed
            temperature=0.75,  # Balanced between stability and variation
            preprocess_reference=True
        )
        
        print(f"\n🎉 Voice cloning completed successfully!")
        print(f"🎵 Output audio: {output_path}")
        print(f"\nTips for better results:")
        print("- Use clean, single-speaker reference audio (6-30 seconds)")
        print("- Ensure good audio quality in the reference")
        print("- Try different temperature values (0.5-1.0) for different styles")
        print("- Experiment with speed settings (0.8-1.2)")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check your file paths.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()