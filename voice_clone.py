import os
import re
import io
import time
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import soundfile as sf
import numpy as np

# Load API key from .env
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")

# Debug: Check if API key is loaded
if not api_key:
    print("Warning: ELEVENLABS_API_KEY not found in environment variables.")
    print("Please check your .env file or set the API key directly in the code.")
    # Uncomment the line below and add your API key directly if needed:
    # api_key = "sk-your_actual_elevenlabs_api_key_here"
else:
    print(f"API Key loaded: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")

if not api_key:
    raise ValueError("No API key provided. Please set ELEVENLABS_API_KEY in your .env file.")

client = ElevenLabs(api_key=api_key)

# Ensure output directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_srt_blocks(srt_path: str):
    """
    Parse SRT into a list of subtitle blocks.
    Returns a list of tuples: (index, start_time, end_time, text)
    """
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

def srt_time_to_ms(time_str):
    """Convert SRT time format to milliseconds"""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

def clone_voice_from_reference(reference_audio_path: str, voice_name: str, description: str = ""):
    """
    Clone a voice from reference audio (can be in any language).
    This maintains the tone and characteristics from the reference audio.
    
    Args:
        reference_audio_path: Path to reference audio file
        voice_name: Name for the cloned voice
        description: Optional description of the voice
    
    Returns:
        voice_id: ID of the cloned voice
    """
    try:
        print(f"Creating voice clone from reference audio: {reference_audio_path}")
        
        # Create voice clone using Instant Voice Cloning
        voice = client.voices.ivc.create(
            name=voice_name,
            description=description or f"Voice cloned from {reference_audio_path}",
            files=[reference_audio_path],
        )
        
        print(f"✅ Voice cloned successfully! Voice ID: {voice.voice_id}")
        return voice.voice_id
        
    except Exception as e:
        print(f"❌ Error cloning voice: {e}")
        return None

def create_voice_with_settings(voice_id: str, stability: float = 0.5, similarity_boost: float = 0.75, 
                              style: float = 0.0, use_speaker_boost: bool = True):
    """
    Create voice settings to control tone and delivery.
    
    Args:
        voice_id: Voice ID to use
        stability: Controls consistency (0.0-1.0). Lower = more expressive, Higher = more stable
        similarity_boost: Controls similarity to original voice (0.0-1.0)
        style: Controls style exaggeration (0.0-1.0). Higher = more expressive
        use_speaker_boost: Whether to use speaker boost for better similarity
    
    Returns:
        Voice settings dict
    """
    return {
        "voice_id": voice_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        }
    }

def generate_tts_for_srt_with_tone_control(
    srt_path: str, 
    output_path: str, 
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    reference_audio_path: str = None,
    voice_settings: dict = None,
    target_language: str = None
):
    """
    Generate TTS audio for SRT subtitle file with advanced tone control.
    
    Args:
        srt_path: Path to the SRT file
        output_path: Path for the output audio file
        voice_id: ElevenLabs voice ID to use
        model_id: Model to use
        reference_audio_path: Optional reference audio for voice cloning
        voice_settings: Optional voice settings for tone control
        target_language: Target language for output (if different from input)
    """
    
    # If reference audio is provided, clone the voice first
    cloned_voice_id = None
    if reference_audio_path and os.path.exists(reference_audio_path):
        voice_name = f"cloned_voice_{int(time.time())}"
        cloned_voice_id = clone_voice_from_reference(
            reference_audio_path, 
            voice_name,
            f"Voice cloned for maintaining tone from {os.path.basename(reference_audio_path)}"
        )
        if cloned_voice_id:
            voice_id = cloned_voice_id
            print(f"Using cloned voice: {voice_id}")
    
    # Set default voice settings if not provided
    if not voice_settings:
        voice_settings = {
            "stability": 0.5,      # Balanced stability
            "similarity_boost": 0.75,  # Good similarity to original
            "style": 0.2,          # Slight style enhancement
            "use_speaker_boost": True
        }
    
    blocks = parse_srt_blocks(srt_path)
    if not blocks:
        raise ValueError("No subtitle blocks found in the SRT file.")

    print(f"Processing {len(blocks)} subtitle blocks with tone control...")
    print(f"Voice settings: {voice_settings}")

    # Collect all audio segments with timing info
    audio_segments = []
    sample_rate = 44100

    for idx, start, end, text in blocks:
        if not text.strip():
            continue

        print(f"Processing block {idx}: {text[:50]}...")

        start_time_ms = srt_time_to_ms(start)
        end_time_ms = srt_time_to_ms(end)

        # Generate TTS for the block with voice settings
        try:
            # Prepare the API call parameters
            tts_params = {
                "text": text,
                "voice_id": voice_id,
                "model_id": model_id,
                "output_format": "mp3_44100_128",
                "voice_settings": voice_settings
            }
            
            # If target language is specified and different, add language hint
            if target_language:
                # Note: ElevenLabs automatically detects language, but you can hint
                # by including language-specific text formatting or phonetic guides
                pass
            
            audio_iterator = client.text_to_speech.convert(**tts_params)

            # Collect all bytes from the iterator
            audio_bytes = b"".join(audio_iterator)

            # Save to temporary file and read with soundfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # Read the audio data
            audio_data, sr = sf.read(temp_file_path)
            
            # Clean up temp file
            os.unlink(temp_file_path)

            # Store segment info
            audio_segments.append({
                'start_ms': start_time_ms,
                'end_ms': end_time_ms,
                'audio_data': audio_data,
                'sample_rate': sr
            })

        except Exception as e:
            print(f"Error generating audio for block {idx}: {e}")
            continue

    if not audio_segments:
        raise ValueError("No audio segments were successfully generated.")

    # Calculate total duration needed
    total_duration_ms = max(seg['end_ms'] for seg in audio_segments)
    total_samples = int((total_duration_ms / 1000.0) * sample_rate)

    # Create the final audio array
    final_audio = np.zeros(total_samples, dtype=np.float32)

    # Place each audio segment at its correct position
    for segment in audio_segments:
        start_sample = int((segment['start_ms'] / 1000.0) * sample_rate)
        audio_data = segment['audio_data']
        
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        end_sample = start_sample + len(audio_data)
        
        # Make sure we don't exceed the final audio length
        if end_sample > len(final_audio):
            audio_data = audio_data[:len(final_audio) - start_sample]
            end_sample = len(final_audio)
        
        final_audio[start_sample:end_sample] = audio_data

    # Save the final combined audio
    sf.write(output_path, final_audio, sample_rate, format='MP3')
    print(f"Successfully generated combined audio file at: {output_path}")
    print(f"Total duration: {len(final_audio) / sample_rate:.2f} seconds")
    
    # Clean up cloned voice if it was created
    if cloned_voice_id and cloned_voice_id != "JBFqnCBsd6RMkjVDRZzb":
        try:
            # Uncomment if you want to delete the cloned voice after use
            # client.voices.delete(cloned_voice_id)
            # print(f"Cleaned up cloned voice: {cloned_voice_id}")
            print(f"Cloned voice {cloned_voice_id} kept for future use.")
        except:
            pass

def test_api_connection():
    """Test if the API key is working"""
    try:
        response = client.voices.search()
        print(f"✅ API connection successful! Found {len(response.voices)} voices.")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def list_available_voices():
    """List available voices from your ElevenLabs account."""
    try:
        response = client.voices.search()
        print("Available voices:")
        for voice in response.voices:
            print(f"- {voice.name}: {voice.voice_id}")
        return response.voices
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return []

def create_emotional_presets():
    """Predefined voice settings for different emotional tones"""
    return {
        "neutral": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True},
        "excited": {"stability": 0.3, "similarity_boost": 0.8, "style": 0.7, "use_speaker_boost": True},
        "calm": {"stability": 0.8, "similarity_boost": 0.75, "style": 0.1, "use_speaker_boost": True},
        "dramatic": {"stability": 0.2, "similarity_boost": 0.85, "style": 0.9, "use_speaker_boost": True},
        "professional": {"stability": 0.7, "similarity_boost": 0.8, "style": 0.2, "use_speaker_boost": True},
        "conversational": {"stability": 0.4, "similarity_boost": 0.7, "style": 0.4, "use_speaker_boost": True}
    }

if __name__ == "__main__":
    # Test API connection first
    print("Testing API connection...")
    if not test_api_connection():
        print("Exiting due to API connection failure.")
        exit(1)
    
    # Example usage with tone control
    try:
        # Uncomment to see available voices
        # list_available_voices()
        
        # Create a dummy SRT file for demonstration
        dummy_srt_content = """1
00:00:01,000 --> 00:00:03,500
Hello there! This is a test.

2
00:00:04,000 --> 00:00:06,800
This is the second line of the subtitle.

3
00:00:07,500 --> 00:00:10,200
And here is the final part.
"""
        with open("subtitle.srt", "w", encoding="utf-8") as f:
            f.write(dummy_srt_content)
        
        # Get emotional presets
        presets = create_emotional_presets()
        
        # Example 1: Using reference audio for tone matching
        # reference_audio = "path/to/your/reference_audio.mp3"  # Uncomment and set path
        reference_audio = None  # Set to None if you don't have reference audio
        
        # Example 2: Choose an emotional preset
        chosen_tone = "conversational"  # Options: neutral, excited, calm, dramatic, professional, conversational
        
        input_srt_path = "subtitle.srt"
        output_audio_path = os.path.join(OUTPUT_DIR, f"combined_audio_{chosen_tone}.mp3")
        
        print(f"Generating audio with {chosen_tone} tone...")
        
        generate_tts_for_srt_with_tone_control(
            srt_path=input_srt_path,
            output_path=output_audio_path,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # George - British male
            model_id="eleven_multilingual_v2",
            reference_audio_path=reference_audio,  # Optional: path to reference audio
            voice_settings=presets[chosen_tone],   # Use emotional preset
            target_language=None  # Optional: specify if different from input
        )

    except FileNotFoundError:
        print("Please create a 'subtitle.srt' file in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")