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
    print(f"subtitle blocks: {blocks}")
    return blocks

def srt_time_to_ms(time_str):
    """Convert SRT time format to milliseconds"""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

def generate_tts_for_srt(srt_path: str, output_path: str, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2"):
    """
    Generate TTS audio for SRT subtitle file using the latest ElevenLabs API.
    
    Args:
        srt_path: Path to the SRT file
        output_path: Path for the output audio file
        voice_id: ElevenLabs voice ID (default: George - a British male voice)
        model_id: Model to use (options: eleven_multilingual_v2, eleven_flash_v2_5, eleven_turbo_v2_5)
    """
    blocks = parse_srt_blocks(srt_path)
    if not blocks:
        raise ValueError("No subtitle blocks found in the SRT file.")

    print(f"Processing {len(blocks)} subtitle blocks...")

    # Collect all audio segments with timing info
    audio_segments = []
    sample_rate = 44100  # Standard sample rate for mp3_44100_128

    for idx, start, end, text in blocks:
        if not text.strip():
            continue

        print(f"Processing block {idx}: {text[:50]}...")

        start_time_ms = srt_time_to_ms(start)
        end_time_ms = srt_time_to_ms(end)

        # Generate TTS for the block using the new API
        try:
            # Use the new text_to_speech.convert method
            audio_iterator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_44100_128"
            )

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

def list_available_voices():
    """
    List available voices from your ElevenLabs account.
    """
    try:
        response = client.voices.search()
        print("Available voices:")
        for voice in response.voices:
            print(f"- {voice.name}: {voice.voice_id}")
        return response.voices
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    try:
        # Uncomment the next line to see available voices
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
        
        # Run the function with the dummy file
        input_srt_path = "subtitle.srt"
        output_audio_path = os.path.join(OUTPUT_DIR, "combined_audio.mp3")
        
        # You can change the voice_id and model_id here
        # Popular voice IDs:
        # "JBFqnCBsd6RMkjVDRZzb" - George (British male)
        # "21m00Tcm4TlvDq8ikWAM" - Rachel (American female)
        # "AZnzlk1XvdvUeBnXmlld" - Domi (American female)
        # "EXAVITQu4vr4xnSDxMaL" - Bella (American female)
        
        generate_tts_for_srt(
            srt_path=input_srt_path, 
            output_path=output_audio_path,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # George - British male
            model_id="eleven_multilingual_v2"  # or "eleven_flash_v2_5" for faster/cheaper
        )

    except FileNotFoundError:
        print("Please create a 'subtitle.srt' file in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")