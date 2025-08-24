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

def generate_tts_for_srt(srt_path: str, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2"):
    """
    Generate TTS audio for SRT subtitle file using the latest ElevenLabs API.
    The output audio file will be saved in the same folder as the SRT file, named 'combined_audio.mp3'.
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

        try:
            audio_iterator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_44100_128"
            )
            audio_bytes = b"".join(audio_iterator)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            audio_data, sr = sf.read(temp_file_path)
            os.unlink(temp_file_path)

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

    total_duration_ms = max(seg['end_ms'] for seg in audio_segments)
    total_samples = int((total_duration_ms / 1000.0) * sample_rate)
    final_audio = np.zeros(total_samples, dtype=np.float32)

    for segment in audio_segments:
        start_sample = int((segment['start_ms'] / 1000.0) * sample_rate)
        audio_data = segment['audio_data']
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        end_sample = start_sample + len(audio_data)
        if end_sample > len(final_audio):
            audio_data = audio_data[:len(final_audio) - start_sample]
            end_sample = len(final_audio)
        final_audio[start_sample:end_sample] = audio_data

    # Determine output path based on SRT file path
    output_dir = os.path.dirname(srt_path)
    filename = os.path.splitext(os.path.basename(srt_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_translated.mp3")

    sf.write(output_path, final_audio, sample_rate, format='MP3')
    print(f"Successfully generated combined audio file at: {output_path}")
    print(f"Total duration: {len(final_audio) / sample_rate:.2f} seconds")
    return output_path

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
        # Run the function with the dummy file
        input_srt_path = "outputs\\8_Nov___Tr\\8_Nov___Trade_Analysis_BankNifty_Option_.srt"
        
        # You can change the voice_id and model_id here
        # Popular voice IDs:
        # "JBFqnCBsd6RMkjVDRZzb" - George (British male)
        # "21m00Tcm4TlvDq8ikWAM" - Rachel (American female)
        # "AZnzlk1XvdvUeBnXmlld" - Domi (American female)
        # "EXAVITQu4vr4xnSDxMaL" - Bella (American female)
        
        output_path = generate_tts_for_srt(
            srt_path=input_srt_path, 
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # George - British male
            model_id="eleven_multilingual_v2"  # or "eleven_flash_v2_5" for faster/cheaper
        )

    except FileNotFoundError:
        print("Please create a 'subtitle.srt' file in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")