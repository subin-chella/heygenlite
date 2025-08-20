from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def generate_tts(text):
    if not text:
        raise ValueError("Text for TTS cannot be empty.")

    try:
        audio = client.generate(
            text=text,
            voice="Adam",
            model="eleven_multilingual_v2"
        )
        audio_path = "outputs/translated_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio)
        return audio_path
    except Exception as e:
        raise RuntimeError(f"ElevenLabs TTS generation failed: {e}")