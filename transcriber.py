import os
import requests
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def transcribe_and_translate(audio_path: str):
    print(f"Audio path: {audio_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        with open(audio_path, "rb") as audio_file:
            response = openai.audio.translations.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt"  # Request subtitle format
            )
        print(f"API response: {response}")
        srt_path = f"{os.path.splitext(audio_path)[0]}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(response)

        return srt_path

    except Exception as e:
        print(f"An error occurred: {e}")
        raise RuntimeError(f"Whisper API transcription failed: {e}")

if __name__ == "__main__":
    audio_path = "outputs/Canva ｜ Dil Se Design Tak ｜ Jaadu Dadu ｜ Tamil.wav"
    transcribe_and_translate(audio_path)