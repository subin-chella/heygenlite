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
                response_format="verbose_json"
            )
        print(f"API response: {response}")

        srt_path = f"{os.path.splitext(audio_path)[0]}.srt"
        segments = getattr(response, "segments", [])
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                # Use attribute access for TranscriptionSegment, fallback to dict
                start = getattr(segment, 'start', None) if hasattr(segment, 'start') else segment.get('start', 0)
                end = getattr(segment, 'end', None) if hasattr(segment, 'end') else segment.get('end', 0)
                text = getattr(segment, 'text', '') if hasattr(segment, 'text') else segment.get('text', '')
                # Ensure start and end are floats and not None
                start = float(start) if start is not None else 0.0
                end = float(end) if end is not None else 0.0
                f.write(f"{i+1}\n")
                f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
                f.write(f"{text.strip()}\n\n")

        return srt_path

    except Exception as e:
        print(f"An error occurred: {e}")
        raise RuntimeError(f"Whisper API transcription failed: {e}")

if __name__ == "__main__":
    audio_path = "outputs\\Live_Bank_\\Live_Bank_Nifty_Option_Trading_____Intra.wav"
    transcribe_and_translate(audio_path)