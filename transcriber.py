import os
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(OPENAI_API_KEY)  # Set your API key as an environment variable

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
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_path), audio_file, "audio/wav"),
            }
            data = {
                "model": "whisper-1",
                "language": "hi",
                "response_format": "verbose_json",
                "task": "translate"
            }
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()

        hindi_transcript = result.get("text", "")
        english_translation = result.get("text", "")
        english_path = f"{os.path.splitext(audio_path)[0]}_english.txt"

        srt_path = f"{os.path.splitext(audio_path)[0]}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result.get("segments", [])):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                f.write(f"{i+1}\n")
                f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                f.write(f"{text.strip()}\n\n")
         # Store English translation in a file
        with open(english_path, "w", encoding="utf-8") as ef:
            ef.write(english_translation.strip())

        return hindi_transcript, english_translation, srt_path

    except Exception as e:
        print(f"An error occurred: {e}")
        raise RuntimeError(f"Whisper API transcription failed: {e}")

if __name__ == "__main__":
    audio_path = "outputs/Canva ｜ Dil Se Design Tak ｜ Jaadu Dadu ｜ Tamil.wav"
    hindi_transcript, english_translation, srt_path = transcribe_and_translate(audio_path)