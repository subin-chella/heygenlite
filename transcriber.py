import whisper
import os

def transcribe_and_translate(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        model = whisper.load_model("large-v3")
        result = model.transcribe(video_path, language="hi", task="translate")

        hindi_transcript = result["text"]
        english_translation = result["translation"]

        srt_path = f"{os.path.splitext(video_path)[0]}.srt"
        with open(srt_path, "w") as f:
            for i, segment in enumerate(result["segments"]):
                start_time = int(segment['start'])
                end_time = int(segment['end'])
                text = segment['text']
                f.write(f"{i+1}\n")
                f.write(f"{start_time // 3600:02d}:{start_time // 60 % 60:02d}:{start_time % 60:02d},000 --> {end_time // 3600:02d}:{end_time // 60 % 60:02d}:{end_time % 60:02d},000\n")
                f.write(f"{text}\n\n")

        return hindi_transcript, english_translation, srt_path

    except Exception as e:
        raise RuntimeError(f"Whisper transcription/translation failed: {e}")
