import os
import subprocess
import re
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

# =========================
# Environment / API Setup
# =========================
print("[INIT] Loading environment variables (.env)")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("[ERROR] OPENAI_API_KEY environment variable not set")
print("[INIT] OPENAI_API_KEY loaded (masked) length:", len(OPENAI_API_KEY))

openai.api_key = OPENAI_API_KEY

# =========================
# Utility
# =========================
def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def get_audio_duration(path: str) -> float:
    """Return duration (seconds) of an audio file using ffprobe."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
        ], text=True).strip()
        return float(out)
    except Exception as e:
        print(f"[DURATION][WARN] Could not get duration for {path}: {e}")
        return 0.0
# =========================
# Transcription
# =========================
def transcribe_and_translate(audio_path: str, time_offset: float = 0.0, translate_to_english: bool = True) -> List[Dict[str, Any]]:
    """
    Transcribe a single audio chunk and return adjusted segments.
    Falls back to a single segment if API returns no per‑segment data.
    """
    print(f"[TRANSCRIBE] Start file='{audio_path}' offset={time_offset:.3f}")
    if not os.path.exists(audio_path):
        print(f"[TRANSCRIBE][ERROR] Missing file: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        with open(audio_path, "rb") as audio_file:
            size = os.path.getsize(audio_path)
            print(f"[TRANSCRIBE] -> Whisper (size={size} bytes) translate_to_english={translate_to_english}")
            if translate_to_english:
                response = openai.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                print("[TRANSCRIBE] Used translations endpoint (forced English)")
            else:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                print("[TRANSCRIBE] Used transcriptions endpoint (original language)")

        # Save the raw response to a file
        response_json_path = os.path.splitext(audio_path)[0] + ".json"
        import json
        with open(response_json_path, "w", encoding="utf-8") as f:
            json.dump(response.to_dict(), f)
        print(f"[TRANSCRIBE] Saved raw response to {response_json_path}")

        # Extract segments (object attributes) or fallback to word list
        segments = getattr(response, "segments", None)
        if segments is None:
            segments = getattr(response, "words", [])
        raw_text = getattr(response, "text", "")
        print(f"[TRANSCRIBE] API segments={len(segments)} raw_text_len={len(raw_text)}")

        adjusted_segments: List[Dict[str, Any]] = []
        for idx, segment in enumerate(segments):
            # segment may be pydantic-like object; fallback to dict-style if needed
            start = float(getattr(segment, 'start', getattr(segment, 'from', 0.0))) if hasattr(segment, 'start') or hasattr(segment, 'from') else float(getattr(segment, 'start', 0.0) if isinstance(segment, dict) else 0.0)
            end = float(getattr(segment, 'end', getattr(segment, 'to', 0.0))) if hasattr(segment, 'end') or hasattr(segment, 'to') else float(getattr(segment, 'end', 0.0) if isinstance(segment, dict) else 0.0)
            txt = getattr(segment, 'text', segment.get('text', '') if isinstance(segment, dict) else '').strip()
            duration = end - start
            print(f"[TRANSCRIBE][SEG] #{idx} ({start:.3f}->{end:.3f} dur={duration:.3f}) text_len={len(txt)}")
            if txt and end > start:
                adjusted_segments.append({
                    'start': start + time_offset,
                    'end': end + time_offset,
                    'text': txt,
                    'type': 'speech'
                })

        if not adjusted_segments and raw_text.strip():
            chunk_dur = get_audio_duration(audio_path)
            if chunk_dur <= 0:
                chunk_dur = 0.5
            print(f"[TRANSCRIBE][FALLBACK] Creating single segment for chunk duration={chunk_dur:.3f}")
            adjusted_segments.append({
                'start': time_offset,
                'end': time_offset + chunk_dur,
                'text': raw_text.strip(),
                'type': 'speech'
            })

        print(f"[TRANSCRIBE] Final adjusted segment count={len(adjusted_segments)}")
        return adjusted_segments
    except Exception as e:
        print(f"[TRANSCRIBE][ERROR] Whisper failure file='{audio_path}' error={e}")
        raise RuntimeError(f"Whisper API transcription failed: {e}")

# =========================
# Silence Based Splitting
# =========================
def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def split_audio_on_silence(input_audio: str, silence_db: int = -30, silence_duration: float = 0.5):
    """
    Split audio based on silence, but KEEP continuous ranges that include the silence.
    Example:
      If silence detected at 100-101 and 300-301 in a 600s file,
      result chunks will be:
        0 → 101
        101 → 301
        301 → 600
    No seconds are dropped.
    """
    print(f"[INPUT] {input_audio}")
    if not os.path.exists(input_audio):
        raise FileNotFoundError(input_audio)

    input_dir = os.path.dirname(input_audio)
    chunk_dir = os.path.join(input_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # --- Detect silence with ffmpeg ---
    detect_cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", input_audio,
        "-af", f"silencedetect=noise={silence_db}dB:d={silence_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(detect_cmd, capture_output=True, text=True)
    stderr = result.stderr

    silence_ranges = []
    current_start = None
    for line in stderr.splitlines():
        if "silence_start" in line:
            current_start = float(re.search(r"silence_start: ([\d.]+)", line).group(1))
        elif "silence_end" in line and current_start is not None:
            end = float(re.search(r"silence_end: ([\d.]+)", line).group(1))
            silence_ranges.append((current_start, end))
            current_start = None

    # --- Get total duration ---
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_audio
    ]
    total_duration = float(subprocess.check_output(duration_cmd, text=True).strip())

    # --- Build non-silence chunks ---
    chunks = []
    prev = 0.0
    for s_start, s_end in silence_ranges:
        if s_start > prev:
            chunks.append((prev, s_end))  # extend through silence
        prev = s_end
    if prev < total_duration:
        chunks.append((prev, total_duration))

    print(f"[CHUNKS] {len(chunks)} chunks created")

    # --- Extract chunks with ffmpeg ---
    chunk_files = []
    for idx, (start, end) in enumerate(chunks):
        out_file = os.path.join(chunk_dir, f"chunk_{idx:03d}.wav")
        print(f"[FFMPEG] chunk {idx}: {start:.2f}s → {end:.2f}s")

        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", input_audio,
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-c", "copy", out_file, "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        chunk_files.append(out_file)

    # Save metadata
    with open(os.path.join(chunk_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return chunk_files, chunks
import os
import re
from typing import List, Dict, Any

import os
import re
from typing import List, Dict, Any

def regenerate_srt_from_json(input_audio: str) -> None:
    """
    Regenerate SRT file from JSON files containing transcription segments.
    Uses duration from each JSON file and accumulates time offsets.
    """
    print(f"[REGENERATE] Input audio: {input_audio}")
    input_dir = os.path.dirname(input_audio)
    chunk_dir = os.path.join(input_dir, "chunks")

    if not os.path.exists(chunk_dir):
        print(f"[REGENERATE][ERROR] Chunk directory not found: {chunk_dir}")
        return

    # Find and sort JSON files
    json_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".json")])
    print(f"[REGENERATE][DEBUG] Found {len(json_files)} JSON files: {json_files}")
    if not json_files:
        print(f"[REGENERATE][ERROR] No JSON files found in {chunk_dir}")
        return

    all_segments: List[Dict[str, Any]] = []
    time_offset = 0.0  # Cumulative time offset

    for json_file in json_files:
        print(f"[REGENERATE][DEBUG] Processing JSON file: {json_file}")
        json_path = os.path.join(chunk_dir, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                response_str = f.read()
                print(f"[REGENERATE][DEBUG] JSON content: {response_str[:100]}...")  # truncate preview

                # Extract duration
                duration_match = re.search(r"duration=([\d.]+)", response_str)
                chunk_duration = float(duration_match.group(1)) if duration_match else 7.03
                print(f"[REGENERATE][DEBUG] Chunk duration: {chunk_duration}s")

                # Extract segments
                segments_match = re.search(r"segments=\[(.*?)\]", response_str, re.DOTALL)
                parsed_segments = []

                if segments_match and segments_match.group(1).strip():
                    segments_str = segments_match.group(1)
                    print(f"[REGENERATE][DEBUG] Segments block found")

                    # Match each TranscriptionSegment(...) block
                    segment_pattern = re.compile(
                        r"TranscriptionSegment\((.*?)\)", re.DOTALL
                    )
                    for sm in segment_pattern.finditer(segments_str):
                        seg_str = sm.group(1)

                        # extract fields in any order
                        start_match = re.search(r"start=([\d.]+)", seg_str)
                        end_match = re.search(r"end=([\d.]+)", seg_str)
                        text_match = re.search(r"text=['\"](.*?)['\"]", seg_str, re.DOTALL)

                        if start_match and end_match:
                            start = float(start_match.group(1))
                            end = float(end_match.group(1))
                            text = text_match.group(1).strip() if text_match else ""
                            if end > start:
                                parsed_segments.append({'start': start, 'end': end, 'text': text})
                                print(f"[REGENERATE][DEBUG] Parsed segment: start={start}, end={end}, text='{text}'")

                # fallback: if no parsed segments, use top-level text
                if not parsed_segments:
                    print(f"[REGENERATE][DEBUG] No valid segments parsed, checking text field")
                    text_match = re.search(r"text=['\"](.*?)['\"]", response_str, re.DOTALL)
                    text = text_match.group(1).strip() if text_match else ""
                    parsed_segments.append({'start': 0.0, 'end': chunk_duration, 'text': text})
                    print(f"[REGENERATE][DEBUG] Fallback text: '{text}'")

                # add with offset
                for seg in parsed_segments:
                    adjusted = {
                        'start': seg['start'] + time_offset,
                        'end': seg['end'] + time_offset,
                        'text': seg['text']
                    }
                    all_segments.append(adjusted)
                    print(f"[REGENERATE][DEBUG] Adjusted segment: {adjusted}")

                # increment offset
                time_offset += chunk_duration
                print(f"[REGENERATE][DEBUG] Updated time_offset: {time_offset}s")

        except Exception as e:
            print(f"[REGENERATE][ERROR] Failed to process {json_file}: {e}")
            continue

    # If no segments at all
    if not all_segments:
        all_segments.append({'start': 0.0, 'end': time_offset, 'text': ''})

    print(f"[REGENERATE][DEBUG] Total segments to write: {len(all_segments)}")

    # Write SRT file
    srt_path = os.path.join(input_dir, os.path.splitext(os.path.basename(input_audio))[0] + "_regenerated.srt")
    print(f"[SRT] Writing SRT to: {srt_path}")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(all_segments, start=1):
            start = seg['start']
            end = max(seg['end'], start + 0.05)  # ensure non-zero length
            text = seg['text'] if seg['text'].strip() else " "
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")
            if i % 25 == 0:
                print(f"[SRT] Wrote {i} segments...")

    print(f"[DONE] Regenerated SRT saved: {srt_path}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"




if __name__ == "__main__":
    INPUT_AUDIO = "outputs/Live_Bank_/Live_Bank_Nifty_Option_Trading_____Intra.wav"
    SILENCE_DB = -30        # Silence threshold in dB
    SILENCE_DURATION = 0.25  # Minimum silence duration in seconds
    MAX_DURATION_SECONDS = 60 # Process only the first 60 seconds
    
    split_audio_on_silence(INPUT_AUDIO, SILENCE_DB, SILENCE_DURATION)
    regenerate_srt_from_json(INPUT_AUDIO)