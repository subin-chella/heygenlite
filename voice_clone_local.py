import os
import warnings
import re
import gc
import pysrt
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import numpy as np
import soundfile as sf
import tempfile
import noisereduce as nr
from pathlib import Path

# Suppress specific warnings
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*GenerationMixin.*")

def preprocess_reference_audio(reference_wav):
    """Preprocess reference audio to improve cloning quality"""
    try:
        audio = AudioSegment.from_wav(reference_wav)

        # Normalize volume
        audio = normalize(audio)

        # Reduce noise
        samples = np.array(audio.get_array_of_samples())
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=0.75)

        # Convert back to AudioSegment
        processed_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        # Create temporary file for processed reference
        temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        processed_audio.export(temp_ref.name, format="wav")

        return temp_ref.name
    except Exception as e:
        print(f"⚠️ Could not preprocess reference audio: {e}")
        return reference_wav

def diagnose_srt(srt_path):
    """Diagnose SRT file content with detailed analysis"""
    print(f"🔍 Analyzing SRT file: {srt_path}")
    try:
        subs = pysrt.open(srt_path)
        print(f"📊 Total subtitles found: {len(subs)}")
        if not subs:
            print("❌ No subtitles found in file!")
            return False

        # Check for overlaps
        overlaps = []
        for i in range(len(subs) - 1):
            if subs[i].end.ordinal > subs[i + 1].start.ordinal:
                overlaps.append((i, i + 1, subs[i].end.ordinal - subs[i + 1].start.ordinal))

        if overlaps:
            print(f"⚠️ Found {len(overlaps)} timing overlaps in subtitles:")
            for idx1, idx2, overlap_ms in overlaps:
                print(f"   Overlap between subtitles {idx1 + 1} and {idx2 + 1}: {overlap_ms}ms")
                print(f"     Subtitle {idx1 + 1}: {subs[idx1].start} -> {subs[idx1].end}")
                print(f"     Subtitle {idx2 + 1}: {subs[idx2].start} -> {subs[idx2].end}")
        else:
            print("✅ No timing overlaps detected")

        # Show all subtitles for debugging
        for i, sub in enumerate(subs):
            print(f"\n--- Subtitle {i + 1} ---")
            print(f"Index: {sub.index}")
            print(f"Start: {sub.start} ({sub.start.ordinal}ms)")
            print(f"End: {sub.end} ({sub.end.ordinal}ms)")
            print(f"Text: '{sub.text}'")
            print(f"Text length: {len(sub.text.strip())} chars")
            print(f"Empty text: {not sub.text.strip()}")

        return True
    except Exception as e:
        print(f"❌ Error reading SRT: {e}")
        return False

def srt_to_voice_enhanced(srt_path, reference_wav, language="en", use_gpu=False):
    """Enhanced version with better voice cloning quality"""
    # Generate output path from SRT path
    srt_path_obj = Path(srt_path)
    output_path = str(srt_path_obj.parent / f"{srt_path_obj.stem}_voice.wav")

    print(f"📁 Output will be saved to: {output_path}")
    # Validate input files
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    if not os.path.exists(reference_wav):
        raise FileNotFoundError(f"Reference WAV file not found: {reference_wav}")

    # Preprocess reference audio
    print("🎵 Preprocessing reference audio...")
    processed_ref = preprocess_reference_audio(reference_wav)

    # Diagnose SRT first
    if not diagnose_srt(srt_path):
        return

    print("\n🎤 Loading TTS model...")
    try:
        # Try to use a different model if available
        try:
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v1.1", gpu=use_gpu)
            print("✅ Using XTTS v1.1 model")
        except:
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
            print("✅ Using XTTS v2 model")
    except Exception as e:
        print(f"❌ Failed to load TTS model: {e}")
        raise

    print("\n📝 Parsing SRT file for processing...")
    subs = pysrt.open(srt_path)

    # Calculate total duration in milliseconds
    total_duration_ms = max(sub.end.ordinal for sub in subs) if subs else 0
    print(f"⏱️ Total duration: {total_duration_ms/1000:.2f} seconds")

    # Get reference WAV properties
    try:
        ref_audio = AudioSegment.from_wav(reference_wav)
        ref_frame_rate = ref_audio.frame_rate
        ref_channels = ref_audio.channels
    except Exception as e:
        print(f"⚠️ Could not read reference WAV properties, using defaults (24000 Hz, 1 channel): {e}")
        ref_frame_rate = 24000
        ref_channels = 1

    # Create a silent audio track with reference WAV properties
    full_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=ref_frame_rate)
    full_audio = full_audio.set_channels(ref_channels)
    print(f"🔇 Created silent audio track of {total_duration_ms}ms, {full_audio.frame_rate}Hz, {full_audio.channels} channels")

    # Temporary directory for generated WAV files
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    # Text preprocessing for better TTS
    def preprocess_text(text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Add short pause for commas
        text = text.replace(',', ',<break time="200ms"/>')
        # Add longer pause for periods
        text = text.replace('.', '.<break time="300ms"/>')
        return text

    try:
        for i, sub in enumerate(subs, 1):
            text = sub.text.strip()
            text = preprocess_text(text)

            if not text:
                print(f"⏭️ Skipping subtitle {i}/{len(subs)}: empty text")
                skipped_count += 1
                continue

            print(f"\n🎯 Processing subtitle {i}/{len(subs)}:")
            print(f"   Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
            print(f"   Time: {sub.start.ordinal/1000:.2f}s - {sub.end.ordinal/1000:.2f}s ({(sub.end.ordinal - sub.start.ordinal)/1000:.2f}s duration)")

            temp_wav = os.path.join(temp_dir, f"sub_{sub.index}.wav")

            try:
                print(f"   🔊 Generating TTS...")

                # Try different TTS parameters for better quality
                tts.tts_to_file(
                    text=text,
                    file_path=temp_wav,
                    speaker_wav=processed_ref,
                    language=language,
                    # Experiment with these parameters
                    speed=1.0,  # Adjust if speech is too fast/slow
                    emotion="Neutral",  # Try different emotions
                    temperature=0.7,  # Lower for more stable output
                )

                if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                    print(f"   ❌ TTS file not created or empty for subtitle {i}")
                    skipped_count += 1
                    continue

                speech = AudioSegment.from_wav(temp_wav)

                if len(speech) == 0:
                    print(f"   ❌ Generated audio for subtitle {i} has zero duration")
                    skipped_count += 1
                    if os.path.exists(temp_wav): os.remove(temp_wav)
                    continue

                print(f"   📊 Generated audio: {len(speech)}ms, {speech.frame_rate}Hz, {speech.channels} channels")

                speech = speech.set_frame_rate(full_audio.frame_rate).set_channels(full_audio.channels)

                # Apply audio enhancements
                speech = normalize(speech)
                speech = compress_dynamic_range(speech, threshold=-20.0, ratio=4.0)

                # Adjust to fit duration
                duration_ms = sub.end.ordinal - sub.start.ordinal
                if duration_ms <= 0:
                    print(f"   ⚠️ Skipping subtitle {i}: Invalid duration ({duration_ms}ms)")
                    skipped_count += 1
                    if os.path.exists(temp_wav): os.remove(temp_wav)
                    continue

                generated_duration = len(speech)

                if generated_duration > duration_ms:
                    speed_ratio = generated_duration / duration_ms
                    print(f"   ⏩ Speeding up audio by {speed_ratio:.2f}x to fit duration.")
                    speech = speech.speedup(playback_speed=speed_ratio, chunk_size=150, crossfade=25)
                elif generated_duration < duration_ms:
                    pad_ms = duration_ms - generated_duration
                    print(f"   ⏸️ Padding with {pad_ms}ms silence.")
                    speech += AudioSegment.silent(duration=pad_ms, frame_rate=speech.frame_rate)

                # Apply a slight fade in/out to avoid clicks
                speech = speech.fade_in(50).fade_out(50)

                full_audio = full_audio.overlay(speech, position=sub.start.ordinal)
                print(f"   ✅ Added to timeline at {sub.start.ordinal}ms")

                if os.path.exists(temp_wav): os.remove(temp_wav)
                processed_count += 1
                gc.collect()

            except Exception as e:
                print(f"   ❌ Error processing subtitle {i}: {e}")
                skipped_count += 1
                if os.path.exists(temp_wav): os.remove(temp_wav)
                continue

        print(f"\n📈 Processing complete:")
        print(f"   ✅ Processed: {processed_count} subtitles")
        print(f"   ⏭️ Skipped: {skipped_count} subtitles")

        if processed_count == 0:
            print("❌ No subtitles were processed successfully, output may be empty")
            return

        print(f"\n💾 Exporting final audio to {output_path}...")
        # Apply final processing to the complete audio
        final_audio = normalize(full_audio, headroom=0.1)
        final_audio = compress_dynamic_range(final_audio, threshold=-20.0, ratio=4.0)
        final_audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "24000"])

        print(f"✅ Output saved to {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        # Clean up processed reference if it was created
        if processed_ref != reference_wav and os.path.exists(processed_ref):
            os.remove(processed_ref)

        if os.path.exists(temp_dir):
            try:
                for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
                print("🧹 Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️ Could not remove temp directory: {e}")

if __name__ == "__main__":
    # Ensure you update these paths to your actual file locations
    reference_wav = "/Users/subin/workspace/heygenlite/Live_Bank_Nifty_Option_Trading_____Intra_segments/Live_Bank_Nifty_Option_Trading_____Intra_part2.wav"
    srt_path = "/Users/subin/workspace/heygenlite/outputs/Live_Bank_/Live_Bank_Nifty_Option_Trading_____Intra.srt"

    print("=" * 60)
    srt_to_voice_enhanced(
        srt_path=srt_path,
        reference_wav=reference_wav,
        language="en",
        use_gpu=False
    )
