import warnings
from pydub import AudioSegment, silence

# Suppress SyntaxWarnings from pydub regex patterns
warnings.filterwarnings("ignore", category=SyntaxWarning)


def remove_silence(input_file, output_file,
                   min_silence_len=700,
                   silence_thresh=-40,
                   padding=200):
    """
    Removes long silences and keeps only speech/audio parts.

    Args:
        input_file (str): Path to input audio file.
        output_file (str): Path to save processed audio file.
        min_silence_len (int): Minimum silence length (ms) to detect as silence.
        silence_thresh (int): Silence threshold (dBFS). Lower = more sensitive.
        padding (int): Padding (ms) to add before and after non-silent chunks.
    """
    print(f"Loading audio from: {input_file}")
    audio = AudioSegment.from_file(input_file)

    # Detect non-silent chunks
    non_silent_chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=audio.dBFS + silence_thresh,
        keep_silence=padding
    )

    if not non_silent_chunks:
        print("⚠️ No speech detected! Nothing to save.")
        return

    # Combine chunks into one audio file
    processed_audio = AudioSegment.empty()
    for i, chunk in enumerate(non_silent_chunks, 1):
        processed_audio += chunk
        print(f"Added chunk {i} ({len(chunk)} ms)")

    # Export processed file
    processed_audio.export(output_file, format="wav")
    print(f"✅ Processed audio saved as: {output_file}")


# Example usage
if __name__ == "__main__":
    input_path = r"outputs\Live_Bank_\Live_Bank_Nifty_Option_Trading_____Intra.wav"
    output_path = r"outputs\Live_Bank_\Live_Bank_Nifty_Option_Trading_____Intra_sample.wav"

    remove_silence(input_path, output_path,
                   min_silence_len=2000,   # adjust for shorter/longer pauses
                   silence_thresh=-40,    # raise/lower sensitivity
                   padding=200)           # keep some silence for natural breaks
