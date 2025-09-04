import subprocess
import os

def extract_speech_demucs(input_audio, output_dir=None):
    """
    Uses Demucs to separate vocals (speech) from music/background.
    Input: audio file (wav, mp3, etc)
    Output: saves vocals.wav in output_dir
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_audio)
    os.makedirs(output_dir, exist_ok=True)

    # If input is not wav or mp3, convert to wav first
    ext = os.path.splitext(input_audio)[1].lower()
    if ext not in [".wav", ".mp3"]:
        print(f"Converting {input_audio} to wav for Demucs...")
        converted_audio = os.path.join(output_dir, os.path.splitext(os.path.basename(input_audio))[0] + "_demucs.wav")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_audio,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", converted_audio
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        input_for_demucs = converted_audio
    else:
        input_for_demucs = input_audio

    # Demucs will create a 'demucs' folder in output_dir
    command = [
        "python", "-m", "demucs", input_for_demucs, "--two-stems=vocals", "--out", output_dir
    ]
    subprocess.run(command, check=True)
    # Find the vocals file
    demucs_dir = os.path.join(output_dir, "demucs")
    # Demucs output folder is demucs/<basename_without_ext>/vocals.wav
    demucs_subdir = os.path.splitext(os.path.basename(input_for_demucs))[0]
    vocals_path = os.path.join(demucs_dir, demucs_subdir, "vocals.wav")
    if os.path.exists(vocals_path):
        print(f"Speech (vocals) extracted: {vocals_path}")
        return vocals_path
    else:
        print("Failed to extract vocals.")
        return None


if __name__ == "__main__":
    input_audio = "outputs\\Live_Bank_\\Live_Bank_Nifty_Option_Trading_____Intra.mp4"
    output_dir = "outputs"
    extract_speech_demucs(input_audio, output_dir)