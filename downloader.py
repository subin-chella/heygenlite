import yt_dlp
import os
import subprocess

def download_video(url):
    if not url:
        raise ValueError("YouTube URL cannot be empty.")

    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        raise RuntimeError(f"Failed to download video: {e}")

def extract_audio(video_path):
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(command, check=True)
    return audio_path