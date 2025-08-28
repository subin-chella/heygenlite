import yt_dlp
import os
import subprocess
import re

def sanitize_folder_name(title: str) -> str:
    # Replace spaces and special characters with underscores, limit to 10 chars
    sanitized = re.sub(r'[^A-Za-z0-9]', '_', title)
    return sanitized[:10]

def sanitize_file_name(title: str) -> str:
    # Replace spaces and special characters with underscores, limit to 40 chars
    sanitized = re.sub(r'[^A-Za-z0-9]', '_', title)
    return sanitized[:40]

def download_video(url):
    if not url:
        raise ValueError("YouTube URL cannot be empty.")

    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, get video info to determine folder and file name
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'video')
        folder_name = sanitize_folder_name(title)
        file_name = sanitize_file_name(title)
        subfolder = os.path.join(output_dir, folder_name)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # Set outtmpl to use sanitized file name in the subfolder
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(subfolder, f'{file_name}.%(ext)s'),
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

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=CtnDiBRrW4c"  # Replace with your URL
    video_path = download_video(url)
    print(f"Video downloaded to: {video_path}")
    audio_path = extract_audio(video_path)
    print(f"Audio extracted to: {audio_path}")



if __name__ == "__main__":
    audio_path = extract_audio("outputs\\Live_Bank_\\Live_Bank_Nifty_Option_Trading_____Intra.mp4")
    print(f"Audio extracted to: {audio_path}")