import yt_dlp
import os

def download_video(url):
    if not url:
        raise ValueError("YouTube URL cannot be empty.")

    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'bestvideo[height=1080][fps=25]+bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        raise RuntimeError(f"Failed to download video: {e}")
