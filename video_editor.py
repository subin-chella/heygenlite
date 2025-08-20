from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
import os

def combine_video_and_audio(video_path, audio_path, srt_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        video_clip = video_clip.set_audio(audio_clip)

        subtitles = TextClip.from_srt(srt_path, fontsize=24, color='white')
        subtitles = subtitles.set_pos(('center', 'bottom')).set_duration(video_clip.duration)

        final_clip = CompositeVideoClip([video_clip, subtitles])

        final_video_path = f"{os.path.splitext(video_path)[0]}_translated.mp4"
        final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

        return final_video_path

    except Exception as e:
        raise RuntimeError(f"Failed to combine video and audio: {e}")
