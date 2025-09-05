import os
import subprocess
from pathlib import Path
import json

class VideoAudioSubtitleMerger:
    """
    A comprehensive tool to merge MP4 video, WAV audio, and SRT subtitles
    """

    def __init__(self):
        self.check_ffmpeg()

    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'],
                         capture_output=True, check=True)
            print("✅ FFmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ FFmpeg not found. Please install FFmpeg:")
            print("Windows: Download from https://ffmpeg.org/download.html")
            print("Mac: brew install ffmpeg")
            print("Linux: sudo apt install ffmpeg")
            raise SystemExit("FFmpeg is required for this program")

    def get_media_info(self, file_path):
        """Get media information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error getting info for {file_path}: {e}")
            return None

    def get_duration(self, file_path):
        """Get duration of media file in seconds"""
        info = self.get_media_info(file_path)
        if info and 'format' in info:
            return float(info['format']['duration'])
        return None

    def validate_files(self, video_path, audio_path, subtitle_path=None):
        """Validate input files"""
        errors = []

        # Check if files exist
        if not os.path.exists(video_path):
            errors.append(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            errors.append(f"Audio file not found: {audio_path}")
        if subtitle_path and not os.path.exists(subtitle_path):
            errors.append(f"Subtitle file not found: {subtitle_path}")

        if errors:
            return False, errors

        # Check file extensions
        if not video_path.lower().endswith('.mp4'):
            errors.append("Video file must be MP4 format")
        if not audio_path.lower().endswith(('.wav', '.mp3')):
            errors.append("Audio file must be WAV or MP3 format")
        if subtitle_path and not subtitle_path.lower().endswith('.srt'):
            errors.append("Subtitle file must be SRT format")

        # Check durations
        video_duration = self.get_duration(video_path)
        audio_duration = self.get_duration(audio_path)

        if video_duration and audio_duration:
            duration_diff = abs(video_duration - audio_duration)
            if duration_diff > 5:  # More than 5 seconds difference
                print(f"⚠️ Warning: Duration mismatch - Video: {video_duration:.1f}s, Audio: {audio_duration:.1f}s")

        return len(errors) == 0, errors

    def generate_output_path(self, video_path, suffix="_merged"):
        """Generate output path based on video path"""
        video_path_obj = Path(video_path)
        output_filename = f"{video_path_obj.stem}{suffix}{video_path_obj.suffix}"
        return str(video_path_obj.parent / output_filename)

    def merge_video_audio_subtitles(self, video_path, audio_path,
                                   subtitle_path=None, options=None):
        """
        Merge video, audio, and optionally subtitles

        Args:
            video_path: Path to MP4 video file
            audio_path: Path to WAV audio file
            subtitle_path: Optional path to SRT subtitle file
            options: Dictionary of additional options
        """

        # Generate output path
        output_path = self.generate_output_path(video_path)

        # Set default options
        default_options = {
            'replace_audio': True,      # Replace original audio or mix
            'audio_codec': 'aac',       # Output audio codec
            'video_codec': 'copy',      # Copy video without re-encoding
            'subtitle_codec': 'mov_text', # Subtitle codec for MP4
            'crf': 23,                  # Video quality (if re-encoding)
            'audio_bitrate': '128k',    # Audio bitrate
            'overwrite': True           # Overwrite output file if exists
        }

        if options:
            default_options.update(options)
        options = default_options

        # Validate files
        valid, errors = self.validate_files(video_path, audio_path, subtitle_path)
        if not valid:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        # Build FFmpeg command
        cmd = ['ffmpeg']

        if options['overwrite']:
            cmd.append('-y')

        # Input files
        cmd.extend(['-i', video_path])
        cmd.extend(['-i', audio_path])

        if subtitle_path:
            cmd.extend(['-i', subtitle_path])

        # Video codec
        cmd.extend(['-c:v', options['video_codec']])

        # Audio handling
        if options['replace_audio']:
            cmd.extend(['-c:a', options['audio_codec']])
            cmd.extend(['-b:a', options['audio_bitrate']])
            cmd.extend(['-map', '0:v:0'])  # Video from first input
            cmd.extend(['-map', '1:a:0'])  # Audio from second input
        else:
            # Mix audio (more complex, requires filter)
            cmd.extend(['-filter_complex',
                       '[0:a][1:a]amix=inputs=2:duration=longest[aout]'])
            cmd.extend(['-map', '0:v:0'])
            cmd.extend(['-map', '[aout]'])
            cmd.extend(['-c:a', options['audio_codec']])
            cmd.extend(['-b:a', options['audio_bitrate']])

        # Subtitle handling
        if subtitle_path:
            if options['replace_audio']:
                cmd.extend(['-map', '2:s:0'])  # Subtitles from third input
            else:
                cmd.extend(['-map', '2:s:0'])
            cmd.extend(['-c:s', options['subtitle_codec']])

        # Video quality (if re-encoding)
        if options['video_codec'] != 'copy':
            cmd.extend(['-crf', str(options['crf'])])

        # Output file
        cmd.append(output_path)

        print(f"Merging files...")
        print(f"Video: {video_path}")
        print(f"Audio: {audio_path}")
        if subtitle_path:
            print(f"Subtitles: {subtitle_path}")
        print(f"Output: {output_path}")
        print(f"Command: {' '.join(cmd)}")

        try:
            # Run FFmpeg command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print("✅ Merge completed successfully!")
            print(f"Output saved to: {output_path}")

            # Show output file info
            output_info = self.get_media_info(output_path)
            if output_info:
                duration = float(output_info['format']['duration'])
                size_mb = int(output_info['format']['size']) / (1024 * 1024)
                print(f"Duration: {duration:.1f} seconds")
                print(f"File size: {size_mb:.1f} MB")

            return output_path

        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"stderr: {e.stderr}")
            return None

    def batch_merge(self, input_dir, output_dir=None):
        """
        Batch merge files in a directory
        Looks for matching MP4, WAV/MP3, and SRT files with same base name
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        if output_dir is None:
            output_dir = input_path / "merged"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Find MP4 files
        mp4_files = list(input_path.glob("*.mp4"))

        if not mp4_files:
            print("No MP4 files found in input directory")
            return

        successful = 0
        failed = 0
        output_files = []

        for mp4_file in mp4_files:
            base_name = mp4_file.stem

            # Look for matching files (try WAV first, then MP3)
            wav_file = input_path / f"{base_name}.wav"
            mp3_file = input_path / f"{base_name}.mp3"
            srt_file = input_path / f"{base_name}.srt"

            # Prefer WAV over MP3
            audio_file = None
            if wav_file.exists():
                audio_file = wav_file
            elif mp3_file.exists():
                audio_file = mp3_file
            else:
                print(f"⚠️ No matching audio file found for {mp4_file.name}")
                failed += 1
                continue

            print(f"\n--- Processing {base_name} ---")

            try:
                result = self.merge_video_audio_subtitles(
                    str(mp4_file),
                    str(audio_file),
                    str(srt_file) if srt_file.exists() else None
                )

                if result:
                    successful += 1
                    output_files.append(result)
                else:
                    failed += 1

            except Exception as e:
                print(f"❌ Error processing {base_name}: {e}")
                failed += 1

        print(f"\n=== Batch Processing Complete ===")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        return output_files

if __name__ == "__main__":
    # =================================================================
    # CONFIGURATION - Modify these variables for your needs
    # =================================================================

    # SINGLE FILE MODE SETTINGS
    VIDEO_PATH = "/Users/subin/workspace/heygenlite/outputs/Live_Bank_/Live_Bank_Nifty_Option_Trading_____Intra.mp4"           # Path to your MP4 video file
    AUDIO_PATH = "/Users/subin/workspace/heygenlite/output_audio_enhanced.wav"       # Path to your WAV audio file
    SUBTITLE_PATH = "/Users/subin/workspace/heygenlite/outputs/Live_Bank_/Live_Bank_Nifty_Option_Trading_____Intra copy.srt"          # Path to your SRT file (set to None if no subtitles)

    # PROCESSING OPTIONS
    # options = {
    #     'replace_audio': True,
    #     'video_codec': 'copy',
    #     'audio_codec': 'aac',
    #     'audio_bitrate': '192k'
    # }

    # =================================================================
    # EXECUTION - No need to modify below this line
    # =================================================================

    print("=== Video + Audio + Subtitle Merger ===")
    try:
        merger = VideoAudioSubtitleMerger()

        print("Single file processing...")
        print(f"Video: {VIDEO_PATH}")
        print(f"Audio: {AUDIO_PATH}")
        if SUBTITLE_PATH:
            print(f"Subtitles: {SUBTITLE_PATH}")
        print()

        output_path = merger.merge_video_audio_subtitles(
            video_path=VIDEO_PATH,
            audio_path=AUDIO_PATH,
            subtitle_path=SUBTITLE_PATH if SUBTITLE_PATH else None,
            # options=options  # Uncomment to use custom options
        )

        if output_path:
            print(f"\n🎉 Single file merge completed successfully!")
            print(f"Output file: {output_path}")
        else:
            print("\n❌ Single file merge failed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your file paths and settings.")
