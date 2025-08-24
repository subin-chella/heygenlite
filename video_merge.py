import os
import subprocess
from pathlib import Path
import json

class VideoAudioSubtitleMerger:
    """
    A comprehensive tool to merge MP4 video, MP3 audio, and SRT subtitles
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
        if not audio_path.lower().endswith('.mp3'):
            errors.append("Audio file must be MP3 format")
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
    
    def merge_video_audio_subtitles(self, video_path, audio_path, output_path, 
                                   subtitle_path=None, options=None):
        """
        Merge video, audio, and optionally subtitles
        
        Args:
            video_path: Path to MP4 video file
            audio_path: Path to MP3 audio file
            output_path: Path for output file
            subtitle_path: Optional path to SRT subtitle file
            options: Dictionary of additional options
        """
        
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
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"stderr: {e.stderr}")
            return False
    
    def batch_merge(self, input_dir, output_dir=None):
        """
        Batch merge files in a directory
        Looks for matching MP4, MP3, and SRT files with same base name
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
        
        for mp4_file in mp4_files:
            base_name = mp4_file.stem
            
            # Look for matching files
            mp3_file = input_path / f"{base_name}.mp3"
            srt_file = input_path / f"{base_name}.srt"
            
            if not mp3_file.exists():
                print(f"⚠️ No matching MP3 found for {mp4_file.name}")
                failed += 1
                continue
            
            output_file = output_path / f"{base_name}_merged.mp4"
            
            print(f"\n--- Processing {base_name} ---")
            
            try:
                success = self.merge_video_audio_subtitles(
                    str(mp4_file),
                    str(mp3_file),
                    str(output_file),
                    str(srt_file) if srt_file.exists() else None
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ Error processing {base_name}: {e}")
                failed += 1
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    # =================================================================
    # CONFIGURATION - Modify these variables for your needs
    # =================================================================
    
    # Mode selection: 'single' for one file, 'batch' for directory processing
    MODE = 'single'  # Options: 'single' or 'batch'
    
    # SINGLE FILE MODE SETTINGS
    VIDEO_PATH = "outputs\\8_Nov___Tr\\8_Nov___Trade_Analysis_BankNifty_Option_.mp4"           # Path to your MP4 video file
    AUDIO_PATH = "outputs\\8_Nov___Tr\\8_Nov___Trade_Analysis_BankNifty_Option__translated.mp3"       # Path to your MP3 audio file
    SUBTITLE_PATH = "outputs\\8_Nov___Tr\\8_Nov___Trade_Analysis_BankNifty_Option_.srt"          # Path to your SRT file (set to None if no subtitles)
    OUTPUT_PATH = "outputs\\8_Nov___Tr\\8_Nov___Trade_Analysis_BankNifty_Option__final_output.mp4"         # Output file path

    # PROCESSING OPTIONS
    OPTIONS = {
        'replace_audio': True,           # True = replace original audio, False = mix both
        'video_codec': 'copy',           # 'copy' = no re-encoding, 'libx264' = re-encode
        'audio_codec': 'aac',            # Audio codec: 'aac', 'mp3', 'libmp3lame'
        'audio_bitrate': '128k',         # Audio bitrate: '128k', '192k', '256k', etc.
        'subtitle_codec': 'mov_text',    # Subtitle codec for MP4
        'crf': 23,                       # Video quality if re-encoding (lower = better)
        'overwrite': True                # Overwrite existing output files
    }
    
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
        print(f"Output: {OUTPUT_PATH}")
        print()
        
        success = merger.merge_video_audio_subtitles(
            video_path=VIDEO_PATH,
            audio_path=AUDIO_PATH,
            output_path=OUTPUT_PATH,
            subtitle_path=SUBTITLE_PATH if SUBTITLE_PATH else None,
            options=OPTIONS
        )
        
        if success:
            print("\n🎉 Single file merge completed successfully!")
        else:
            print("\n❌ Single file merge failed!")
    
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your file paths and settings.")