import whisperx
import torch

try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    
    def transcribe_with_vad_fixed(audio_path):
        # Step 1: Convert MP4 to WAV first (since MP4 causes issues)
        print("Converting MP4 to audio format...")
        
        # Use WhisperX's own audio loading (it handles MP4 internally)
        whisperx_audio = whisperx.load_audio(audio_path)
        
        # Save as temporary WAV for VAD processing
        import torchaudio
        temp_wav_path = audio_path.replace('.mp4', '_temp.wav')
        
        # Convert to tensor format for saving
        audio_tensor = torch.from_numpy(whisperx_audio).unsqueeze(0)  # Add channel dimension
        torchaudio.save(temp_wav_path, audio_tensor, 16000)
        
        print("Loading VAD model...")
        vad_model = load_silero_vad()
        
        print("Analyzing speech segments...")
        # Use the temporary WAV file
        wav = read_audio(temp_wav_path, sampling_rate=16000)
        
        # Get speech timestamps with only supported parameters
        speech_timestamps = get_speech_timestamps(
            wav, 
            vad_model, 
            sampling_rate=16000,
            threshold=0.4,  # Speech probability threshold
            min_speech_duration_ms=250,  # Minimum speech duration
            min_silence_duration_ms=300,  # Minimum silence duration
            window_size_samples=1024
            # Removed speech_pad_samples - not supported in your version
        )
        
        print(f"VAD found {len(speech_timestamps)} natural speech segments")
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        
        # Print VAD segments info
        for i, speech in enumerate(speech_timestamps):
            start_sec = speech['start'] / 16000
            end_sec = speech['end'] / 16000
            duration = end_sec - start_sec
            print(f"VAD Segment {i+1}: {start_sec:.2f}s - {end_sec:.2f}s ({duration:.2f}s)")
        
        # Step 2: Transcribe entire audio with full context
        print("\nTranscribing entire audio with full context...")
        model = whisperx.load_model("large", device="cpu", compute_type="int8")
        
        result_translated = model.transcribe(
            whisperx_audio,  # Use the original audio 
            task="translate",
            chunk_size=5,  # Small chunks but maintains context
            print_progress=True
        )
        
        print(f"Original transcription: {len(result_translated['segments'])} segments")
        
        return result_translated
    

    def seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds (float) to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def save_segments_to_srt(segments, output_path: str):
        """
        Save transcribed segments to an SRT file.

        Args:
            segments (list): list of dicts with "start", "end", "text"
            output_path (str): path to save the .srt file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start_time = seconds_to_srt_time(seg["start"])
                end_time = seconds_to_srt_time(seg["end"])
                text = seg["text"].strip()

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

        print(f"SRT file saved at: {output_path}")
    
    # Use the fixed method
    audio_path = "C:\\Users\\Administrator\\workspace\\heygenlite\\outputs\\Live_Bank_\\Live_Bank_Nifty_Option_Trading_____Intra.mp4"
    result_translated = transcribe_with_vad_fixed(audio_path)
    
    # print(f"\n=== FINAL RESULTS ===")
    # for i, segment in enumerate(result_translated["segments"]):
    #     duration = segment["end"] - segment["start"]
    #     print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] ({duration:.2f}s): {segment['text']}")
    import os
    srt_path = os.path.splitext(audio_path)[0] + ".srt"
    save_segments_to_srt(result_translated["segments"], srt_path)
except Exception as e:
    print(f"VAD method failed: {e}")
   