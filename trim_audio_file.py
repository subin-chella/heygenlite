from pydub import AudioSegment
import math
import os

def split_wav(file_path, segment_length=4*60*1000):  # 4 minutes in ms
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Calculate number of segments
    num_segments = math.ceil(len(audio) / segment_length)
    
    # Output folder
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{base_name}_segments"
    os.makedirs(output_dir, exist_ok=True)
    print(base_name)
    # Split and export segments
    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(audio))
        segment = audio[start:end]
        
        segment.export(
            os.path.join(output_dir, f"{base_name}_part{i+1}.wav"),
            format="wav"
        )
        print(f"Exported segment {i+1}/{num_segments}")
    
    print("All segments exported successfully.")

if __name__ == "__main__":
    split_wav("outputs\\Live_Bank_\\Live_Bank_Nifty_Option_Trading_____Intra.wav")
