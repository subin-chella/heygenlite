import streamlit as st
import os
from downloader import download_video
from transcriber import transcribe_and_translate
from tts_generator import generate_tts
from video_editor import combine_video_and_audio
from utils import get_timestamp

st.title("Simplified Heygen-Style Translator")

if 'processing' not in st.session_state:
    st.session_state.processing = False

youtube_url = st.text_input("Enter YouTube URL:")

if st.button("Process Video") and not st.session_state.processing:
    if youtube_url:
        st.session_state.processing = True
        try:
            st.info("Downloading video...")
            video_path = download_video(youtube_url)

            st.info("Transcribing and translating...")
            hindi_transcript, english_translation, srt_path = transcribe_and_translate(video_path)

            st.info("Generating audio...")
            audio_path = generate_tts(english_translation)

            st.info("Merging video and audio...")
            final_video_path = combine_video_and_audio(video_path, audio_path, srt_path)

            st.success("Video processed successfully!")
            st.video(final_video_path)

            with open(final_video_path, "rb") as f:
                st.download_button("Download Video", f, file_name=os.path.basename(final_video_path))

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            st.session_state.processing = False
    else:
        st.error("Please enter a valid YouTube URL.")
