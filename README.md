# Heygen-Lite: Simplified Video Translation

This project is a simplified version of a Heygen-style video translation tool. It takes a YouTube URL, downloads the video, transcribes the audio, translates it to English, generates new audio, and creates a new video with the translated audio and subtitles.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/heygen-lite.git
    cd heygen-lite
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root of the project and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    ELEVENLABS_API_KEY="your_elevenlabs_api_key"
    ```

## How to Run

1.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

3.  Open your browser and go to `http://localhost:8501`.
