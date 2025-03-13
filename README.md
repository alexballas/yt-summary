# YouTube Audio Transcription and Summarization

This script allows you to download audio from a YouTube video, transcribe it using a multimodal model, and summarize the transcript. The transcription is done in chunks to handle 
long audio files efficiently.

## Features

- Downloads audio from a YouTube video.
- Transcribes the audio into text using a multimodal model.
- Summarizes the transcribed text while keeping section headings and key titles intact.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/alexballas/yt-summary
   cd yt-summary
   ```

2. Install the required dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Ensure you have a CUDA-compatible GPU and the necessary drivers installed for running the model on a GPU.

4. Install `yt-dlp` if it is not already installed:
```bash
pip install yt-dlp
```

## Usage

You can run the script from the command line with the following options:

```bash
python3 yt.py --youtube <YOUTUBE_URL>
```

### Arguments

- `--youtube`: URL of the YouTube video to download and process.
- `--audio`: Local audio file to process.

## Example

To transcribe and summarize audio from a YouTube video:

```bash
source venv/bin/activate
python yt.py --youtube https://www.youtube.com/watch?v=example_video_id
```

## Notes

- Ensure that the summarization API (`http://localhost:11434/api/generate`) is running on your local machine.
- The script uses the `qwen2.5:latest` model for summarization. Make sure this model is available and correctly configured.

## License

This project is licensed under the MIT License.
