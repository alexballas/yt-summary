import requests
import torch
import os
import subprocess
import tempfile
import shutil
import soundfile as sf
import argparse
import whisperx

device = "cuda"
batch_size = 32 # reduce if low on GPU mem
compute_type = "float16"
model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type)


TEMP_DIR = os.path.join(tempfile.gettempdir(), "audio_processing")

def cleanup_temp_directory():
    if os.path.exists(TEMP_DIR):
        print(f"Cleaning up temporary directory: {TEMP_DIR}")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")

    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Created temporary directory: {TEMP_DIR}")

def extract_audio_from_youtube(youtube_url, output_dir=TEMP_DIR):
    """Extract audio from YouTube URL and save to temporary directory"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "youtube_audio.%(ext)s")

    print(f"Extracting audio from YouTube URL: {youtube_url}")
    command = [
        "yt-dlp", 
        "-x", 
        "--audio-format", "opus", 
        "--audio-quality", "0", 
        "-o", output_file, 
        youtube_url  
    ]
    
    try:
        subprocess.run(command, check=True)
        for file in os.listdir(output_dir):
            if file.startswith("youtube_audio"):
                return os.path.join(output_dir, file)
        raise FileNotFoundError("Extracted audio file not found")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def process_audio_file(audio_file):
    try:
        audio, samplerate = sf.read(audio_file)
        print(f"Audio loaded: {len(audio)/samplerate:.2f} seconds at {samplerate} Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    texts = [segment["text"] for segment in result["segments"]]
    full_transcript = " ".join(texts)

    clean_text = remove_non_ascii(full_transcript)
    
    print("\n--- SUMMARIZING ---")

    if len(clean_text) > 100000:
        parts = [clean_text[i:i+100000] for i in range(0, len(clean_text), 100000)]
        summaries = [summarize_with_ollama(part) for part in parts]
        combined = " ".join(summaries)
        summary = summarize_with_ollama(combined, prompt="Combine the summaries into one:")
    else:
        summary = summarize_with_ollama(clean_text)
    
    return full_transcript, summary


def summarize_with_ollama(text, prompt="Summarize this transcript for me, but keep all the section headings and key titles intact:"):
    torch.cuda.empty_cache()
    
    text_length = len(text)
    ctx_options = [8192, 16384, 32768, 65536, 131072]
    print(text_length)

    num_ctx = min(next((ctx for ctx in ctx_options if text_length+4000 <= ctx), 131072), 131072)

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:latest",
        "prompt": f"{prompt}\n\n{text}",
        "stream": False,
        "options": {
            "num_ctx": num_ctx
        },
        "keep_alive": 0
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        summary = response.json().get("response", "Summarization failed.")
        final = summary.replace("\\n", "\n")
        return final
    else:
        return f"Error: {response.status_code}, {response.text}"

def remove_non_ascii(text):
    return text.encode("ascii", "ignore").decode("ascii")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio or YouTube video with Phi-4-multimodal")
    parser.add_argument("--youtube", type=str, help="YouTube URL to download and process")
    parser.add_argument("--audio", type=str, help="Local audio file to process (default: a.opus)")
    
    args = parser.parse_args()
    
    cleanup_temp_directory()

    audio_file = args.audio
    
    if args.youtube:
        try:
            audio_file = extract_audio_from_youtube(args.youtube)
            print(f"Downloaded audio from YouTube to: {audio_file}")
        except Exception as e:
            print(f"Failed to download from YouTube: {e}")
            if not os.path.exists(audio_file):
                print(f"Fallback audio file {audio_file} doesn't exist either. Exiting.")
                exit(1)
            print(f"Using fallback audio file: {audio_file}")
    
    try:
        full, sum = process_audio_file(audio_file)
        print(f"\n--- FULL TRANSCRIPT ---\n{full}")
        print(f"\n--- SUMMARY ---\n{sum}")
    finally:
        if os.path.exists(TEMP_DIR):
            print(f"Cleaning up temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)
