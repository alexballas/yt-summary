import requests
import torch
import os
import subprocess
import tempfile
import shutil
import soundfile as sf
import argparse
import whisperx
import time
import unicodedata
from typing import List, Tuple

device = "cuda"
batch_size = 32 # reduce if low on GPU mem

# Prefer float16, but fall back to float32 on older GPUs
compute_type = "float16"
try:
    model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type)
except Exception:
    compute_type = "float32"
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
        "--quiet",
        "--no-warnings",
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

    # Transcribe
    audio_wave = whisperx.load_audio(audio_file)
    result = model.transcribe(audio_wave, batch_size=batch_size)

    # Alignment to improve word boundaries and timestamps
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language", "en"), device=device
        )
        aligned_result = whisperx.align(
            result["segments"], align_model, metadata, audio_wave, device
        )
        segments = aligned_result.get("segments", result["segments"])
    except Exception as e:
        print(f"Alignment unavailable or failed, using raw segments: {e}")
        segments = result["segments"]

    # Optional diarization (only if configured). Skips silently if not available
    diarize_speaker_segments = None
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_speaker_segments = diarize_pipeline(audio_file)
            speaker_annotated = whisperx.assign_word_speakers(diarize_speaker_segments, {"segments": segments})
            segments = speaker_annotated.get("segments", segments)
        except Exception as e:
            print(f"Diarization unavailable or failed, continuing without it: {e}")

    # Build a structured transcript with timestamps and optional speakers
    transcript_lines = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        txt = seg.get("text", "").strip()
        speaker = seg.get("speaker")
        ts = f"[{format_timestamp(start)} - {format_timestamp(end)}]"
        if speaker:
            line = f"{ts} {speaker}: {txt}"
        else:
            line = f"{ts} {txt}"
        transcript_lines.append(line)

    full_transcript = "\n".join(transcript_lines)
    clean_text = normalize_text(full_transcript)

    print("\n--- SUMMARIZING ---")
    summary = summarize_text_hierarchical(clean_text)
    return full_transcript, summary


def summarize_with_ollama(text, prompt="Summarize this transcript for me in English. Preserve structure with clear section headings, timestamps where useful, and bullet points for key takeaways. Be faithful to the content; avoid inventing details.", num_ctx=65536):
    torch.cuda.empty_cache()
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:latest",
        "prompt": f"{prompt}\n\n{text}",
        "stream": False,
        "options": {"num_ctx": int(num_ctx)},
        "keep_alive": 0
    }
    return _ollama_post_with_retry(url, payload)

def _ollama_post_with_retry(url: str, payload: dict, retries: int = 3, timeout: int = 120) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                summary = resp.json().get("response", "Summarization failed.")
                return summary.replace("\\n", "\n")
            last_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(2 ** attempt)
    return f"Error: {last_err}"

def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 characters per token
    return max(1, int(len(text) / 4))

def split_text_token_aware(text: str, max_tokens: int, overlap_tokens: int = 200) -> List[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    approx_chars_per_token = max(1, int(len(text) / estimate_tokens(text)))
    max_chars = max_tokens * approx_chars_per_token
    overlap_chars = overlap_tokens * approx_chars_per_token
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to end at a newline or sentence boundary for quality
        candidate = text[start:end]
        last_nl = candidate.rfind("\n")
        last_dot = candidate.rfind(". ")
        cut = max(last_nl, last_dot)
        if cut > 0 and (end - start) > 2000:
            end = start + cut + 1
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return parts

def summarize_text_hierarchical(text: str) -> str:
    # Choose a conservative context window; adjust if very long
    target_ctx = 65536 if estimate_tokens(text) > 12000 else 32768
    chunks = split_text_token_aware(text, max_tokens=int(target_ctx * 0.8), overlap_tokens=400)
    if len(chunks) == 1:
        return summarize_with_ollama(chunks[0], num_ctx=target_ctx)

    # Map phase
    map_summaries = [
        summarize_with_ollama(
            c,
            prompt="Summarize this section faithfully. Keep timestamps and speakers if present. Use concise bullets and short headings.",
            num_ctx=target_ctx,
        )
        for c in chunks
    ]

    # Reduce phase
    combined = "\n\n".join(map_summaries)
    reduced = summarize_with_ollama(
        combined,
        prompt="Combine these section summaries into a single coherent summary. Merge overlapping points, keep important timestamps and headings, and avoid duplication.",
        num_ctx=target_ctx,
    )
    return reduced

def normalize_text(text: str) -> str:
    # Preserve Unicode while normalizing forms and whitespace
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")

def format_timestamp(seconds: float) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def remove_non_ascii(text):
    # Kept for backward compatibility; prefer normalize_text
    return normalize_text(text)

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
