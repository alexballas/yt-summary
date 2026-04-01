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
from typing import List, Optional

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_BATCH_SIZE = 16
DEFAULT_OLLAMA_MODEL = "qwen2.5:latest"

device = "cuda"
compute_type = "float16"

model = None
model_name = None

def load_whisper_model(name: str = DEFAULT_MODEL):
    global model, model_name, compute_type
    model_name = name
    try:
        model = whisperx.load_model(name, device, compute_type=compute_type)
    except Exception:
        compute_type = "float32"
        model = whisperx.load_model(name, device, compute_type=compute_type)
    print(f"Loaded WhisperX model: {name} (compute_type={compute_type})")

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


def process_audio_file(audio_file, batch_size: int = DEFAULT_BATCH_SIZE):
    try:
        audio, samplerate = sf.read(audio_file)
        duration_seconds = len(audio) / samplerate
        print(f"Audio loaded: {format_duration(duration_seconds)} at {samplerate} Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

    print(f"Transcribing with batch_size={batch_size}...")
    start_time = time.time()
    
    audio_wave = whisperx.load_audio(audio_file)
    result = model.transcribe(audio_wave, batch_size=batch_size)
    
    elapsed = time.time() - start_time
    print(f"Transcription completed in {format_duration(elapsed)}")
    
    torch.cuda.empty_cache()

    print("Aligning transcription...")
    align_start = time.time()
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language", "en"), device=device
        )
        aligned_result = whisperx.align(
            result["segments"], align_model, metadata, audio_wave, device
        )
        segments = aligned_result.get("segments", result["segments"])
        del align_model
    except Exception as e:
        print(f"Alignment unavailable or failed, using raw segments: {e}")
        segments = result["segments"]
    
    torch.cuda.empty_cache()
    print(f"Alignment completed in {format_duration(time.time() - align_start)}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Running speaker diarization...")
        diarize_start = time.time()
        try:
            diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_speaker_segments = diarize_pipeline(audio_file)
            speaker_annotated = whisperx.assign_word_speakers(diarize_speaker_segments, {"segments": segments})
            segments = speaker_annotated.get("segments", segments)
            print(f"Diarization completed in {format_duration(time.time() - diarize_start)}")
        except Exception as e:
            print(f"Diarization unavailable or failed, continuing without it: {e}")
        
        torch.cuda.empty_cache()

    transcript_lines = []
    for seg in segments:
        txt = seg.get("text", "").strip()
        speaker = seg.get("speaker")
        if speaker:
            line = f"{speaker}: {txt}"
        else:
            line = f"{txt}"
        transcript_lines.append(line)

    full_transcript = "\n".join(transcript_lines)
    clean_text = normalize_text(full_transcript)

    print(f"\nTranscript length: {len(clean_text)} chars (~{estimate_tokens(clean_text)} tokens)")
    print("\n--- SUMMARIZING ---")
    summary = summarize_text_hierarchical(clean_text)
    return full_transcript, summary


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    else:
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours}h {mins}m {secs}s"


def summarize_with_ollama(text, prompt="Summarize this transcript in English. Preserve structure with clear section headings and bullet points for key takeaways. Be faithful to the content; avoid inventing details. Be concise", num_ctx=65536, ollama_model: str = DEFAULT_OLLAMA_MODEL):
    torch.cuda.empty_cache()
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": ollama_model,
        "prompt": f"{prompt}\n\n{text}",
        "stream": False,
        "options": {"num_ctx": int(num_ctx)},
        "keep_alive": 0
    }
    return _ollama_post_with_retry(url, payload)

def _ollama_post_with_retry(url: str, payload: dict, retries: int = 3, timeout: int = 300) -> str:
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
            wait_time = 2 ** attempt
            print(f"Ollama request failed (attempt {attempt}/{retries}), retrying in {wait_time}s...")
            time.sleep(wait_time)
    return f"Error: {last_err}"

_tiktoken_warned = False

def estimate_tokens(text: str) -> int:
    global _tiktoken_warned
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        if not _tiktoken_warned:
            print("Warning: tiktoken not installed. Using character-count fallback for token estimation.")
            _tiktoken_warned = True
        return max(1, int(len(text) / 4))

def split_text_token_aware(text: str, max_tokens: int, overlap_tokens: int = 400) -> List[str]:
    total_tokens = estimate_tokens(text)
    if total_tokens <= max_tokens:
        return [text]
    
    approx_chars_per_token = len(text) / total_tokens
    max_chars = max(1, int(max_tokens * approx_chars_per_token))
    overlap_chars = int(overlap_tokens * approx_chars_per_token)
    parts = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
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

def summarize_text_hierarchical(text: str, ollama_model: str = DEFAULT_OLLAMA_MODEL, max_iterations: int = 10) -> str:
    total_tokens = estimate_tokens(text)
    target_ctx = 65536 if total_tokens > 12000 else 32768
    max_chunk_tokens = int(target_ctx * 0.75)
    
    chunks = split_text_token_aware(text, max_tokens=max_chunk_tokens, overlap_tokens=400)
    
    if len(chunks) == 1:
        print(f"Single chunk ({estimate_tokens(chunks[0])} tokens), summarizing directly...")
        return summarize_with_ollama(chunks[0], num_ctx=target_ctx, ollama_model=ollama_model)
    
    print(f"Split into {len(chunks)} chunks for map-reduce summarization...")
    
    map_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Summarizing chunk {i}/{len(chunks)} ({estimate_tokens(chunk)} tokens)...")
        summary = summarize_with_ollama(
            chunk,
            prompt="Summarize this section faithfully. Keep speakers if present. Use concise bullets and short headings.",
            num_ctx=target_ctx,
            ollama_model=ollama_model,
        )
        map_summaries.append(summary)
    
    combined = "\n\n".join(map_summaries)
    combined_tokens = estimate_tokens(combined)
    
    print(f"Combined map summaries: {combined_tokens} tokens")
    
    iteration = 1
    while combined_tokens > target_ctx * 0.8 and iteration < max_iterations:
        print(f"Combined summaries exceed context ({combined_tokens} > {int(target_ctx * 0.8)}), reducing (iteration {iteration})...")
        
        reduce_chunks = split_text_token_aware(combined, max_tokens=max_chunk_tokens, overlap_tokens=200)
        
        if len(reduce_chunks) == 1:
            break
        
        reduce_summaries = []
        for i, chunk in enumerate(reduce_chunks, 1):
            print(f"  Reducing chunk {i}/{len(reduce_chunks)} ({estimate_tokens(chunk)} tokens)...")
            summary = summarize_with_ollama(
                chunk,
                prompt="Compress this summary further while keeping key points. Be concise.",
                num_ctx=target_ctx,
                ollama_model=ollama_model,
            )
            reduce_summaries.append(summary)
        
        combined = "\n\n".join(reduce_summaries)
        combined_tokens = estimate_tokens(combined)
        iteration += 1
    
    print(f"Final reduce phase ({combined_tokens} tokens)...")
    final_summary = summarize_with_ollama(
        combined,
        prompt="Combine these section summaries into a single coherent summary. Merge overlapping points, keep clear headings, and avoid duplication.",
        num_ctx=target_ctx,
        ollama_model=ollama_model,
    )
    
    return final_summary

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")

def remove_non_ascii(text):
    return normalize_text(text)

def _get_audio_duration_seconds(audio_path: str) -> Optional[int]:
    try:
        with sf.SoundFile(audio_path) as f:
            frames = len(f)
            sr = f.samplerate
            if sr > 0:
                return int(frames / sr)
    except Exception:
        return None
    return None

def check_model_updates(whisper_model: str = DEFAULT_MODEL, ollama_model: str = DEFAULT_OLLAMA_MODEL):
    print("\n=== Model Update Check ===\n")
    
    print(f"WhisperX model: {whisper_model}")
    print("  Note: Whisper models are versioned by name (e.g., large-v2, large-v3).")
    print("  Check https://huggingface.co/models?search=whisper for newer versions.")
    print(f"  Currently using: {whisper_model}")
    
    print(f"\nOllama model: {ollama_model}")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if ollama_model.split(":")[0] in line:
                    print(f"  Installed: {line}")
                    break
            else:
                print(f"  Model not found locally. Run: ollama pull {ollama_model}")
        else:
            print(f"  Could not check Ollama models: {result.stderr}")
    except FileNotFoundError:
        print("  Ollama not found. Install from: https://ollama.ai")
    except subprocess.TimeoutExpired:
        print("  Ollama check timed out")
    except Exception as e:
        print(f"  Error checking Ollama: {e}")
    
    print("\nTo update Ollama model: ollama pull " + ollama_model)
    print("To use a different Whisper model, run with: --model <model_name>")
    print("")

def load_transcript_from_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio or YouTube video with WhisperX and summarize with Ollama")
    parser.add_argument("--youtube", type=str, help="YouTube URL to download and process")
    parser.add_argument("--audio", type=str, help="Local audio file to process")
    parser.add_argument("--summarize-only", type=str, metavar="FILE", help="Summarize an existing transcript file (skip audio processing)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"WhisperX model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for transcription (default: {DEFAULT_BATCH_SIZE}, lower if OOM)")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model for summarization (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--check-updates", action="store_true", help="Check for model updates and exit")
    
    args = parser.parse_args()
    
    if args.check_updates:
        check_model_updates(args.model, args.ollama_model)
        exit(0)
    
    print(f"Device: {device}, Batch size: {args.batch_size}, Model: {args.model}")
    load_whisper_model(args.model)
    
    cleanup_temp_directory()
    
    if args.summarize_only:
        if not os.path.exists(args.summarize_only):
            print(f"Transcript file not found: {args.summarize_only}")
            exit(1)
        print(f"Loading transcript from: {args.summarize_only}")
        transcript = load_transcript_from_file(args.summarize_only)
        clean_text = normalize_text(transcript)
        print(f"Transcript length: {len(clean_text)} chars (~{estimate_tokens(clean_text)} tokens)")
        print("\n--- SUMMARIZING ---")
        summary = summarize_text_hierarchical(clean_text, ollama_model=args.ollama_model)
        print(f"\n--- SUMMARY ---\n{summary}")
        exit(0)
    
    audio_file = args.audio
    
    if args.youtube:
        try:
            audio_file = extract_audio_from_youtube(args.youtube)
            print(f"Downloaded audio from YouTube to: {audio_file}")
        except Exception as e:
            print(f"Failed to download from YouTube: {e}")
            if not audio_file or not os.path.exists(audio_file):
                print("No fallback audio file available. Exiting.")
                exit(1)
            print(f"Using fallback audio file: {audio_file}")
    
    if not audio_file:
        print("No audio file specified. Use --youtube or --audio.")
        exit(1)
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        exit(1)
    
    try:
        full, summary = process_audio_file(audio_file, batch_size=args.batch_size)
        if full:
            print(f"\n--- FULL TRANSCRIPT ---\n{full}")
        if summary:
            print(f"\n--- SUMMARY ---\n{summary}")
    finally:
        if os.path.exists(TEMP_DIR):
            print(f"Cleaning up temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)
