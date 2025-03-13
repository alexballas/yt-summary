import requests
import torch
import os
import io
import subprocess
from PIL import Image
import soundfile as sf
import numpy as np
import argparse
import shutil
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen

def extract_audio_from_youtube(youtube_url, output_dir="./temp_audio"):
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

model_path = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    attn_implementation='flash_attention_2',
).cuda()

generation_config = GenerationConfig.from_pretrained(model_path)

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

def process_audio_file(audio_file):
    print(f"\n--- PROCESSING AUDIO FILE: {audio_file} ---")
    
    try:
        audio, samplerate = sf.read(audio_file)
        print(f"Audio loaded: {len(audio)/samplerate:.2f} seconds at {samplerate} Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    transcriptions = process_audio_in_chunks(audio, samplerate)
    full_transcript = " ".join(transcriptions)
    print("\n--- FULL TRANSCRIPT ---")
    print(full_transcript)
   
    clean_text = remove_non_ascii(full_transcript) 
    
    print("\n--- SUMMARIZING ---")
    summary = summarize_with_ollama(clean_text)
   
    print(summary)

    
    return full_transcript, summary

def process_audio_in_chunks(audio, samplerate, chunk_seconds=20):
    chunk_size = chunk_seconds * samplerate
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    transcriptions = []
    
    print(f"Processing audio in {num_chunks} chunks of {chunk_seconds} seconds each")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        
        chunk_prompt = f'{user_prompt}<|audio_1|>Transcribe this audio chunk to text.{prompt_suffix}{assistant_prompt}'
        
        inputs = processor(text=chunk_prompt, audios=[(chunk, samplerate)], return_tensors='pt').to('cuda:0')
        
        torch.cuda.empty_cache()
        
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        transcriptions.append(response.strip())
        print(f"Chunk {i+1} transcription: {response[:50]}...")
        
        torch.cuda.empty_cache()
    
    return transcriptions

def summarize_with_ollama(text):
    globals().pop("model", None)
    torch.cuda.empty_cache()
    
    text_length = len(text)
    ctx_options = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    num_ctx = min(next((ctx for ctx in ctx_options if text_length <= ctx), 131072), 131072)

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:latest",
        "prompt": f"Summarize this transcript for me, but keep all the section headings and key titles intact:\n\n{text}",
        "stream": False,
        "options": {
            "num_ctx": num_ctx
        },
        "keep_alive": 0
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        summary = response.json().get("response", "Summarization failed.")
        return summary.replace("\\n", "\n")
    else:
        return f"Error: {response.status_code}, {response.text}"

def remove_non_ascii(text):
    return text.encode("ascii", "ignore").decode("ascii")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio or YouTube video with Phi-4-multimodal")
    parser.add_argument("--youtube", type=str, help="YouTube URL to download and process")
    parser.add_argument("--audio", type=str, help="Local audio file to process (default: a.opus)")
    
    args = parser.parse_args()
    
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
    
    process_audio_file(audio_file)
    shutil.rmtree("./temp_audio")
