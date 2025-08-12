#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
import uuid
import whisper
import srt
import datetime
from tqdm import tqdm
import asyncio
import edge_tts

# Install transformers for NLLB
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "sentencepiece", "sacremoses"])
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Pydub for combining audio segments
try:
    from pydub import AudioSegment
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
    from pydub import AudioSegment


def extract_audio_if_video(input_path, tmp_dir):
    video_exts = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(input_path.lower())
    if ext in video_exts:
        out_audio = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", out_audio]
        print("Extracting audio with ffmpeg...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_audio
    else:
        return input_path


def segments_to_srt(segments):
    subs = []
    for i, seg in enumerate(segments, start=1):
        start_td = datetime.timedelta(seconds=float(seg['start']))
        end_td = datetime.timedelta(seconds=float(seg['end']))
        subs.append(srt.Subtitle(index=i, start=start_td, end=end_td, content=seg['text'].strip()))
    return srt.compose(subs)


# Language code map for NLLB — MUST BE DEFINED BEFORE USAGE
lang_code_map = {
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "es": "spa_Latn",  # Spanish
    "fr": "fra_Latn",  # French
    "de": "deu_Latn",  # German
    "zh": "zho_Hans",  # Chinese Simplified
    "ar": "ara_Arab",  # Arabic
    # Add more languages here if needed
}

# Load NLLB model for translation once
print("Loading NLLB model for translation...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def get_translator_pipeline(src_lang="en", tgt_lang="hi"):
    src_code = lang_code_map.get(src_lang, "eng_Latn")
    tgt_code = lang_code_map.get(tgt_lang, "hin_Deva")
    return pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_code,
        tgt_lang=tgt_code,
        max_length=512
    )


def translate_segments(segments, src_lang="en", target_lang="hi"):
    translator_pipeline = get_translator_pipeline(src_lang, target_lang)
    translated_segments = []
    print(f"Translating subtitles from {src_lang} to {target_lang} with NLLB...")
    for seg in tqdm(segments):
        translated_text = translator_pipeline(seg['text'].strip())[0]['translation_text']
        translated_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': translated_text
        })
    return translated_segments


async def tts_segment(text, voice, duration_ms):
    """Generate a single TTS segment and pad if necessary."""
    temp_file = f"{uuid.uuid4().hex}.mp3"
    await edge_tts.Communicate(text, voice=voice).save(temp_file)
    speech = AudioSegment.from_mp3(temp_file)
    os.remove(temp_file)
    if len(speech) < duration_ms:
        speech += AudioSegment.silent(duration=duration_ms - len(speech))
    return speech


async def text_to_speech_synced(segments, output_mp3, voice):
    """Generate time-synced audio with edge-tts."""
    print(f"Generating time-synced audio: {output_mp3}")
    full_audio = AudioSegment.silent(duration=0)
    current_position = 0

    for seg in tqdm(segments):
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        duration_ms = end_ms - start_ms

        if current_position < start_ms:
            full_audio += AudioSegment.silent(duration=start_ms - current_position)

        speech = await tts_segment(seg['text'], voice, duration_ms)
        full_audio += speech
        current_position = start_ms + len(speech)

    full_audio.export(output_mp3, format="mp3")
    print(f"Time-synced audio file created: {output_mp3}")


def transcribe_with_whisper(audio_path, model_size='large', language=None):
    print(f"Loading whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    options = {"task": "transcribe"}
    if language:
        options['language'] = language

    print("Transcribing...")
    result = model.transcribe(audio_path, **options)
    detected_lang = result.get("language", None)
    return result.get('segments', []), detected_lang


def main():
    parser = argparse.ArgumentParser(description="Generate English & multi-language subtitles + synced TTS audio")
    parser.add_argument("input", help="Path to input video or audio file")
    parser.add_argument("--model", "-m", default="large", help="Whisper model size")
    parser.add_argument("--language", "-l", default=None, help="Language code for transcription")
    parser.add_argument("--target_lang", "-t", default="hi",
                        help="Target language codes for translation (comma-separated, e.g. hi,es,fr)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_srt_eng = base + ".srt"
    eng_synced_audio = base + "_audio_synced.mp3"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = extract_audio_if_video(input_path, tmpdir)

        segments, detected_lang = transcribe_with_whisper(audio_path, model_size=args.model, language=args.language)
        if not segments:
            print("No transcription segments returned.")
            return

        # Save English subtitles
        with open(out_srt_eng, "w", encoding="utf-8") as f:
            f.write(segments_to_srt(segments))
        print(f"English subtitle file written: {out_srt_eng} (Detected language: {detected_lang})")

        # Generate English synced audio
        asyncio.run(text_to_speech_synced(segments, eng_synced_audio, voice="en-US-AriaNeural"))

        target_langs = args.target_lang.split(",")
        if detected_lang == "en":
            for tgt_lang in target_langs:
                tgt_lang = tgt_lang.strip()
                if tgt_lang not in lang_code_map:
                    print(f"Target language '{tgt_lang}' not supported, skipping.")
                    continue

                translated_segments = translate_segments(segments, src_lang="en", target_lang=tgt_lang)
                out_srt_tgt = f"{base}_{tgt_lang}.srt"
                out_audio_tgt = f"{base}_{tgt_lang}_audio_synced.mp3"

                with open(out_srt_tgt, "w", encoding="utf-8") as f:
                    f.write(segments_to_srt(translated_segments))
                print(f"{tgt_lang} subtitle file written: {out_srt_tgt}")

                # Map language codes to edge-tts voices
                voice_map = {
                    "hi": "hi-IN-SwaraNeural",
                    "es": "es-ES-ElviraNeural",
                    "fr": "fr-FR-DeniseNeural",
                    "de": "de-DE-KatjaNeural",
                    "zh": "zh-CN-XiaoxiaoNeural",
                    "ar": "ar-SA-HamedNeural",
                    # Add more voices as needed
                }
                voice = voice_map.get(tgt_lang, "en-US-AriaNeural")
                asyncio.run(text_to_speech_synced(translated_segments, out_audio_tgt, voice=voice))
        else:
            print("Detected language is not English — skipping translation.")


if __name__ == "__main__":
    main()
