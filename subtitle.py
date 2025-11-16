
import argparse
import os
import subprocess
import sys
import tempfile
import uuid
import whisper
import srt
import datetime
import pysubs2
import re
import numpy as np
import inflect
import torch
import ffmpeg
import regex
import unicodedata
from deepmultilingualpunctuation import PunctuationModel    
import nltk
import string
from tqdm import tqdm
from ttsmms import download, TTS
from pydub import AudioSegment, effects
from pydub import silence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

nltk.download('punkt', quiet=True)

def restore_punctuation(text):
    """Restore punctuation for better sentence segmentation."""
    if not hasattr(restore_punctuation, "model"):
        restore_punctuation.model = PunctuationModel()
    return restore_punctuation.model.restore_punctuation(text)

def split_text_smart(text, max_chars=400):
    """Split text into sentences while respecting max token length."""
    sentences = nltk.sent_tokenize(text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < max_chars:
            cur += (" " if cur else "") + s
        else:
            chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    return chunks

def sentence_case_contextual(segments):

    fixed_segments = []
    sentence_continues = False

    # Pattern detecting *any* medical/technical acronym
    ACRONYM_PATTERN = re.compile(
        r"""
        (?:[A-Z]{2,})                 # pure uppercase acronyms (TSH, MRI, LDL)
        |(?:[A-Za-z]*\d+[A-Za-z\d]*)  # alphanumeric medical terms (T3, T4, B12, HbA1c)
        """,
        re.VERBOSE
    )

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            fixed_segments.append(seg)
            continue

        # Extract original acronyms BEFORE lowering
        original_tokens = re.findall(ACRONYM_PATTERN, text)
        original_upper = {tok.lower(): tok for tok in original_tokens}

        # Lowercase entire segment
        text = text.lower()

        # 1. Detect if the first word is an acronym → don't capitalize

        words_list = text.split()
        first_word_raw = words_list[0] if words_list else ""
        first_word_norm = re.sub(r"[^A-Za-z0-9]", "", first_word_raw)

        is_acronym_start = bool(re.fullmatch(ACRONYM_PATTERN, first_word_norm))

        if not sentence_continues and not is_acronym_start:
            match = re.search(r"[A-Za-z]", text)
            if match:
                i = match.start()
                text = text[:i] + text[i].upper() + text[i+1:]

        def cap_after_punc(match):
            return f"{match.group(1)}{match.group(2)}{match.group(3).upper()}"

        text = re.sub(r"([.!?;:])(\s*)([a-z])", cap_after_punc, text)

        # Fix pronoun "I"
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"\bi'([a-z])", lambda m: "I'" + m.group(1).upper(), text)

        for lower_tok, orig_tok in original_upper.items():
            text = re.sub(
                rf"(?<![A-Za-z0-9]){lower_tok}(?![A-Za-z0-9])",
                orig_tok,
                text,
                flags=re.IGNORECASE
            )

        sentence_continues = not bool(re.search(r"[.!?;:]\s*$", text))

        seg["text"] = text
        fixed_segments.append(seg)

    return fixed_segments

def clean_translation_text(text: str, lang_nllb: str = "eng_Latn") -> str:

    if not text:
        return text

    text = regex.sub(
        r"(?<=\b[A-Za-z0-9]+)\s*,\s*(?=\b[A-Za-z0-9]+\b)",
        " , ",
        text
    )

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible/control chars
    text = regex.sub(r"[\p{C}&&[^\n\t]]+", "", text)

    # Normalize quotes/dashes
    text = (text.replace("“", '"').replace("”", '"')
                 .replace("‘", "'").replace("’", "'")
                 .replace("–", "-").replace("—", "-").replace("―", "-"))

    # Collapse repeated punctuation
    text = regex.sub(r"([!?।,.])\1+", r"\1", text)

    # Normalize spaces
    text = regex.sub(r"\s*([,.!?;:])\s*", r"\1 ", text)
    text = regex.sub(r"\s*([(){}\[\]])\s*", r"\1", text)

    # --- Convert all script-specific numerals to Western digits
    text = regex.sub(r"[०१२३४५६७८९]", lambda m: str("०१२३४५६७८९".index(m.group(0))), text)
    text = regex.sub(r"[٠١٢٣٤٥٦٧٨٩]", lambda m: str("٠١٢٣٤٥٦٧٨٩".index(m.group(0))), text)
    text = regex.sub(r"[০১২৩৪৫৬৭৮৯]", lambda m: str("০১২৩৪৫৬৭৮৯".index(m.group(0))), text)
    text = regex.sub(r"[௦-௯]", lambda m: str(ord(m.group(0)) - 0x0BE6), text)
    text = regex.sub(r"[౦-౯]", lambda m: str(ord(m.group(0)) - 0x0C66), text)
    text = regex.sub(r"[೦-೯]", lambda m: str(ord(m.group(0)) - 0x0CE6), text)
    text = regex.sub(r"[൦-൯]", lambda m: str(ord(m.group(0)) - 0x0D66), text)

        # --- Convert Western digits (0-9) to target language numerals
    if lang_nllb.endswith("_Deva"):  # Hindi, Marathi, Nepali
        text = regex.sub(r"\d", lambda m: "०१२३४५६७८९"[int(m.group(0))], text)
    elif lang_nllb.endswith("_Beng"):  # Bengali, Assamese
        text = regex.sub(r"\d", lambda m: "০১২৩৪৫৬৭৮৯"[int(m.group(0))], text)
    elif lang_nllb.endswith("_Arab"):  # Arabic, Urdu, Persian
        text = regex.sub(r"\d", lambda m: "٠١٢٣٤٥٦٧٨٩"[int(m.group(0))], text)
    elif lang_nllb.endswith("_Taml"):  # Tamil
        text = regex.sub(r"\d", lambda m: chr(0x0BE6 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Telu"):  # Telugu
        text = regex.sub(r"\d", lambda m: chr(0x0C66 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Knda"):  # Kannada
        text = regex.sub(r"\d", lambda m: chr(0x0CE6 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Mlym"):  # Malayalam
        text = regex.sub(r"\d", lambda m: chr(0x0D66 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Orya"):  # Odia
        text = regex.sub(r"\d", lambda m: chr(0x0B66 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Sinh"):  # Sinhala
        text = regex.sub(r"\d", lambda m: chr(0x0DE6 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Mymr"):  # Burmese
        text = regex.sub(r"\d", lambda m: chr(0x1040 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Khmr"):  # Khmer
        text = regex.sub(r"\d", lambda m: chr(0x17E0 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Laoo"):  # Lao
        text = regex.sub(r"\d", lambda m: chr(0x0ED0 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Thai"):  # Thai
        text = regex.sub(r"\d", lambda m: chr(0x0E50 + int(m.group(0))), text)
    elif lang_nllb.endswith("_Ethi"):  # Amharic, Tigrinya (Ethiopic numerals are alphabetic, so skip)
        pass
    # Remove duplicated words
    tokens = text.split()
    cleaned_tokens, prev_norm = [], None
    for tok in tokens:
        norm = tok.lower().strip(string.punctuation)
        if norm == prev_norm:
            continue
        cleaned_tokens.append(tok)
        prev_norm = norm
    text = " ".join(cleaned_tokens)

    # --- Punctuation localization ---
    if lang_nllb.endswith("_Deva"):  # Hindi, Marathi, Nepali, etc.
        text = text.replace(".", "।")
        text = text.replace("?", "?")  # same
        text = text.replace(",", ",")  # same

    elif lang_nllb.endswith("_Beng"):  # Bengali, Assamese
        text = text.replace(".", "।")

    elif lang_nllb.endswith("_Arab"):  # Arabic, Urdu, Persian
        text = (text.replace(".", "۔")
                     .replace(",", "،")
                     .replace("?", "؟"))

    elif lang_nllb.endswith(("_Hans", "_Hant", "_Jpan")):  # Chinese, Japanese
        text = (text.replace(".", "。")
                     .replace("?", "？")
                     .replace("!", "！"))

    elif lang_nllb.endswith("_Hang"):  # Korean
        text = (text.replace("?", "?")
                     .replace(".", "."))  # punctuation kept but spacing normalized

    elif lang_nllb.endswith("_Cyrl"):  # Russian etc.
        text = (text.replace("?", "?")
                     .replace(".", "."))  # unchanged

    elif lang_nllb.endswith("_Taml"):  # Tamil uses Latin punctuation
        pass

    elif lang_nllb.endswith("_Mymr"):  # Burmese
        text = text.replace(".", "။")

    elif lang_nllb.endswith("_Khmr"):  # Khmer
        text = text.replace(".", "។")

    elif lang_nllb.endswith("_Ethi"):  # Amharic
        text = text.replace(".", "።")

    # Final cleanup: fix spacing
    text = regex.sub(r"\s+([,.!?।،۔؟。！？።])", r"\1", text)
    text = regex.sub(r"\s+", " ", text)
    text = regex.sub(r"([,.!?।،۔؟。！？።])\s*[,.!?।،۔؟。！？።]+", r"\1", text)

    # Remove empty brackets or stray symbols
    text = regex.sub(r"\(\s*\)", "", text)
    text = regex.sub(r"\[\s*\]", "", text)
    text = regex.sub(r"\{\s*\}", "", text)

    # Trim + capitalize first letter if lowercase (for Latin scripts)
    text = text.strip()
    if lang_nllb.endswith("_Latn") and len(text) > 1 and text[0].islower():
        text = text[0].upper() + text[1:]

    return text.strip()

def distribute_by_duration(translated_text, segments_in_block, lang_nllb="eng_Latn"):

    # --- Tokenize into words (keeps Indic ligatures intact) ---
    words = regex.findall(r"\S+", translated_text.strip())
    total_words = len(words)
    if total_words == 0 or not segments_in_block:
        return [""] * len(segments_in_block)

    # --- Compute duration ratios ---
    durations = [max(0.1, s["end"] - s["start"]) for s in segments_in_block]
    total_dur = sum(durations)
    ratios = np.array(durations) / total_dur

    # --- Cumulative allocation (avoids starving last segments) ---
    cumulative = np.round(np.cumsum(ratios) * total_words).astype(int)
    prev = 0
    distributed_texts = []

    for c in cumulative:
        c = min(c, total_words)
        piece = words[prev:c]
        distributed_texts.append(" ".join(piece).strip())
        prev = c

    # Guarantee that last segment gets all remaining words
    if len(distributed_texts) < len(segments_in_block):
        distributed_texts += [""] * (len(segments_in_block) - len(distributed_texts))
    elif len(distributed_texts) > len(segments_in_block):
        distributed_texts = distributed_texts[:len(segments_in_block)]

    # Fill empty final ones if some words remain
    if prev < total_words and distributed_texts:
        distributed_texts[-1] += " " + " ".join(words[prev:])

    # --- Ensure readable duration ---
    for seg in segments_in_block:
        dur = seg["end"] - seg["start"]
        if dur < 0.6:
            seg["end"] = seg["start"] + 0.6

    return distributed_texts

def merge_short_segments_duration_aware(segments, max_words=6, max_gap_s=0.6):

    if not segments:
        return []

    # Normalize & sort (safety)
    segments = sorted(segments, key=lambda x: x["start"])

    # Universal sentence-ending punctuation (across major scripts)
    END_PUNCT = r"[.!?।؟。！？።٫،]"

    merged = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        cur_text = cur.get("text", "").strip()
        cur_words = regex.findall(r"\S+", cur_text)

        start = cur["start"]
        end = cur["end"]
        text = cur_text

        # --- Try merging forward
        while (len(cur_words) < max_words and 
               not regex.search(f"{END_PUNCT}$", text) and 
               i + 1 < len(segments)):

            nxt = segments[i + 1]
            nxt_text = nxt.get("text", "").strip()
            nxt_words = regex.findall(r"\S+", nxt_text)

            # Compute silence/gap duration
            gap = nxt["start"] - end
            if gap > max_gap_s:
                break  # too long, natural pause → stop merging

            # Merge if total word count fits limit
            total_words = len(cur_words) + len(nxt_words)
            if total_words <= max_words:
                text = (text + " " + nxt_text).strip()
                end = nxt["end"]
                cur_words = regex.findall(r"\S+", text)
                i += 1
            else:
                break

        merged.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text.strip()
        })
        i += 1

    print(f"[MERGE SMALL+DUR] {len(segments)} → {len(merged)} segments (≤ {max_words} words, ≤ {max_gap_s:.1f}s gap)")
    return merged

def _normalize_token_for_compare(tok: str, lang_nllb: str = "eng_Latn") -> str:
    if not tok:
        return ""

    # --- Universal Unicode normalization
    tok = unicodedata.normalize("NFKC", tok)

    # --- Remove zero-width, control, or invisible chars
    tok = regex.sub(r"[\p{C}\p{Zl}\p{Zp}]+", "", tok)

    # --- Remove enclosing punctuation, quotes, etc.
    tok = regex.sub(r"^[\p{P}\p{S}]+|[\p{P}\p{S}]+$", "", tok)

    # --- Script-specific normalization ---
    if lang_nllb.endswith("_Latn"):
        # Latin: lowercase + strip accents
        tok = tok.casefold()
        tok = "".join(c for c in unicodedata.normalize("NFD", tok)
                      if unicodedata.category(c) != "Mn")

    elif lang_nllb.endswith("_Deva"):
        # Devanagari: remove nukta & chandrabindu, normalize anusvara
        tok = regex.sub(r"[\u093C\u0901\u0902]", "", tok)
        tok = regex.sub(r"ं", "न", tok)
        tok = regex.sub(r"[।॥]", "", tok)

    elif lang_nllb.endswith("_Arab"):
        # Arabic script: normalize forms and remove tatweel
        tok = regex.sub(r"ـ", "", tok)  # tatweel
        tok = regex.sub(r"[ًٌٍَُِّْٰ]", "", tok)  # diacritics
        tok = regex.sub(r"[؟،٫٬]", "", tok)
        tok = regex.sub(r"ي", "ی", tok)  # unify forms
        tok = regex.sub(r"ك", "ک", tok)

    elif lang_nllb.endswith(("_Hans", "_Hant", "_Jpan")):
        # Chinese & Japanese: remove full-width punctuation
        tok = regex.sub(r"[。、「」『』【】（）！？；：]", "", tok)

    elif lang_nllb.endswith("_Hang"):
        # Korean Hangul: no casefold; strip punct only
        tok = regex.sub(r"[.?!,·]", "", tok)

    elif lang_nllb.endswith("_Cyrl"):
        # Cyrillic: lowercase and remove punct
        tok = tok.casefold()
        tok = regex.sub(r"[.?!,]", "", tok)

    elif lang_nllb.endswith("_Beng"):
        # Bengali: remove danda and nukta
        tok = regex.sub(r"[।॥]", "", tok)
        tok = regex.sub(r"়", "", tok)

    elif lang_nllb.endswith("_Taml"):
        # Tamil: basic punctuation cleanup
        tok = regex.sub(r"[.?!,]", "", tok)

    elif lang_nllb.endswith("_Mymr"):
        # Burmese/Myanmar: remove tone marks and punctuation
        tok = regex.sub(r"[၊။]", "", tok)
        tok = regex.sub(r"[\u1037\u1038\u102B\u102C]", "", tok)

    # --- Numbers normalization (cross-script)
    tok = regex.sub(r"\d+", lambda m: str(int(m.group(0))), tok)
    tok = regex.sub(r"[०१२३४५६७८९]", lambda m: str("०१२३४५६७८९".index(m.group(0))), tok)
    tok = regex.sub(r"[٠١٢٣٤٥٦٧٨٩]", lambda m: str("٠١٢٣٤٥٦٧٨٩".index(m.group(0))), tok)

    # --- Whitespace cleanup
    tok = regex.sub(r"\s+", "", tok)

    return tok

def split_caption_text_two_lines(text, max_chars=40):

    if not text:
        return [""]

    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + (1 if cur else 0) + len(w) <= max_chars:
            cur += (" " if cur else "") + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # If already 1 or 2 lines -> good
    if len(lines) <= 2:
        return [ln.strip() for ln in lines]

    total_chars = sum(len(w) for w in words) + (len(words) - 1)
    target = total_chars // 2
    l1_words = []
    cur_len = 0
    i = 0
    while i < len(words):
        w = words[i]
        add_len = len(w) + (1 if l1_words else 0)
        if cur_len + add_len <= target or not l1_words:
            l1_words.append(w)
            cur_len += add_len
            i += 1
        else:
            break
    l1 = " ".join(l1_words).strip()
    l2 = " ".join(words[i:]).strip()
    if not l2:
        return [l1]
    return [l1, l2]

def safe_filename(name: str) -> str:
    """Sanitize filenames to avoid illegal characters."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

# ---------- Optional runtime installs ----------
def ensure_pkg(mod_name, pip_name=None):
    try:
        __import__(mod_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])

for pkg in ["transformers","sentencepiece","sacremoses","ttsmms","pydub","pysubs2","pysoundfile","inflect"]:
    ensure_pkg(pkg)

p = inflect.engine()

# ----------------------------- Utilities -----------------------------
def extract_audio_if_video(input_path, tmp_dir):
    video_exts = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(input_path.lower())
    if ext in video_exts:
        out_audio = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
        cmd = ["ffmpeg","-y","-i",input_path,"-ac","1","-ar","16000","-vn",out_audio]
        print("Extracting audio with ffmpeg...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_audio
    return input_path

def sync_embedded_subs_to_audio(
    video_path: str,
    input_srt: str,
    output_srt: str = None,
    energy_window_s: float = 0.03,
    silence_floor_db: float = -32,
    min_speech_len_s: float = 0.25,
    offset_correction_s: float = 0.0
):

    tmpdir = tempfile.mkdtemp()
    tmp_audio = os.path.join(tmpdir, "track.wav")

    print(f"[SYNC] Extracting mono audio from {video_path}...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-vn", tmp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load subtitles
    with open(input_srt, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    subs = pysubs2.load(input_srt, encoding="utf-8")

    # Load audio and compute silence
    audio = AudioSegment.from_file(tmp_audio)
    silence_thresh = audio.dBFS + silence_floor_db

    nonsilent_regions = silence.detect_nonsilent(
        audio,
        min_silence_len=int(min_speech_len_s * 1000),
        silence_thresh=silence_thresh
    )
    nonsilent_regions = [(s / 1000.0, e / 1000.0) for s, e in nonsilent_regions]
    print(f"[SYNC] Detected {len(nonsilent_regions)} speech segments.")

    def nearest_speech_time(t, direction="start"):
        """Find nearest speech-active boundary."""
        for s, e in nonsilent_regions:
            if direction == "start" and s <= t <= e:
                return t  # already inside speech
            if direction == "start" and t < s:
                return s
            if direction == "end" and s <= t <= e:
                return t
            if direction == "end" and t < s:
                continue
            if direction == "end" and t < e:
                return e
        return t  # fallback

    aligned = []
    for ev in subs:
        start_s, end_s = ev.start / 1000.0, ev.end / 1000.0

        # Align to nearest speech
        new_start = nearest_speech_time(start_s, "start")
        new_end = nearest_speech_time(end_s, "end")

        # Find overlap with actual speech
        overlaps = [max(0, min(new_end, e) - max(new_start, s)) for s, e in nonsilent_regions]
        total_overlap = sum(overlaps)

        if total_overlap < 0.2:  # mostly silence → skip
            continue

        ev.start = int(max(0, (new_start + offset_correction_s) * 1000))
        ev.end = int(max(ev.start + 150, (new_end + offset_correction_s) * 1000))
        aligned.append(ev)

    subs.events = aligned

    output_srt = output_srt or input_srt.replace(".srt", "_synced.srt")
    subs.save(output_srt, encoding="utf-8", format_="srt")

    # Restore original text lines (no loss of formatting)
    with open(output_srt, "r+", encoding="utf-8") as f:
        new_lines = f.readlines()
        for i, line in enumerate(new_lines):
            if not line.strip().isdigit() and "-->" not in line and line.strip():
                if i < len(raw_lines) and not raw_lines[i].strip().isdigit() and "-->" not in raw_lines[i]:
                    new_lines[i] = raw_lines[i]
        f.seek(0)
        f.writelines(new_lines)
        f.truncate()

    print(f"[SYNC] ✅ Subtitles re-aligned to speech (silence suppressed) → {output_srt}")
    return output_srt

def split_audio(audio_path, chunk_length_ms=30 * 1000, overlap_ms=500):
    """
    Split audio into chunks with a small overlap to avoid losing words at boundaries.
    Returns list of tuples: (chunk_path, chunk_start_offset_ms)
    """
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_paths = []
    start = 0
    idx = 0
    while start < total_ms:
        end = min(start + chunk_length_ms, total_ms)
        chunk = audio[start:end]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(tmp.name, format="wav")
        chunk_paths.append((tmp.name, start))  # return start offset for accurate timestamping
        idx += 1
        # advance start but keep overlap with previous chunk
        if end == total_ms:
            break
        start = end - overlap_ms if (end - overlap_ms) > start else end
    return chunk_paths


def segments_to_srt(segments):
    subs = []
    for i, seg in enumerate(segments, start=1):
        start_td = datetime.timedelta(seconds=float(seg['start']))
        end_td = datetime.timedelta(seconds=float(seg['end']))
        subs.append(srt.Subtitle(index=i,start=start_td,end=end_td,content=seg['text'].strip()))
    return srt.compose(subs)

def convert_srt_to_ass(srt_file, ass_file, tgt_lang_nllb, max_line_len=70, base_font_size=20):
    with open(srt_file, "rb") as f:
        raw = f.read()

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    subs = pysubs2.SSAFile.from_string(text, encoding="utf-8", format_="srt")

    font = get_font_for_lang(tgt_lang_nllb)
    if "Default" not in subs.styles:
        subs.styles["Default"] = pysubs2.SSAStyle()

    style = subs.styles["Default"]
    style.fontname = font
    style.fontsize = base_font_size
    style.wrapstyle = 2        # ✅ smart word wrapping
    style.scale_x = 0.9
    style.scale_y = 1.0
    style.marginv = 40
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER

    for ev in subs.events:
        ev.text = "\n".join(split_caption_text_two_lines(ev.text, max_chars=max_line_len))
        if ev.end <= ev.start:
            ev.end = ev.start + 100  # 100 ms minimum

    subs.save(ass_file, format_="ass", encoding="utf-8")
    print(f"[ASS] Converted safely with wrapping → {ass_file}")

def speedup_audio_to_fit_segment(audio: AudioSegment, target_duration_ms: int, max_step=1.5) -> AudioSegment:

    import tempfile
    import subprocess
    import os

    current_duration = len(audio)
    if current_duration <= target_duration_ms:
        # Too short → pad with silence
        return audio + AudioSegment.silent(duration=(target_duration_ms - current_duration))

    # Calculate speed factor (ratio)
    speed_factor = current_duration / target_duration_ms

    # If difference is small, skip processing
    if abs(speed_factor - 1.0) < 0.05:
        return audio

    # ffmpeg atempo supports only 0.5–2.0 range
    speed_factor = max(0.5, min(2.0, speed_factor))

    # Export to temp wav
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp_in.name, format="wav")

    # Use ffmpeg atempo for pitch-preserving time stretch
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_in.name,
        "-filter:a", f"atempo={speed_factor}",
        tmp_out.name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    stretched = AudioSegment.from_file(tmp_out.name, format="wav")

    # Cleanup
    os.remove(tmp_in.name)
    os.remove(tmp_out.name)

    return stretched

def split_audio_on_silence(audio_path, min_silence_len=900, silence_thresh_offset=-38, overlap_ms=600):

    audio = AudioSegment.from_file(audio_path)
    silence_thresh = audio.dBFS + silence_thresh_offset

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=overlap_ms
    )

    chunk_paths, offset = [], 0
    for chunk in chunks:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(tmp.name, format="wav")
        chunk_paths.append((tmp.name, offset))
        # keep a conservative offset step: chunk length - overlap
        offset += max(len(chunk) - overlap_ms, 0)
    return chunk_paths

# ----------------------------- Font Map -----------------------------
LANG_FONTS = {
    # Indic scripts
    "_Deva": "Noto Sans Devanagari",     # Hindi, Marathi, Sanskrit, Bhojpuri, Maithili, Nepali
    "_Beng": "Noto Sans Bengali",        # Bengali, Assamese, Manipuri (Bangla script)
    "_Gujr": "Noto Sans Gujarati",       # Gujarati
    "_Guru": "Noto Sans Gurmukhi",       # Punjabi
    "_Taml": "Noto Sans Tamil",          # Tamil
    "_Telu": "Noto Sans Telugu",         # Telugu
    "_Knda": "Noto Sans Kannada",        # Kannada
    "_Mlym": "Noto Sans Malayalam",      # Malayalam
    "_Orya": "Noto Sans Oriya",          # Odia
    "_Sinh": "Noto Sans Sinhala",        # Sinhala
    "_Olck": "Noto Sans Ol Chiki",       # Santali (Ol Chiki)
    "_Laoo": "Noto Sans Lao",            # Lao

    # East Asian
    "_Hans": "Noto Sans SC",             # Simplified Chinese
    "_Hant": "Noto Sans TC",             # Traditional Chinese
    "_Jpan": "Noto Sans JP",             # Japanese
    "_Hang": "Noto Sans KR",             # Korean Hangul
    "_Tibt": "Noto Sans Tibetan",        # Tibetan
    "_Mymr": "Noto Sans Myanmar",        # Burmese, etc.
    "_Khmr": "Noto Sans Khmer",          # Khmer
    "_Thai": "Noto Sans Thai",           # Thai

    # Middle Eastern
    "_Arab": "Noto Naskh Arabic",        # Arabic, Urdu, Persian, Pashto
    "_Hebr": "Noto Sans Hebrew",         # Hebrew
    "_Syrc": "Noto Sans Syriac",         # Syriac
    "_Tfng": "Noto Sans Tifinagh",       # Tamazight (Berber)

    # Cyrillic
    "_Cyrl": "Noto Sans",                # Russian, Ukrainian, Serbian, etc. (Noto Sans has full Cyrillic)
    "_Glag": "Noto Sans Glagolitic",     # Glagolitic (rare)

    # European Latin
    "_Latn": "Noto Sans",                # English, French, German, Spanish, etc.
    "_Grek": "Noto Sans Greek",          # Greek
    "_Armn": "Noto Sans Armenian",       # Armenian
    "_Geor": "Noto Sans Georgian",       # Georgian

    # African + others
    "_Ethi": "Noto Sans Ethiopic",       # Amharic, Tigrinya
    "_Cher": "Noto Sans Cherokee",       # Cherokee
    "_Vaii": "Noto Sans Vai",            # Vai
    "_Tale": "Noto Sans Tai Le",         # Tai scripts
    "_Talu": "Noto Sans New Tai Lue",
    "_Phag": "Noto Sans Phags Pa",

    # Default
    "default": "Noto Sans"               # Fallback
}

# ----------------------------- Language Maps -----------------------------
ISO2_TO_NLLB = {
    "hi": "hin_Deva", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya",
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    # --- Extended MMS-supported entries ---
    "af": "afr_Latn", "am": "amh_Ethi", "as": "asm_Beng", "ast": "ast_Latn", "az": "azj_Latn",
    "ba": "bak_Cyrl", "be": "bel_Cyrl", "bem": "bem_Latn", "bg": "bul_Cyrl", "bho": "bho_Deva",
    "bm": "bam_Latn", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn", "ceb": "ceb_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "dv": "div_Thaa", "dz": "dzo_Tibt",
    "el": "ell_Grek", "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fil": "fil_Latn", "fj": "fij_Latn", "fo": "fao_Latn", "fy": "fry_Latn", "ga": "gle_Latn",
    "gd": "gla_Latn", "gl": "glg_Latn", "gn": "grn_Latn", "ha": "hau_Latn", "haw": "haw_Latn",
    "he": "heb_Hebr", "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "ig": "ibo_Latn", "ilo": "ilo_Latn", "is": "isl_Latn", "it": "ita_Latn",
    "jv": "jav_Latn", "kab": "kab_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "kns": "kik_Latn", "rw": "kin_Latn", "ky": "kir_Cyrl", "ku": "ckb_Arab", "lb": "ltz_Latn",
    "lg": "lug_Latn", "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mai": "mai_Deva", "mg": "plt_Latn", "mk": "mkd_Cyrl", "mn": "khk_Cyrl", "mni": "mni_Beng",
    "mos": "mos_Latn", "ms": "msa_Latn", "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva",
    "nl": "nld_Latn", "nn": "nno_Latn", "no": "nob_Latn", "ny": "nya_Latn", "oc": "oci_Latn",
    "om": "gaz_Latn", "pap": "pap_Latn", "pl": "pol_Latn", "ps": "pbt_Arab", "qu": "quy_Latn",
    "ro": "ron_Latn", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sm": "smo_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "ss": "ssw_Latn", "st": "sot_Latn", "su": "sun_Latn",
    "sv": "swe_Latn", "sw": "swh_Latn", "taq": "taq_Latn", "tg": "tgk_Cyrl", "th": "tha_Thai",
    "ti": "tir_Ethi", "tk": "tuk_Latn", "tn": "tsn_Latn", "to": "ton_Latn", "tr": "tur_Latn",
    "ts": "tso_Latn", "tum": "tum_Latn", "tw": "twi_Latn", "tzm": "tzm_Tfng", "ug": "uig_Arab",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn", "wo": "wol_Latn",
    "xh": "xho_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "yue": "yue_Hant", "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant", "zu": "zul_Latn",
    # --- Additional MMS-supported languages ---
    "ak": "aka_Latn", "ee": "ewe_Latn", "knj": "kan_Latn", "tl": "tgl_Latn", "kg": "kon_Latn",
    "kr": "kau_Latn",
}

LETTER_MAPS = {
    "_Deva": {  # Hindi, Marathi, Nepali (Devanagari)
        "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ",
        "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
        "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
        "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू", "X": "एक्स",
        "Y": "वाय", "Z": "जेड"
    },
    "_Beng": {  # Bengali
        "A": "এ", "B": "বি", "C": "সি", "D": "ডি", "E": "ই", "F": "এফ",
        "G": "জি", "H": "এইচ", "I": "আই", "J": "জে", "K": "কে", "L": "এল",
        "M": "এম", "N": "এন", "O": "ও", "P": "পি", "Q": "কিউ", "R": "আর",
        "S": "এস", "T": "টি", "U": "ইউ", "V": "ভি", "W": "ডাবলিউ", "X": "এক্স",
        "Y": "ওয়াই", "Z": "জেড"
    },
    "_Arab": {  # Urdu / Arabic
        "A": "اے", "B": "بی", "C": "سی", "D": "ڈی", "E": "ای", "F": "ایف",
        "G": "جی", "H": "ایچ", "I": "آئی", "J": "جے", "K": "کے", "L": "ایل",
        "M": "ایم", "N": "این", "O": "او", "P": "پی", "Q": "کیو", "R": "آر",
        "S": "ایس", "T": "ٹی", "U": "یو", "V": "وی", "W": "ڈبلیو", "X": "ایکس",
        "Y": "وائے", "Z": "زی"
    },
    "_Taml": {  # Tamil (uses Latin phonetics)
        "A": "ஏ", "B": "பீ", "C": "ஸீ", "D": "டி", "E": "ஈ", "F": "எஃப்",
        "G": "ஜி", "H": "எச்", "I": "ஐ", "J": "ஜே", "K": "கே", "L": "எல்",
        "M": "எம்", "N": "என்", "O": "ஓ", "P": "பி", "Q": "க்யூ", "R": "ஆர்",
        "S": "எஸ்", "T": "டி", "U": "யூ", "V": "வி", "W": "டபிள்யூ", "X": "எக்ஸ்",
        "Y": "வை", "Z": "ஸெட்"
    },
    "_Latn": {  # fallback
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G",
        "H": "H", "I": "I", "J": "J", "K": "K", "L": "L", "M": "M", "N": "N",
        "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T", "U": "U",
        "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z"
    }
}

# ----------------------------- Auto-detect Non-Latin suffixes -----------------------------
# Extract all suffixes that are NOT "_Latn" (Latin alphabet)
NON_LATIN_SUFFIXES = sorted({
    code.split("_")[1] for code in ISO2_TO_NLLB.values() if "_" in code and not code.endswith("_Latn")
})
# Rebuild into full suffix strings like "_Deva", "_Cyrl", etc.
NON_LATIN_SUFFIXES = [f"_{suf}" for suf in NON_LATIN_SUFFIXES]

print(f"[INFO] Non-Latin suffixes detected: {NON_LATIN_SUFFIXES}")

def to_nllb_code(code: str) -> str:
    if "_" in code and len(code.split("_")[0])==3:
        return code
    return ISO2_TO_NLLB.get(code.lower(),"eng_Latn")

NLLB_TO_MMS_OVERRIDES={"zho":"cmn"}

def nllb_to_mms(nllb_code: str) -> str:
    return NLLB_TO_MMS_OVERRIDES.get(nllb_code.split("_")[0], nllb_code.split("_")[0])

def universal_normalize_text(text: str, lang_hint="eng") -> str:

    if not text:
        return text

    # If not English → minimal cleanup only
    if not lang_hint.startswith("eng"):
        return re.sub(r"\s+", " ", text).strip()

    # English-specific normalization
    abbrs = {
        "etc.": "et cetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "mr.": "mister",
        "mrs.": "missus",
        "dr.": "doctor",
        "st.": "saint"
    }

    raw_tokens = re.findall(r"\d+\.\d+|\d+%?|\w+['-]?\w*|[^\w\s]", text)
    out_tokens = []

    for tok in raw_tokens:
        lower = tok.lower()
        if lower in abbrs:
            out_tokens.extend(abbrs[lower].split())
            continue
        if tok.isalpha() and tok.isupper() and len(tok) >= 2:
            out_tokens.extend(list(tok))
            continue
        if re.fullmatch(r"\d+", tok):  # numbers → words
            try:
                out_tokens.extend(p.number_to_words(int(tok)).split())
            except Exception:
                out_tokens.append(tok)
            continue
        m_ord = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok, flags=re.IGNORECASE)
        if m_ord:
            try:
                out_tokens.extend(p.ordinal(p.number_to_words(int(m_ord.group(1)))).split())
            except Exception:
                out_tokens.append(tok)
            continue
        if re.fullmatch(r"\d+\.\d+", tok):
            integer, frac = tok.split(".")
            try:
                int_part = p.number_to_words(int(integer)).split()
                frac_part = " ".join([p.number_to_words(int(d)) for d in frac])
                out_tokens.extend(int_part + ["point"] + frac_part.split())
            except Exception:
                out_tokens.append(tok)
            continue
        m = re.match(r"([A-Za-z]+)?(\d+)([A-Za-z]+)?", tok)
        if m and (m.group(1) or m.group(3)):
            if m.group(1): out_tokens.append(m.group(1))
            try:
                out_tokens.extend(p.number_to_words(int(m.group(2))).split())
            except Exception:
                out_tokens.append(m.group(2))
            if m.group(3): out_tokens.append(m.group(3))
            continue
        out_tokens.append(tok)

    normalized = " ".join(out_tokens)
    return re.sub(r"\s+", " ", normalized).strip()

# ----------------------------- Translation (NLLB) -----------------------------
print("Loading NLLB model for translation...")
NLLB_MODEL_NAME="facebook/nllb-200-1.3B"
tokenizer=AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    NLLB_MODEL_NAME,
    torch_dtype=torch.float16,   # 👈 half precision reduces VRAM
    low_cpu_mem_usage=True,
    use_safetensors = True
).to("cuda" if torch.cuda.is_available() else "cpu")

def create_translation_pipeline(model, tokenizer, src_lang, tgt_lang, max_length=512):
    """Create translation pipeline with improved decoding."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
        max_length=max_length,
        num_beams=5,            # Beam search
        repetition_penalty=1.2, # Avoid repeated tokens
        temperature=0.7          # More natural fluency
    )

def redistribute_by_whisper_timing(translated_text, segs_in_block):

    import regex

    if not translated_text.strip():
        return [""] * len(segs_in_block)

    # Step 1️⃣: Split translated text by major punctuation
    sentences = regex.split(r'(?<=[।.!?！？。])\s+', translated_text.strip())
    n_sent = len(sentences)
    n_segs = len(segs_in_block)

    # Case A: Perfect match
    if n_sent == n_segs:
        return [s.strip() for s in sentences]

    # Case B: Too many sentences — merge smallest ones
    while len(sentences) > n_segs:
        min_i = min(range(len(sentences) - 1), key=lambda i: len(sentences[i]))
        sentences[min_i:min_i + 2] = [" ".join(sentences[min_i:min_i + 2])]
    
    # Case C: Too few sentences — split long ones by duration
    while len(sentences) < n_segs:
        longest_i = max(range(len(sentences)), key=lambda i: len(sentences[i]))
        long_s = sentences.pop(longest_i)
        half = len(long_s) // 2
        # split near space for natural break
        split_pos = long_s[:half].rfind(" ")
        if split_pos == -1:
            split_pos = half
        sentences.insert(longest_i, long_s[:split_pos].strip())
        sentences.insert(longest_i + 1, long_s[split_pos:].strip())

    # Case D: Final fallback — time-weighted slicing
    if len(sentences) != n_segs:
        total_dur = sum(max(0.1, s["end"] - s["start"]) for s in segs_in_block)
        total_chars = sum(len(c) for c in translated_text)
        offsets, parts, start = [], [], 0
        for seg in segs_in_block[:-1]:
            ratio = (seg["end"] - seg["start"]) / total_dur
            cut = int(len(translated_text) * ratio)
            parts.append(translated_text[start:start+cut].strip())
            start += cut
        parts.append(translated_text[start:].strip())
        sentences = parts

    return [clean_translation_text(s.strip()) for s in sentences]

def convert_numbers_to_words_in_lang(text: str, lang_nllb: str) -> str:

    if not text.strip():
        return text

    # === Normalize localized numerals to Arabic digits (0–9) ===
    digit_maps = {
        "०१२३४५६७८९": "0123456789", "০১২৩৪৫৬৭৮৯": "0123456789",
        "٠١٢٣٤٥٦٧٨٩": "0123456789", "௦௧௨௩௪௫௬௭௮௯": "0123456789",
        "౦౧౨౩౪౫౬౭౮౯": "0123456789", "೦೧೨೩೪೫೬೭೮೯": "0123456789",
        "൦൧൨൩൪൫൬൭൮൯": "0123456789", "០១២៣៤៥៦៧៨៩": "0123456789",
        "໐໑໒໓໔໕໖໗໘໙": "0123456789", "๐๑๒๓๔๕๖๗๘๙": "0123456789"
    }
    for native, ascii_digits in digit_maps.items():
        text = text.translate(str.maketrans(native, ascii_digits))

    # === Base number-to-words generators ===
    def hindi_num(n):
        units = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"]
        tens = ["", "दस", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]
        teens = ["ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस"]
        if n == 0: return "शून्य"
        if n < 10: return units[n]
        if 10 < n < 20: return teens[n - 11]
        if n < 100:
            t, u = divmod(n, 10)
            return (tens[t] + (" " + units[u] if u else "")).strip()
        if n < 1000:
            h, r = divmod(n, 100)
            return (units[h] + " सौ " + (hindi_num(r) if r else "")).strip()
        if n < 100000:
            th, r = divmod(n, 1000)
            return (hindi_num(th) + " हज़ार " + (hindi_num(r) if r else "")).strip()
        if n < 10000000:
            l, r = divmod(n, 100000)
            return (hindi_num(l) + " लाख " + (hindi_num(r) if r else "")).strip()
        cr, r = divmod(n, 10000000)
        return (hindi_num(cr) + " करोड़ " + (hindi_num(r) if r else "")).strip()

    def beng_num(n):
        units = ["", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]
        tens = ["", "দশ", "বিশ", "ত্রিশ", "চল্লিশ", "পঞ্চাশ", "ষাট", "সত্তর", "আশি", "নব্বই"]
        teens = ["এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোল", "সতেরো", "আঠারো", "উনিশ"]
        if n == 0: return "শূন্য"
        if n < 10: return units[n]
        if 10 < n < 20: return teens[n - 11]
        if n < 100:
            t, u = divmod(n, 10)
            return (tens[t] + (" " + units[u] if u else "")).strip()
        if n < 1000:
            h, r = divmod(n, 100)
            return (units[h] + " শত " + (beng_num(r) if r else "")).strip()
        if n < 100000:
            th, r = divmod(n, 1000)
            return (beng_num(th) + " হাজার " + (beng_num(r) if r else "")).strip()
        if n < 10000000:
            l, r = divmod(n, 100000)
            return (beng_num(l) + " লক্ষ " + (beng_num(r) if r else "")).strip()
        cr, r = divmod(n, 10000000)
        return (beng_num(cr) + " কোটি " + (beng_num(r) if r else "")).strip()

    def arabic_num(n):
        units = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
        tens = ["", "عشرة", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"]
        if n == 0: return "صفر"
        if n < 10: return units[n]
        if n < 100:
            t, u = divmod(n, 10)
            return (tens[t] + (" و " + units[u] if u else "")).strip()
        if n < 1000:
            h, r = divmod(n, 100)
            return (units[h] + " مئة " + (arabic_num(r) if r else "")).strip()
        if n < 1000000:
            th, r = divmod(n, 1000)
            return (arabic_num(th) + " ألف " + (arabic_num(r) if r else "")).strip()
        if n < 1000000000:
            m, r = divmod(n, 1000000)
            return (arabic_num(m) + " مليون " + (arabic_num(r) if r else "")).strip()
        b, r = divmod(n, 1000000000)
        return (arabic_num(b) + " مليار " + (arabic_num(r) if r else "")).strip()

    def english_num(n):
        units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        teens = ["eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        if n == 0: return "zero"
        if n < 10: return units[n]
        if 10 < n < 20: return teens[n - 11]
        if n < 100:
            t, u = divmod(n, 10)
            return (tens[t] + (" " + units[u] if u else "")).strip()
        if n < 1000:
            h, r = divmod(n, 100)
            return (units[h] + " hundred " + (english_num(r) if r else "")).strip()
        if n < 1000000:
            th, r = divmod(n, 1000)
            return (english_num(th) + " thousand " + (english_num(r) if r else "")).strip()
        if n < 1000000000:
            m, r = divmod(n, 1000000)
            return (english_num(m) + " million " + (english_num(r) if r else "")).strip()
        b, r = divmod(n, 1000000000)
        return (english_num(b) + " billion " + (english_num(r) if r else "")).strip()

    def dravidian_num(n, lang):
        DIGITS = {
            "_Taml": ["பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"],
            "_Telu": ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"],
            "_Knda": ["ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು"],
            "_Mlym": ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒന്‍പത്"]
        }
        nums = DIGITS.get(lang, DIGITS["_Taml"])
        return " ".join(nums[int(d)] for d in str(n))

    # === Main dispatcher by language ===
    def num_to_words(n):
        if lang_nllb.endswith("_Deva"): return hindi_num(n)
        if lang_nllb.endswith("_Beng"): return beng_num(n)
        if lang_nllb.endswith(("_Taml", "_Telu", "_Knda", "_Mlym")): return dravidian_num(n, lang_nllb[-5:])
        if lang_nllb.endswith("_Arab"): return arabic_num(n)
        if lang_nllb.endswith("_Latn"): return english_num(n)
        return english_num(n)

    # === Fractions and percentages ===
    FRACTIONS = {
        "½": {"_Deva": "आधा", "_Beng": "অর্ধেক", "_Taml": "அரை", "_Arab": "نصف", "_Latn": "half"},
        "¼": {"_Deva": "पौना", "_Beng": "চতুর্থাংশ", "_Taml": "கால்", "_Arab": "ربع", "_Latn": "quarter"},
        "¾": {"_Deva": "सवा", "_Beng": "তিন চতুর্থাংশ", "_Taml": "மூன்று கால்", "_Arab": "ثلاثة أرباع", "_Latn": "three quarters"},
    }
    for frac, mapping in FRACTIONS.items():
        if frac in text:
            word = mapping.get(next((k for k in mapping if lang_nllb.endswith(k)), "_Latn"))
            text = text.replace(frac, word)

    def handle_percent(match):
        num = int(match.group(1))
        word = num_to_words(num)
        if lang_nllb.endswith("_Deva"): return f"{word} प्रतिशत"
        if lang_nllb.endswith("_Beng"): return f"{word} শতাংশ"
        if lang_nllb.endswith("_Arab"): return f"{word} في المئة"
        if lang_nllb.endswith("_Latn"): return f"{word} percent"
        return f"{word} percent"

    text = regex.sub(r"\b(\d{1,4})\s*%", handle_percent, text)

    # === Handle decimals ===
    def handle_decimal(match):
        val = match.group(0)
        if "." in val:
            int_part, frac_part = val.split(".")
            int_words = num_to_words(int(int_part))
            frac_words = " ".join(num_to_words(int(d)) for d in frac_part)
            if lang_nllb.endswith("_Deva"):
                return f"{int_words} दशमलव {frac_words}"
            elif lang_nllb.endswith("_Beng"):
                return f"{int_words} দশমিক {frac_words}"
            elif lang_nllb.endswith("_Arab"):
                return f"{int_words} فاصلة {frac_words}"
            else:
                return f"{int_words} point {frac_words}"
        return num_to_words(int(val))

    # Replace decimals and integers
    text = regex.sub(r"\b\d+(\.\d+)?\b", handle_decimal, text)

    return text.strip()

def _merge_by_sentence_boundaries(segments, lang_nllb="eng_Latn"):

    if not segments:
        return []

    END_PUNCT = r"[.!?।؟。！？።]"
    sentence_re = regex.compile(rf"(.+?{END_PUNCT})(\s+|$)")

    merged = []
    cur_text = ""
    cur_start = None
    cur_end = None

    for seg in sorted(segments, key=lambda s: s["start"]):

        text = seg["text"].strip()
        if not text:
            continue

        # FIND all full sentences inside the segment
        matches = list(sentence_re.finditer(text))

        if matches:
            # There are one or more complete sentences inside this segment
            prev_end = 0

            for m in matches:
                full_sentence = m.group(1).strip()

                # proportional timing start/end inside segment
                char_start = m.start(1)
                char_end = m.end(1)
                seg_len = max(len(text), 1)

                est_start = seg["start"] + (char_start / seg_len) * (seg["end"] - seg["start"])
                est_end = seg["start"] + (char_end / seg_len) * (seg["end"] - seg["start"])

                # If this is the very first chunk of a multi-seg sentence
                if cur_start is None:
                    cur_start = est_start

                cur_text += (" " if cur_text else "") + full_sentence
                cur_end = est_end

                # Sentence complete → commit
                merged.append({
                    "start": round(cur_start, 3),
                    "end": round(cur_end, 3),
                    "text": cur_text.strip()
                })

                # Reset for next sentence
                cur_text = ""
                cur_start = None
                cur_end = None

                prev_end = m.end()

            # Check if there's leftover trailing text (after last punctuation)
            leftover = text[prev_end:].strip()
            if leftover:
                # Start a new partial sentence
                # timing = from end of last full sentence to end of segment
                char_start = prev_end
                seg_len = max(len(text), 1)
                est_start = seg["start"] + (char_start / seg_len) * (seg["end"] - seg["start"])

                cur_start = est_start if cur_start is None else cur_start
                cur_text = leftover
                cur_end = seg["end"]

        else:
            # No sentence-ending punctuation inside this segment
            if cur_start is None:
                cur_start = seg["start"]

            cur_text += (" " if cur_text else "") + text
            cur_end = seg["end"]

    # Commit remaining partial sentence
    if cur_text:
        merged.append({
            "start": round(cur_start, 3),
            "end": round(cur_end, 3),
            "text": cur_text.strip()
        })

    print(f"[MERGE] {len(segments)} segments → {len(merged)} sentence blocks (accurate timing)")
    return merged

def force_target_language_only(text: str, tgt_lang_nllb: str, translator=None, src_lang_nllb: str = "eng_Latn") -> str:

    if not text.strip():
        return text
    
    # --- Detect script of target language ---
    script_hint = tgt_lang_nllb.split("_")[-1]
    script_name_map = {
        "Deva": "Devanagari", "Beng": "Bengali", "Guru": "Gurmukhi", "Gujr": "Gujarati",
        "Taml": "Tamil", "Telu": "Telugu", "Knda": "Kannada", "Mlym": "Malayalam",
        "Orya": "Oriya", "Sinh": "Sinhala", "Arab": "Arabic", "Cyrl": "Cyrillic",
        "Grek": "Greek", "Hans": "Han", "Hant": "Han", "Jpan": "Han",
        "Hang": "Hangul", "Mymr": "Myanmar", "Khmr": "Khmer", "Thai": "Thai",
        "Ethi": "Ethiopic", "Latn": "Latin"
    }
    tgt_script = script_name_map.get(script_hint, "Latin")

    # --- Source script detection for removal ---
    src_script_hint = src_lang_nllb.split("_")[-1] if src_lang_nllb else "Latn"
    src_script = script_name_map.get(src_script_hint, "Latin")

    # --- Static letter map for acronym expansion ---
    letter_map = next(
        (LETTER_MAPS[k] for k in LETTER_MAPS if tgt_lang_nllb.endswith(k)),
        LETTER_MAPS["_Latn"]
    )

    # --- Tokenization ---
    words = text.split()
    fixed_words = []

    # --- Helper: check if a token is in a given script ---
    def is_in_script(token, script_name):
        return not regex.search(rf"[^\p{{{script_name}}}\p{{N}}\p{{P}}\p{{S}}\p{{Z}}]", token)

    # --- Pass 1: Translate only non-target-script words ---
    for w in words:
        if not regex.search(r"\p{L}", w):  # Skip pure punctuation/numbers
            fixed_words.append(w)
            continue

        # If already in target script → leave unchanged
        if is_in_script(w, tgt_script):
            fixed_words.append(w)
            continue

        # Detect if in source script (i.e., untranslated)
        in_source_script = is_in_script(w, src_script)
        is_acronym = bool(regex.fullmatch(r"[A-Z0-9]{1,6}", w.strip(string.punctuation)))

        tr_word = None
        if in_source_script or is_acronym:
            try:
                # Try translating via model
                if translator:
                    tr_word = translator(w, max_length=64)[0]["translation_text"].strip()
                    tr_word = clean_translation_text(tr_word, tgt_lang_nllb)
                    # Reject unchanged echos
                    if _normalize_token_for_compare(tr_word) == _normalize_token_for_compare(w):
                        tr_word = None

                # Always attempt acronym expansion if translation failed or acronym detected
                if is_acronym or (in_source_script and not tr_word):
                    letters = []
                    for ch in w:
                        if ch.isalpha():
                            mapped = letter_map.get(ch.upper())
                            letters.append(mapped if mapped else ch)
                        elif ch.isdigit():
                            letters.append(ch)
                    tr_word = "".join(letters)  # 👈 No spaces between letters
                    
                # Fallback → lowercase original
                if not tr_word:
                    tr_word = w.lower()

            except Exception:
                tr_word = w.lower()

            fixed_words.append(tr_word)
        else:
            fixed_words.append(w)

    joined = " ".join(fixed_words)

    # Remove source-script remnants only if source ≠ target script
    if src_script != tgt_script:
        # --- PATCH: preserve multi-letter Latin acronyms like TSH, T3, T4, MRI, etc. ---
        def _preserve_acronyms(m):
            token = m.group(0)
            # If it's part of an acronym (2–6 letters or letter+digit), keep it
            if regex.fullmatch(r"[A-Z0-9]{2,6}", token):
                return token
            return ""

        # --- HARD PATCH: preserve punctuation around Latin acronyms ---
        # Do NOT remove commas or punctuation adjacent to acronyms
        joined = regex.sub(
            rf"(?<![,،؛؛،।。！？!?])[\p{{{src_script}}}]+(?![,،؛؛،।。！？!?])",
            _preserve_acronyms,
            joined
        )

    return joined.strip()

def expand_acronyms_for_tts(text, tgt_lang_nllb, translator=None, join_acronyms=False):

    if not text or not text.strip():
        return text

    ACR = r"(?:[A-Z]{2,}|[A-Za-z]*\d+[A-Za-z\d]*)"
    text = regex.sub(
        rf"({ACR})\s+(?={ACR}\b)",
        r"\1 , ",
        text
    )
    letter_map = next(
        (LETTER_MAPS[k] for k in LETTER_MAPS if tgt_lang_nllb.endswith(k)),
        LETTER_MAPS["_Latn"]
    )

    # English phonetic expansion for each letter
    ENGLISH_LETTERS = {
        "A": "ay", "B": "bee", "C": "cee", "D": "dee", "E": "ee",
        "F": "ef", "G": "jee", "H": "aitch", "I": "eye", "J": "jay",
        "K": "kay", "L": "el", "M": "em", "N": "en", "O": "oh",
        "P": "pee", "Q": "cue", "R": "ar", "S": "ess", "T": "tee",
        "U": "you", "V": "vee", "W": "double you", "X": "ex",
        "Y": "why", "Z": "zee"
    }

    def digit_to_word(d):
        if tgt_lang_nllb.endswith("_Deva"):   # Hindi/Marathi
            return {
                "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन",
                "4": "चार", "5": "पाँच", "6": "छह", "7": "सात",
                "8": "आठ", "9": "नौ"
            }[d]
        # Other languages fallback (English etc.)
        return p.number_to_words(int(d))

    tokens = regex.findall(r"([\p{L}\p{N}]+|[^\p{L}\p{N}\s]+|\s+)", text)
    expanded_tokens = []

    for tok in tokens:

        # Keep spaces/punctuation as-is
        if tok.isspace() or not regex.search(r"\p{L}", tok):
            expanded_tokens.append(tok)
            continue

        stripped = tok.strip(string.punctuation)

        # Detect acronyms (TSH, MRI, T3, B12, HbA1c, etc.)
        is_acronym = bool(
            regex.fullmatch(r"[A-Z0-9]{2,}|[A-Z]\d+|[A-Za-z]*\d+[A-Za-z\d]*", stripped)
        )

        if not is_acronym:
            expanded_tokens.append(tok)
            continue

        english_parts = []

        for ch in stripped:
            if ch.isalpha():
                english_parts.append(ENGLISH_LETTERS.get(ch.upper(), ch))
            elif ch.isdigit():
                # ⭐ KEY FIX ⭐ — convert digits to full words BEFORE translation
                english_parts.append(digit_to_word(ch))
            else:
                english_parts.append(ch)

        english_phrase = " ".join(english_parts)

        target_parts = []
        for part in english_phrase.split():
            mapped = letter_map.get(part.upper())
            target_parts.append(mapped if mapped else part)

        final = " ".join(target_parts)

        # Ensure final script cleanup
        final = clean_translation_text(final, tgt_lang_nllb)

        expanded_tokens.append(final)

    return "".join(expanded_tokens).strip()

def safe_chunk_text(text, max_chars=2500):
    """
    Safely chunk text for NLLB translation without losing any words.
    """
    chunks, cur = [], []
    for word in text.split():
        cur.append(word)
        if sum(len(w) + 1 for w in cur) > max_chars:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def translate_segments_nllb_batched(
    segments,
    src_lang_nllb,
    tgt_lang_nllb,
    batch_size=8,
    max_length=512,
    src_reference_segments=None
):

    merged_blocks = _merge_by_sentence_boundaries(segments, lang_nllb=src_lang_nllb)
    print(f"[MERGE] {len(merged_blocks)} sentence-level blocks created from {len(segments)} Whisper segments.")

    translator = create_translation_pipeline(
        nllb_model, tokenizer, src_lang_nllb, tgt_lang_nllb, max_length=max_length
    )

    # Helper: detect if translation contains mixed scripts
    def is_mixed_language(text, tgt_script):
        non_target_chars = regex.findall(rf"[^\p{{{tgt_script}}}\p{{P}}\p{{S}}\p{{N}}\p{{Z}}]", text)
        return len(non_target_chars) > 0

    script_hint = tgt_lang_nllb.split("_")[-1]
    script_name_map = {
        "Deva": "Devanagari", "Beng": "Bengali", "Guru": "Gurmukhi", "Gujr": "Gujarati",
        "Taml": "Tamil", "Telu": "Telugu", "Knda": "Kannada", "Mlym": "Malayalam",
        "Orya": "Oriya", "Sinh": "Sinhala", "Arab": "Arabic", "Cyrl": "Cyrillic",
        "Grek": "Greek", "Hans": "Han", "Hant": "Han", "Jpan": "Han",
        "Hang": "Hangul", "Mymr": "Myanmar", "Khmr": "Khmer", "Thai": "Thai",
        "Ethi": "Ethiopic", "Latn": "Latin"
    }
    tgt_script = script_name_map.get(script_hint, "Latin")

    translated_blocks = []

    # ---------------- TRANSLATE BLOCKS ---------------- #
    for block in tqdm(merged_blocks, desc="[NLLB] Translating sentence blocks"):
        src_text = restore_punctuation(block["text"].strip())

        # split into safe chunks
        chunks = split_text_smart(src_text, max_chars=400)
        tr_parts = []
        for c in chunks:
            try:
                out = translator(
                    c,
                    max_length=max_length,
                    repetition_penalty=1.3,
                    num_beams=5,
                    temperature=0.7
                )[0]["translation_text"].strip()
                out = clean_translation_text(out, tgt_lang_nllb)

                # re-translate if mixed language
                if is_mixed_language(out, tgt_script):
                    print(f"[WARN] Mixed script detected, retrying: {out[:50]}...")
                    out_retry = translator(
                        c,
                        max_length=max_length,
                        repetition_penalty=1.4,
                        num_beams=6,
                        temperature=0.6
                    )[0]["translation_text"].strip()
                    out = clean_translation_text(out_retry, tgt_lang_nllb)

                tr_parts.append(out)

            except Exception as e:
                print(f"[WARN] Translation chunk failed: {e}")
                tr_parts.append(c)

        tr_text = " ".join(tr_parts).strip()
        tr_text = clean_translation_text(tr_text, tgt_lang_nllb)
        tr_text = force_target_language_only(tr_text, tgt_lang_nllb, translator)
        tr_text = collapse_repeated_runs_in_text(tr_text)

        translated_blocks.append({
            "start": block["start"],
            "end": block["end"],
            "text": tr_text
        })

    translated_segments = []

    for block in translated_blocks:

        segs_in_block = [
            s for s in segments
            if (s["start"] >= block["start"] - 0.05 and s["end"] <= block["end"] + 0.05)
        ]
        if not segs_in_block:
            segs_in_block = [block]

        # duration metadata for proportional slicing
        meta = [{"start": s["start"], "end": s["end"]} for s in segs_in_block]

        distributed = distribute_by_duration(
            block["text"],
            meta,
            lang_nllb=tgt_lang_nllb
        )

        # assign redistributed sentences back to whisper segments
        for seg, txt in zip(segs_in_block, distributed):
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": clean_translation_text(txt, tgt_lang_nllb)
            })

    # ---------------- FINAL CLEANUP ---------------- #
    translated_segments = tidy_translated_segments(translated_segments)

    for seg in translated_segments:
        cleaned = clean_translation_text(seg["text"], tgt_lang_nllb)
        cleaned = collapse_repeated_runs_in_text(cleaned)
        seg["text"] = cleaned

    translated_segments = sentence_case_contextual(translated_segments)

    print(f"[INFO] ✅ {len(translated_segments)} translated segments aligned using duration-based redistribution.")
    return translated_segments

def translate_srt_file_segment(input_srt, output_srt, src_lang_nllb, tgt_lang_nllb):
    """
    Translates an SRT file segment-by-segment using NLLB,
    preserving timings and punctuation. Always works on file paths.
    """

    # === Load source SRT ===
    subs = pysubs2.load(input_srt, encoding="utf-8")

    # === Create translator ===
    translator = create_translation_pipeline(
        nllb_model, tokenizer, src_lang_nllb, tgt_lang_nllb
    )

    # === Merge sentences based on punctuation ===
    merged_blocks = []
    cur_block, cur_start, cur_end, cur_events = [], None, None, []
    for ev in subs:
        txt = ev.text.strip()
        if not txt:
            continue

        if not cur_block:
            cur_start = ev.start

        cur_block.append(txt)
        cur_events.append(ev)
        cur_end = ev.end

        # sentence end markers (multi-script compatible)
        if txt.endswith((".", "?", "!", "।")):
            merged_blocks.append({
                "start": cur_start,
                "end": cur_end,
                "events": list(cur_events)
            })
            cur_block, cur_start, cur_end, cur_events = [], None, None, []

    # handle any remainder
    if cur_block:
        merged_blocks.append({
            "start": cur_start,
            "end": cur_end,
            "events": list(cur_events)
        })

    # === Translate each merged block ===
    translated_blocks = []
    for block in tqdm(merged_blocks, desc="[NLLB] Translating SRT blocks"):
        block_text = restore_punctuation(" ".join(ev.text.strip() for ev in block["events"]))

        # --- SAFE CHUNKED TRANSLATION (no skipped sentences) ---
        translated_chunks = []
        for subtext in safe_chunk_text(block_text, max_chars=2500):
            success = False
            while not success:
                try:
                    tr_chunk = translator(
                        subtext,
                        max_length=2048,
                        num_beams=5,
                        repetition_penalty=1.2,
                        temperature=0.7
                    )[0]["translation_text"].strip()
                    tr_chunk = clean_translation_text(tr_chunk, tgt_lang_nllb)
                    translated_chunks.append(tr_chunk)
                    success = True
                except Exception as e:
                    print(f"[RETRY] Translation failed, retrying chunk: {e}")
                    continue

        # --- Combine and clean ---
        tr_text = " ".join(translated_chunks)
        tr_text = clean_translation_text(tr_text, tgt_lang_nllb)
        tr_text = force_target_language_only(tr_text, tgt_lang_nllb, translator)
        tr_text = collapse_repeated_runs_in_text(tr_text)

        translated_blocks.append({
            "start": block["start"],
            "end": block["end"],
            "text": tr_text,
            "events": block["events"]
        })

    # === Redistribute translated text by original segment durations ===
    final_events = []
    for block in translated_blocks:
        events = block["events"]

        segments_meta = [{"start": e.start / 1000, "end": e.end / 1000} for e in events]
        assigned_texts = distribute_by_duration(
            block["text"], segments_meta, lang_nllb=tgt_lang_nllb
        )

        for ev, txt in zip(events, assigned_texts):
            ev.text = force_target_language_only(
                clean_translation_text(txt, tgt_lang_nllb),
                tgt_lang_nllb, translator
            )
            final_events.append(ev)
            
        # === APPLY WHISPER-LIKE CONTEXTUAL SENTENCE CASING ON FINAL SRT EVENTS ===
        print("[INFO] Applying final contextual sentence casing...")

        casing_segments = [{"text": ev.text} for ev in final_events]
        cased_segments = sentence_case_contextual(casing_segments)

        for seg, ev in zip(cased_segments, final_events):
            ev.text = seg["text"]

        # === Save translated subtitles ===
        ssa = pysubs2.SSAFile()
        ssa.events = final_events
        ssa.save(output_srt, format_="srt", encoding="utf-8")
        print(f"[SAVE] ✅ Translated SRT (target-only, adaptive-timed) → {output_srt}")

def get_font_for_lang(tgt_lang_nllb: str) -> str:
    for suffix, font in LANG_FONTS.items():
        if tgt_lang_nllb.endswith(suffix):
            return font
    return LANG_FONTS["default"]

# ----------------------------- MMS TTS with adaptive audible speed -----------------------------
tts_models = {}

def get_tts_model(tgt_lang_nllb):
    mms_code = nllb_to_mms(tgt_lang_nllb)
    model_dir = os.path.join("models", mms_code)
    if tgt_lang_nllb not in tts_models:
        if not os.path.exists(model_dir):
            print(f"Downloading MMS TTS model for {tgt_lang_nllb} → {mms_code} ...")
            download(mms_code, "./models")
        tts_models[tgt_lang_nllb] = TTS(model_dir)
    return tts_models[tgt_lang_nllb]

def split_caption_text(text, max_chars=60):
    """Split caption text intelligently without splitting words."""
    if len(text) <= max_chars:
        return [text.strip()]
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def detect_silences_ffmpeg(audio_path, silence_thresh_db=-38, min_silence_s=0.8):

    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_s}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    silence_starts = re.findall(r"silence_start: ([0-9.]+)", result.stderr)
    silence_ends = re.findall(r"silence_end: ([0-9.]+)", result.stderr)
    return [(float(s), float(e)) for s, e in zip(silence_starts, silence_ends)]


def build_audio_driven_subs(
    words,
    audio_path,
    max_chars=42,
    min_display_s=0.4,
    max_duration_s=5.0,
    silence_gap_s=0.6,
    silence_thresh_db=-38
):

    if not words:
        return []

    # === Detect silences using ffmpeg ===
    silences = detect_silences_ffmpeg(audio_path, silence_thresh_db, silence_gap_s)
    silence_boundaries = sorted({s for pair in silences for s in pair})

    subs = []
    cur_words = []
    cur_start = words[0]["start"]

    def finalize_segment(cur_words):
        if not cur_words:
            return None
        text = " ".join(w["word"] for w in cur_words).strip()
        seg_start = cur_words[0]["start"]
        seg_end = cur_words[-1]["end"]
        seg_dur = seg_end - seg_start

        if seg_dur < min_display_s:
            seg_end = seg_start + min_display_s
        if seg_dur > max_duration_s:
            seg_end = seg_start + max_duration_s

        return {
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": "\n".join(split_caption_text_two_lines(text, max_chars=max_chars))
        }

    def next_silence_after(t):
        return min((s for s in silence_boundaries if s > t), default=None)

    for i, w in enumerate(words):
        cur_words.append(w)
        next_gap = (words[i+1]["start"] - w["end"]) if i < len(words) - 1 else 0
        ends_with_punct = bool(regex.search(r'[.?!।…]$', w["word"]))

        # --- Check silence boundaries ---
        next_sil = next_silence_after(w["end"])
        silence_break = next_sil is not None and (next_sil - w["end"]) < 0.1

        seg_dur = w["end"] - cur_start
        too_long = seg_dur >= max_duration_s
        too_texty = len(" ".join(x["word"] for x in cur_words)) > max_chars

        if too_long or too_texty or next_gap > silence_gap_s or ends_with_punct or silence_break or i == len(words) - 1:
            seg = finalize_segment(cur_words)
            if seg:
                subs.append(seg)
            cur_words = []
            if i < len(words) - 1:
                cur_start = words[i+1]["start"]

    # --- Final smoothing: prevent overlaps & enforce monotonicity ---
    for i in range(len(subs) - 1):
        if subs[i]["end"] > subs[i+1]["start"]:
            subs[i]["end"] = max(subs[i]["start"] + 0.2, subs[i+1]["start"] - 0.05)

    print(f"[SYNC+FFMPEG] {len(subs)} subtitles built (avg {np.mean([s['end']-s['start'] for s in subs]):.2f}s each)")
    return subs

def tts_from_srt_global_fit_refined(
    srt_file,
    tgt_lang_nllb,
    video_path,
    output_audio,
    pre_pad_ms=200,
    post_pad_ms=300,
    fade_ms=80
):
    p = inflect.engine()
    subs = pysubs2.load(srt_file, encoding="utf-8")
    full_text = " ".join(sub.text.strip() for sub in subs if sub.text.strip())
    if not full_text:
        raise ValueError("No subtitle text to speak.")

    tts_model = get_tts_model(tgt_lang_nllb)

    # -------------------------------------------------------------
    # 🔹 English letter-to-sound map
    # -------------------------------------------------------------
    ENGLISH_LETTERS = {
        "A": "ay", "B": "bee", "C": "cee", "D": "dee", "E": "ee",
        "F": "ef", "G": "jee", "H": "aitch", "I": "eye", "J": "jay",
        "K": "kay", "L": "el", "M": "em", "N": "en", "O": "oh",
        "P": "pee", "Q": "cue", "R": "ar", "S": "ess", "T": "tee",
        "U": "you", "V": "vee", "W": "double you", "X": "ex",
        "Y": "why", "Z": "zee"
    }
    def expand_acronyms_and_medical_terms(text):

        def looks_like_acronym(token):
            # Pure uppercase (TSH, MRI, HDL)
            if re.fullmatch(r"[A-Z]{2,}", token):
                return True
            # Mixed alphanumeric (HbA1c, B12, H1N1, D3)
            if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z\d]*", token):
                return True
            # One uppercase + digits (T3, D4)
            if re.fullmatch(r"[A-Z]\d+", token):
                return True
            return False

        def expand_token(tok):
            if not looks_like_acronym(tok):
                return tok
            expanded = []
            for ch in tok:
                if ch.isalpha():
                    expanded.append(ENGLISH_LETTERS.get(ch.upper(), ch))
                elif ch.isdigit():
                    expanded.append(p.number_to_words(int(ch)))
                else:
                    expanded.append(ch)
            return " ".join(expanded)

        tokens = re.findall(r"[A-Za-z0-9]+|\S", text)
        expanded_tokens = [expand_token(t) for t in tokens]
        return " ".join(expanded_tokens)

    # Always expand acronyms before TTS synthesis
    full_text = expand_acronyms_for_tts(full_text, tgt_lang_nllb, join_acronyms=True)
    full_text = convert_numbers_to_words_in_lang(full_text, tgt_lang_nllb)

    wav_out = tts_model.synthesis(full_text)
    y = wav_out["x"]
    sr = int(wav_out["sampling_rate"])
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y / max(1.0, np.max(np.abs(y))) * 0.9

    tensor_int16 = (y * 32767).astype(np.int16)
    tts_audio = AudioSegment(
        tensor_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    tts_audio = AudioSegment.silent(duration=pre_pad_ms) + tts_audio + AudioSegment.silent(duration=post_pad_ms)
    if len(tts_audio) > 2 * fade_ms:
        tts_audio = tts_audio.fade_in(fade_ms).fade_out(fade_ms)
    tts_audio = effects.normalize(tts_audio, headroom=3.0)

    pitch_factor = 0.90  # slightly deeper
    tts_audio = tts_audio._spawn(tts_audio.raw_data, overrides={
        "frame_rate": int(tts_audio.frame_rate * pitch_factor)
    }).set_frame_rate(tts_audio.frame_rate)

    probe = ffmpeg.probe(video_path)
    video_duration = float(probe["format"]["duration"]) * 1000.0
    tts_duration = len(tts_audio)
    video_duration *= 0.98
    speed_factor = tts_duration / video_duration
    speed_factor = max(0.5, min(2.0, speed_factor))

    print(f"[SYNC] Video={video_duration/1000:.2f}s | TTS={tts_duration/1000:.2f}s | speed_factor={speed_factor:.3f}")

    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.close(); tmp_out.close()

    tts_audio.export(tmp_in.name, format="wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_in.name,
        "-filter:a", f"atempo={speed_factor}",
        tmp_out.name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    final_audio = AudioSegment.from_file(tmp_out.name, format="wav")

    for f in [tmp_in.name, tmp_out.name]:
        try:
            os.remove(f)
        except:
            pass

    final_audio.export(output_audio, format="mp3")
    print(f"[TTS] ✅ Final acronym-safe medical TTS saved → {output_audio}")

    return output_audio

# ----------------------------- Whisper -----------------------------
def transcribe_in_chunks(
    audio_path,
    language=None,
    chunk_length_ms=None,
    overlap_ms=1000,
    use_silence_splitting=True,
    beam_size=5,
    temperature=0.0
):

    # --- Prefer GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Whisper large-v3 with fallback ---
    try:
        print(f"[WHISPER] Loading large-v3 model on {device} ...")
        model = whisper.load_model("large-v3", device=device)
        model_size = "large-v3"
        print(f"[OK] Whisper model loaded: {model_size} on {device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[WARN] CUDA OOM – trying smaller model (medium).")
            torch.cuda.empty_cache()
            model = whisper.load_model("medium", device=device)
            model_size = "medium"
        else:
            raise

    # === Split audio ===
    if use_silence_splitting:
        print("🔊 Splitting audio by silence for natural subtitle timing...")
        chunk_info = split_audio_on_silence(
            audio_path,
            min_silence_len=800,
            silence_thresh_offset=-38,
            overlap_ms=overlap_ms
        )
    else:
        if not chunk_length_ms:
            chunk_length_ms = 45 * 1000
        print(f"🔊 Splitting audio into {chunk_length_ms//1000}s chunks with {overlap_ms}ms overlap...")
        chunk_info = split_audio(audio_path, chunk_length_ms, overlap_ms=overlap_ms)

    all_words = []
    detected_lang = None

    # === Transcribe chunks ===
    for idx, (chunk_path, chunk_start_ms) in enumerate(tqdm(chunk_info, desc=f"[{model_size.upper()}] Transcribing chunks")):
        opts = dict(
            word_timestamps=True,
            verbose=False,
            temperature=temperature,
            beam_size=beam_size,
            condition_on_previous_text=False  # disable context carryover
        )

        if idx == 0 and language is None:
            res = model.transcribe(chunk_path, language=None, **opts)
            detected_lang = res.get("language", None)
            print(f"[LANG] Auto-detected language: {detected_lang}")
        else:
            res = model.transcribe(chunk_path, language=(language or detected_lang), **opts)

        offset = chunk_start_ms / 1000.0

        # Collect word-level timestamps
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                all_words.append({
                    "word": word_text,
                    "start": float(w.get("start", 0.0)) + offset,
                    "end": float(w.get("end", 0.0)) + offset,
                })

    segments = build_audio_driven_subs(
        all_words,
        audio_path=audio_path,
        max_chars=42,
        min_display_s=0.4,
        max_duration_s=5.0,
        silence_gap_s=0.6
    )


    for seg in segments:
        if seg["end"] <= seg["start"]:
            seg["end"] = seg["start"] + 0.05

    avg_len = sum(s["end"] - s["start"] for s in segments) / len(segments)
    print(f"[INFO] Segments: {len(segments)} (avg {avg_len:.2f}s each) from model {model_size}")

    return segments, (language or detected_lang), all_words

def adaptive_min_display(text, base_per_char=50, min_ms=800, max_ms=3000):
    est = len(text) * base_per_char
    return max(min_ms, min(est, max_ms))

def enforce_line_wrap(text, max_len=40):
    """Force line breaks for ASS/SRT so ffmpeg renderer never skips words."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur += (" " if cur else "") + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def refine_srt(input_srt, output_srt, max_line_length=42, lang_hint="eng"):

    subs = pysubs2.load(input_srt, encoding="utf-8")
    refined = []

    for ev in subs:
        text = ev.text.strip()

        if lang_hint.startswith("eng"):
            # cleanup fillers
            text = re.sub(r"\b(uh|um|ah|er|hmm)\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            if text and not text.endswith((".", "?", "!")):
                text += "."
            if text:
                text = text[0].upper() + text[1:]

        # line wrapping only
        wrapped_lines = split_caption_text_two_lines(text, max_chars=max_line_length)
        ev.text = "\n".join(wrapped_lines)
        refined.append(ev)

    ssa_file = pysubs2.SSAFile()
    ssa_file.events = refined
    ssa_file.save(output_srt, encoding="utf-8", format_="srt")
    print(f"[SAFE] Refined subtitles saved (timings preserved) → {output_srt}")

def refine_subtitle_sync(segments, all_words, audio_path, margin_ms=80):

    audio = AudioSegment.from_file(audio_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    samples /= np.max(np.abs(samples)) + 1e-9

    # 20 ms frame energy
    frame = int(0.02 * sr)
    energy = np.convolve(np.abs(samples), np.ones(frame) / frame, "same")

    def find_local_energy(time_s, search_window=0.35):
        """Find actual onset in ±window seconds."""
        center = int(time_s * sr)
        half = int(search_window * sr)
        left = max(0, center - half)
        right = min(len(energy), center + half)
        local = energy[left:right]
        thresh = 0.25 * np.max(local)
        idx = np.where(local > thresh)[0]
        if len(idx) == 0:
            return time_s
        return (left + idx[0]) / sr

    refined = []
    for seg in segments:
        # word-level anchors within segment
        words_in_seg = [
            w for w in all_words if w["start"] >= seg["start"] and w["end"] <= seg["end"]
        ]
        if words_in_seg:
            start_time = words_in_seg[0]["start"]
            end_time = words_in_seg[-1]["end"]
        else:
            start_time, end_time = seg["start"], seg["end"]

        # use energy to refine onset and offset
        refined_start = find_local_energy(start_time, 0.25)
        refined_end = find_local_energy(end_time, 0.25)

        seg["start"] = max(0, refined_start - margin_ms / 1000)
        seg["end"] = min(len(audio) / 1000, refined_end + margin_ms / 1000)
        refined.append(seg)

    print("[SYNC] Hybrid alignment complete (±50 ms).")
    return refined

def correct_early_whisper_segments(segments, audio_path, max_delay_s=1.0, silence_thresh_offset=-35):
    """
    Detects when Whisper subtitles start too early and shifts them to the actual speech onset.
    """
    from pydub import AudioSegment, silence
    import numpy as np

    audio = AudioSegment.from_file(audio_path)
    silence_thresh = audio.dBFS + silence_thresh_offset

    # detect non-silent regions
    nonsilent = silence.detect_nonsilent(audio, silence_thresh=silence_thresh, min_silence_len=150)
    nonsilent = [(s/1000, e/1000) for s, e in nonsilent]

    def nearest_active_start(t):
        for s, e in nonsilent:
            if t < s:
                return s
            if s <= t <= e:
                return t
        return t

    corrected = []
    for seg in segments:
        s, e = seg["start"], seg["end"]
        new_start = nearest_active_start(s)
        if 0 < (new_start - s) < max_delay_s:
            seg["start"] = new_start
        if seg["end"] <= seg["start"]:
            seg["end"] = seg["start"] + 0.1
        corrected.append(seg)

    print(f"[FIX] Early Whisper timing corrected (shifted up to {max_delay_s}s)")
    return corrected

def collapse_repeated_runs_in_text(text: str) -> str:

    if not text:
        return text
    toks = text.split()
    out = []
    prev_norm = None
    for t in toks:
        norm = _normalize_token_for_compare(t)
        if norm == prev_norm:
            # skip repeated token
            continue
        out.append(t)
        prev_norm = norm
    return " ".join(out)

def _remove_boundary_duplicates(segments):

    if not segments:
        return segments
    segments_sorted = sorted(segments, key=lambda s: s['start'])
    for i in range(len(segments_sorted) - 1):
        a_tokens = segments_sorted[i]['text'].split()
        b_tokens = segments_sorted[i + 1]['text'].split()
        if not a_tokens or not b_tokens:
            continue
        max_k = min(3, len(a_tokens), len(b_tokens))
        # check longer overlaps first
        for k in range(max_k, 0, -1):
            end_a = a_tokens[-k:]
            start_b = b_tokens[:k]
            if [_normalize_token_for_compare(x) for x in end_a] == [_normalize_token_for_compare(x) for x in start_b]:
                # remove the overlapping tokens from the start of b
                segments_sorted[i + 1]['text'] = " ".join(b_tokens[k:]).strip()
                break
    return segments_sorted

def tidy_translated_segments(segments):

    if not segments:
        return segments
    # collapse repeated runs inside each segment
    for seg in segments:
        seg['text'] = collapse_repeated_runs_in_text(seg.get('text','')).strip()

    # remove boundary duplicates (end-of-seg == start-of-next)
    segments = _remove_boundary_duplicates(segments)

    # final cleanup: trim whitespace
    for seg in segments:
        seg['text'] = seg['text'].strip()
    return segments

# ----------------------------- Burn-in -----------------------------
def burn_subtitles_and_audio_to_video(original_video, subtitle_file, audio_file, output_video, tgt_lang_nllb="eng_Latn"):

    if subtitle_file.lower().endswith(".srt"):
        ass_file = subtitle_file.replace(".srt", ".ass")
        convert_srt_to_ass(subtitle_file, ass_file, tgt_lang_nllb)
    else:
        ass_file = subtitle_file

    font = get_font_for_lang(tgt_lang_nllb)
    vf_filter = f"ass={ass_file}"

    print(f"Burning subtitles with font '{font}' from {ass_file} ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video, "-i", audio_file,
        "-vf", vf_filter,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"Output video created with font '{font}': {output_video}")
def generate_video_with_tts_audio(original_video, tts_audio_file, tgt_lang_nllb, subtitle_file=None, output_video=None):
    if not output_video:
        base = safe_filename(os.path.splitext(os.path.basename(original_video))[0])
        output_video = f"{base}_tts_only.mp4"

    cmd = ["ffmpeg", "-y", "-i", original_video, "-i", tts_audio_file, "-map", "0:v:0", "-map", "1:a:0"]

    # Explicitly skip subtitles unless provided
    if subtitle_file is not None:
        ass_file = subtitle_file.replace(".srt", ".ass")
        convert_srt_to_ass(subtitle_file, ass_file, tgt_lang_nllb)
        cmd.extend(["-vf", f"ass={ass_file}"])

    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-b:a", "192k", "-shortest", output_video
    ])

    print(f"Generating TTS-only video: {output_video} ...")
    subprocess.run(cmd, check=True)
    print(f"TTS-only video created: {output_video}")
    return output_video

def split_segments_on_sentence_end(segments, lang_nllb="eng_Latn"):

    SENTENCE_END_PUNCT = {
        "_Latn": r"[.!?:]",
        "_Deva": r"[।!?:]",
        "_Arab": r"[؟!۔:]", 
        "_Beng": r"[।!?:]",
        "_Cyrl": r"[.!?:]",
        "_Hans": r"[。！？:]", 
        "_Hant": r"[。！？:]", 
        "_Jpan": r"[。！？:]", 
        "_Hang": r"[.?!:]", 
        "_default": r"[.!?।؟۔。！？።:]", 
    }

    script_suffix = next((s for s in SENTENCE_END_PUNCT if lang_nllb.endswith(s)), "_default")
    punct_pattern = SENTENCE_END_PUNCT.get(script_suffix, SENTENCE_END_PUNCT["_default"])

    # Split after punctuation followed by space and a letter (any language)
    split_regex = rf"(?<={punct_pattern})(?=\s*[\p{{L}}])"

    new_segments = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Mask URLs / times
        safe = text.replace("http://", "http⨯//").replace("https://", "https⨯//")
        safe = regex.sub(r'(?<!\d):(?=\d{1,2}\b)', '⨯:', safe)

        parts = regex.split(split_regex, safe)
        parts = [p.replace("⨯:", ":").replace("http⨯//", "http://").replace("https⨯//", "https://").strip()
                 for p in parts if p.strip()]

        if len(parts) == 1:
            new_segments.append(seg)
            continue

        # redistribute duration proportionally by text length
        total_chars = sum(len(p) for p in parts)
        duration = seg["end"] - seg["start"]
        start_t = seg["start"]

        for part in parts:
            ratio = len(part) / total_chars if total_chars else 1 / len(parts)
            seg_dur = duration * ratio
            end_t = start_t + seg_dur
            if end_t <= start_t:
                end_t = start_t + 0.05
            new_segments.append({
                "start": round(start_t, 3),
                "end": round(end_t, 3),
                "text": part
            })
            start_t = end_t

    # fix tiny overlaps
    for i in range(len(new_segments) - 1):
        if new_segments[i]["end"] > new_segments[i + 1]["start"]:
            new_segments[i]["end"] = max(
                new_segments[i]["start"] + 0.05,
                new_segments[i + 1]["start"] - 0.05
            )

    print(f"[SENTENCE SPLIT] {len(segments)} → {len(new_segments)} (colon-aware, {lang_nllb})")
    return new_segments

def correct_early_whisper_segments(segments, audio_path, max_delay_s=1.0, silence_thresh_offset=-35):
    """
    Detects when Whisper subtitles start too early and shifts them to the actual speech onset.
    """
    from pydub import AudioSegment, silence
    import numpy as np

    audio = AudioSegment.from_file(audio_path)
    silence_thresh = audio.dBFS + silence_thresh_offset

    # detect non-silent regions
    nonsilent = silence.detect_nonsilent(audio, silence_thresh=silence_thresh, min_silence_len=150)
    nonsilent = [(s/1000, e/1000) for s, e in nonsilent]

    def nearest_active_start(t):
        for s, e in nonsilent:
            if t < s:
                return s
            if s <= t <= e:
                return t
        return t

    corrected = []
    for seg in segments:
        s, e = seg["start"], seg["end"]
        new_start = nearest_active_start(s)
        if 0 < (new_start - s) < max_delay_s:
            seg["start"] = new_start
        if seg["end"] <= seg["start"]:
            seg["end"] = seg["start"] + 0.1
        corrected.append(seg)

    print(f"[FIX] Early Whisper timing corrected (shifted up to {max_delay_s}s)")
    return corrected

def main():
    parser = argparse.ArgumentParser(
        description="Whisper → NLLB → MMS TTS → Burn-in (punctuation-preserving Whisper SRT)"
    )
    parser.add_argument("input", help="Input video/audio path")
    parser.add_argument(
        "--language", "-l", default=None,
        help="Force transcription language ISO-639-1 (e.g., hi, mr, es)"
    )
    parser.add_argument(
        "--target_langs", "-t",
        nargs="+",
        default=["hin_Deva"],
        help="One or more target languages (NLLB codes)"
    )
    args = parser.parse_args()

    input_path = args.input
    base = safe_filename(os.path.splitext(os.path.basename(input_path))[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        # === STEP 1: Extract & Transcribe ===
        audio_path = extract_audio_if_video(input_path, tmpdir)

        segments, detected_lang, all_words = transcribe_in_chunks(
            audio_path, language=args.language
        )
        segments = correct_early_whisper_segments(segments, audio_path)

        src_lang_nllb = to_nllb_code(detected_lang or "en")
        print(f"[SRC] Using source language: {detected_lang} → {src_lang_nllb}")

        # === STEP 2: Restore punctuation in Whisper transcript ===
        print("[INFO] Restoring punctuation in Whisper transcript...")
        try:
            full_text = " ".join([seg["text"] for seg in segments])
            punctuated_text = restore_punctuation(full_text)

            sentences = nltk.sent_tokenize(punctuated_text)
            if len(sentences) == len(segments):
                for i, s in enumerate(sentences):
                    segments[i]["text"] = s.strip()
            else:
                # proportional redistribution by duration
                words = punctuated_text.split()
                total_tokens = len(words)
                durations = [max(0.1, seg["end"] - seg["start"]) for seg in segments]
                total_duration = sum(durations)
                counts = [max(1, int(round((d / total_duration) * total_tokens))) for d in durations]
                diff = total_tokens - sum(counts)
                counts[-1] += diff
                idx = 0
                for i, c in enumerate(counts):
                    chunk = words[idx: idx + c]
                    segments[i]["text"] = " ".join(chunk).strip()
                    idx += c

            print("[INFO] ✅ Whisper transcript punctuation restored successfully.")
        except Exception as e:
            print(f"[WARN] Punctuation restoration failed ({e}), keeping Whisper's raw output.")

        for seg in segments:
            segments = sentence_case_contextual(segments)
        # Save Whisper SRT
        out_srt_original = f"{base}_{src_lang_nllb}_whisper.srt"
        with open(out_srt_original, "w", encoding="utf-8") as f:
            f.write(segments_to_srt(segments))
        print(f"[SAVE] Whisper SRT → {out_srt_original}")

        # === STEP 3: Translation for ALL target languages ===
        for tgt_lang in args.target_langs:
            tgt_lang_nllb = to_nllb_code(tgt_lang)
            out_srt_translated = f"{base}_{tgt_lang_nllb}_translated.srt"

            # === 🧠 Skip translation if both source and target are English ===
            if src_lang_nllb == "eng_Latn" and tgt_lang_nllb == "eng_Latn":
                print(f"[SKIP] Source and target are both English → using Whisper SRT directly.")
                out_srt_translated = out_srt_original
            else:
                print(f"\n[TRANSLATE] Translating full Whisper SRT → {tgt_lang_nllb}")

                # --- Determine translation strategy ---
                if not src_lang_nllb.endswith("_Latn"):  # Non-English source
                    print(f"[TRANSLATE] Source '{src_lang_nllb}' is non-English → using batch translation.")
                    
                    subs = pysubs2.load(out_srt_original, encoding="utf-8")
                    segments = [
                        {"start": e.start / 1000.0, "end": e.end / 1000.0, "text": e.text.strip()}
                        for e in subs if e.text.strip()
                    ]
                    translated_segments = translate_segments_nllb_batched(
                        segments, src_lang_nllb, tgt_lang_nllb
                    )
                    with open(out_srt_translated, "w", encoding="utf-8") as f:
                        f.write(segments_to_srt(translated_segments))
                else:
                    print(f"[TRANSLATE] Source '{src_lang_nllb}' is English/Latin → using segment translation.")
                    translate_srt_file_segment(out_srt_original, out_srt_translated, src_lang_nllb, tgt_lang_nllb)

            print(f"[SAVE] Translation completed → {out_srt_translated}")

            # === NEW: Sync the translated SRT to actual audio waveform (avoid duplicate SRTs) ===
            try:
                print(f"[SYNC] Re-aligning translated SRT to audio waveform: {out_srt_translated} ...")
                # overwrite the translated file with the synced timings
                sync_embedded_subs_to_audio(
                    video_path=input_path,
                    input_srt=out_srt_translated,
                    output_srt=out_srt_translated
                )
                print(f"[SYNC] ✅ Translated SRT re-aligned with audio → {out_srt_translated}")
            except Exception as e:
                print(f"[WARN] Could not sync translated subtitles ({e}). Proceeding without sync.")

            # === POST-TRANSLATION SENTENCE SPLIT ===
            print("\n[POST] ✂️ Splitting translated SRT at sentence boundaries...")

            try:
                subs = pysubs2.load(out_srt_translated, encoding="utf-8")
                segments_translated = [
                    {"start": e.start / 1000.0, "end": e.end / 1000.0, "text": e.text.strip()}
                    for e in subs
                ]
            
                # Apply sentence-end splitting (only on translated file)
                segments_split = split_segments_on_sentence_end(segments_translated, lang_nllb=tgt_lang_nllb)

                # 🔹 Merge small translated segments for smoother readability
                segments_split = merge_short_segments_duration_aware(segments_split, max_words=6, max_gap_s=0.6)

                # Write back the updated SRT
                with open(out_srt_translated, "w", encoding="utf-8") as f:
                    f.write(segments_to_srt(segments_split))

                print(f"[POSTPROCESS] ✅ Sentence-end split applied successfully → {out_srt_translated}")

            except Exception as e:
                print(f"[WARN] ⚠️ Sentence split postprocess failed: {e}")

            # === STEP 4: Generate subtitled video ===
            ass_file = out_srt_translated.replace(".srt", ".ass")
            
            convert_srt_to_ass(out_srt_translated, ass_file, tgt_lang_nllb)
            out_video_subs = f"{base}_{tgt_lang_nllb}_subs.mp4"

            subprocess.run([
                "ffmpeg", "-y", "-i", input_path, "-vf", f"ass={ass_file}",
                "-c:v", "libx264", "-preset", "fast", "-c:a", "copy", out_video_subs
            ], check=True)
            print(f"[SAVE] Subtitled video → {out_video_subs}")

            # === STEP 5: Generate TTS dubbed version ===
            print("[TTS] Generating dubbed audio...")
            out_video_tts = f"{base}_{tgt_lang_nllb}_dub.mp4"
            out_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name

            tts_from_srt_global_fit_refined(
                out_srt_translated, tgt_lang_nllb,
                video_path=input_path,
                output_audio=out_audio_file
            )

            subprocess.run([
                "ffmpeg", "-y", "-i", input_path, "-i", out_audio_file,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out_video_tts
            ], check=True)
            print(f"[SAVE] Dubbed video → {out_video_tts}")

        # === Final summary ===
        print("\n=== FINAL OUTPUTS ===")
        print(f"1. Whisper SRT        : {out_srt_original}")
        print(f"2. Translated SRTs     : {[f'{base}_{to_nllb_code(t)}_translated.srt' for t in args.target_langs]}")
        print(f"3. Subtitled Videos    : {[f'{base}_{to_nllb_code(t)}_subs.mp4' for t in args.target_langs]}")
        print(f"4. Dubbed Videos       : {[f'{base}_{to_nllb_code(t)}_dub.mp4' for t in args.target_langs]}")

if __name__ == "__main__":
    main()

