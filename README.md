

Run this script from your terminal (Command Prompt or PowerShell on Windows) with a video/audio file path as input.

Basic usage (English → Hindi): python subtitle.py "input_video.mp4"

Multiple translations (English → Hindi, Spanish, French): python subtitle.py "input_video.mp4" --target_lang hi,es,fr

Force a specific Whisper model size (faster ones: base, small, medium; most accurate: large): python subtitle.py "input_video.mp4" --model medium

Force transcription language (if Whisper is unsure): python subtitle.py "input_video.mp4" --language en
