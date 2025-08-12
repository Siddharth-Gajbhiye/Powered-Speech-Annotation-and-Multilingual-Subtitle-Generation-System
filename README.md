# Powered-Speech-Annotation-and-Multilingual-Subtitle-Generation-System
An AI-powered system that transcribes speech, translates it into multiple languages, and generates time-synced subtitles and voice output, enabling seamless multilingual communication and accessibility for meetings, lectures, and video content.

Make sure dependencies are installed and also make sure ffmpeg is installed and available in PATH :

pip install openai-whisper edge-tts pydub tqdm transformers sentencepiece sacremoses

You can run this script from your terminal (Command Prompt or PowerShell on Windows) with a video/audio file path as input.

Basic usage (English → Hindi): python subtitle.py "input_video.mp4"

Multiple translations (English → Hindi, Spanish, French): python subtitle.py "input_video.mp4" --target_lang hi,es,fr

Force a specific Whisper model size (faster ones: base, small, medium; most accurate: large): python subtitle.py "input_video.mp4" --model medium

Force transcription language (if Whisper is unsure): python subtitle.py "input_video.mp4" --language en
