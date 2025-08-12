# Powered-Speech-Annotation-and-Multilingual-Subtitle-Generation-System
An AI-powered system that transcribes speech, translates it into multiple languages, and generates time-synced subtitles and voice output, enabling seamless multilingual communication and accessibility for meetings, lectures, and video content.

## Features
- Supports multiple target languages
- Outputs `.srt` subtitle files
- Outputs synced `.mp3` audio files
- Works with audio & video files

Run this script from your terminal (Command Prompt or PowerShell on Windows) with a video/audio file path as input.

# Basic usage [English to Hindi]
- python subtitle.py "input_video.mp4"

# Multiple translations [English to Hindi, Spanish, French]
- python subtitle.py "input_video.mp4" --target_lang hi,es,fr

# Force a specific Whisper model size [faster ones: base, small, medium; most accurate: large]
- python subtitle.py "input_video.mp4" --model medium

# Force transcription language [if Whisper is unsure]
- python subtitle.py "input_video.mp4" --target_lang hi

