"""
LiveKit Local Plugins
=====================

Custom STT and TTS plugins for running LiveKit Agents with fully local
speech processing - no cloud APIs required.

Plugins:
    FasterWhisperSTT: GPU-accelerated speech-to-text using faster-whisper
    PiperTTS: Fast local text-to-speech using Piper

Example:
    >>> from local_livekit_plugins import FasterWhisperSTT, VieNeuTTS
    >>>
    >>> stt = FasterWhisperSTT(model_size="medium", device="cuda")
    >>> tts = VieNeuTTS(api_base="http://localhost:23333/v1")

Repository: https://github.com/CoreWorxLab/local-livekit-plugins
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Corey MacPherson"

from .faster_whisper_stt import FasterWhisperSTT
from .piper_tts import PiperTTS
from .vieneu_tts import VieNeuTTS

__all__ = ["FasterWhisperSTT", "PiperTTS", "VieNeuTTS", "__version__"]
