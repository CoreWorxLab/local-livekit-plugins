"""
Piper TTS Plugin for LiveKit Agents
====================================

Local text-to-speech using Piper with optional GPU acceleration.
No cloud APIs required - runs entirely on your hardware.

Features:
    - Fast CPU-based synthesis
    - Optional GPU acceleration (CUDA version dependent)
    - Multiple voice models available
    - Configurable speed, volume, and voice characteristics
    - DEBUG-level latency logging for benchmarking

Requirements:
    - piper-tts >= 1.2.0
    - Voice model files (.onnx + .onnx.json)

Note on GPU Acceleration:
    Piper CAN run on GPU but has specific CUDA/onnxruntime version requirements.
    If you're on CUDA 12+, you may need to use CPU mode or containerize
    with a compatible CUDA version. CPU mode is often faster for short utterances anyway.
    See: https://github.com/rhasspy/piper/discussions/544

Example:
    >>> from local_livekit_plugins import PiperTTS
    >>>
    >>> tts = PiperTTS(
    ...     model_path="/path/to/en_US-ryan-high.onnx",
    ...     use_cuda=False,  # CPU mode (most compatible)
    ...     speed=1.0
    ... )
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
import wave
from typing import TYPE_CHECKING

from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

__all__ = ["PiperTTS"]

logger = logging.getLogger(__name__)


class _PiperChunkedStream(tts.ChunkedStream):
    """
    Internal ChunkedStream implementation for Piper TTS.

    Handles the async bridge between LiveKit's streaming interface
    and Piper's synchronous synthesis.
    """

    def __init__(
        self,
        *,
        tts_plugin: PiperTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._piper_tts = tts_plugin

    async def _run(self, emitter: AudioEmitter) -> None:
        """Synthesize audio and emit it to LiveKit."""
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._piper_tts.sample_rate,
            num_channels=self._piper_tts.num_channels,
            mime_type="audio/pcm",
        )

        # Run blocking synthesis in thread pool
        start_time = time.perf_counter()
        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._synthesize_blocking,
            self._input_text
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(f"TTS latency: {elapsed_ms:.0f}ms for {len(self._input_text)} chars")

        emitter.push(audio_bytes)

    def _synthesize_blocking(self, text: str) -> bytes:
        """
        Blocking synthesis operation.

        Runs in a thread pool to avoid blocking the async event loop.
        """
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=1.0 / self._piper_tts.speed,
            noise_scale=self._piper_tts.noise_scale,
            noise_w_scale=self._piper_tts.noise_w,
            volume=self._piper_tts.volume,
        )

        # Synthesize to WAV in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            self._piper_tts.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=syn_config,
                set_wav_format=True
            )

        # Extract raw PCM frames from WAV
        wav_io.seek(0)
        with wave.open(wav_io, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())

        return frames


class PiperTTS(tts.TTS):
    """
    LiveKit TTS plugin using Piper for local speech synthesis.

    This plugin integrates the Piper TTS engine with LiveKit's Agents
    framework, enabling fully local text-to-speech without cloud dependencies.

    Args:
        model_path: Path to the .onnx voice model file.
            Download voices from: https://huggingface.co/rhasspy/piper-voices
        use_cuda: Enable GPU acceleration. Note: Has CUDA version constraints.
            Set to False for maximum compatibility.
        speed: Speech rate multiplier. 1.0 = normal, 1.5 = faster, 0.8 = slower
        volume: Volume level. 1.0 = normal
        noise_scale: Phoneme noise scale for variation (0.0-1.0). Default: 0.667
        noise_w: Phoneme width noise scale (0.0-1.0). Default: 0.8

    Performance Notes:
        - CPU is often as fast as GPU for short utterances
        - Enable DEBUG logging to see latency metrics
        - Quality: Good, clearly synthetic but natural sounding

    Popular Voice Models:
        - en_US-ryan-high: Male US English (recommended)
        - en_US-amy-medium: Female US English
        - en_GB-alan-medium: Male British English

    Example:
        >>> from local_livekit_plugins import PiperTTS
        >>> from livekit.agents import AgentSession
        >>>
        >>> tts = PiperTTS(
        ...     model_path="/models/piper/en_US-ryan-high.onnx",
        ...     use_cuda=False,
        ...     speed=1.0
        ... )
        >>>
        >>> session = AgentSession(stt=..., llm=..., tts=tts)
    """

    def __init__(
        self,
        model_path: str,
        use_cuda: bool = False,
        speed: float = 1.0,
        volume: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        from piper.voice import PiperVoice

        # Piper models typically output 22050 Hz mono audio
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1
        )

        self.speed = speed
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w = noise_w

        logger.info(f"Loading Piper voice: {model_path} (CUDA: {use_cuda})")
        self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        logger.info(f"Piper ready - speed={speed}, volume={volume}")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            conn_options: LiveKit connection options (optional)

        Returns:
            ChunkedStream that yields audio data
        """
        if conn_options is None:
            conn_options = APIConnectOptions()

        logger.debug(f"Synthesizing ({len(text)} chars): {text[:50]}...")

        return _PiperChunkedStream(
            tts_plugin=self,
            input_text=text,
            conn_options=conn_options,
        )
