"""
VieNeu-TTS Plugin for LiveKit Agents
====================================

High-quality Vietnamese text-to-speech with instant voice cloning.
Integrates the VieNeu-TTS remote client with LiveKit's streaming architecture.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Dict

from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

# Add VieNeu-TTS source directory to path if it exists locally
_project_root = Path(__file__).parent.parent.parent
_vieneu_src = _project_root / "models" / "VieNeu-TTS" / "src"
if _vieneu_src.exists() and str(_vieneu_src) not in sys.path:
    # Insert at the beginning to prefer the version in the models folder
    sys.path.insert(0, str(_vieneu_src))

try:
    from vieneu.remote import RemoteVieNeuTTS
except ImportError:
    # This might happen during initial setup if paths aren't fully ready
    RemoteVieNeuTTS = None

logger = logging.getLogger(__name__)

class VieNeuTTS(tts.TTS):
    """
    LiveKit TTS plugin using VieNeu-TTS for high-quality Vietnamese speech.
    
    Args:
        api_base: URL of the VieNeu-TTS server (e.g., LMDeploy server).
        model_name: Name of the model registered on the server.
        voice_id: Optional ID of a preset voice. If None, uses the model's default.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:23333/v1",
        model_name: str = "pnnbao-ump/VieNeu-TTS",
        voice_id: Optional[str] = None,
    ) -> None:
        if RemoteVieNeuTTS is None:
            raise ImportError(
                "Could not find 'vieneu' package. Ensure 'models/VieNeu-TTS/src' is in your path."
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000, # VieNeu standard sample rate
            num_channels=1,
        )
        
        self._api_base = api_base
        self._model_name = model_name
        self._voice_id = voice_id
        
        logger.info(f"Connecting to VieNeu-TTS server at {api_base} (model: {model_name})")
        
        # Initialize client
        self._client = RemoteVieNeuTTS(
            api_base=api_base,
            model_name=model_name
        )
        
        # Resolve voice data if specified
        self._voice_data = None
        if voice_id:
            try:
                # VieNeu preset voices are available via this call
                self._voice_data = self._client.get_preset_voice(voice_id)
                logger.info(f"VieNeuTTS initialized with voice: {voice_id}")
            except Exception as e:
                logger.error(f"Failed to resolve VieNeu voice '{voice_id}': {e}")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> tts.ChunkedStream:
        """
        Synthesize speech from Vietnamese text.
        """
        if conn_options is None:
            conn_options = APIConnectOptions()

        logger.debug(f"Synthesizing ({len(text)} chars) via VieNeu: {text[:50]}...")

        return _VieNeuChunkedStream(
            tts_plugin=self,
            input_text=text,
            conn_options=conn_options,
        )

class _VieNeuChunkedStream(tts.ChunkedStream):
    """
    Manages the streaming response from the VieNeu-TTS server.
    """
    
    def __init__(
        self,
        *,
        tts_plugin: VieNeuTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._plugin = tts_plugin

    async def _run(self, emitter: AudioEmitter) -> None:
        """
        Execution loop for the synthesis stream.
        """
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._plugin.sample_rate,
            num_channels=self._plugin.num_channels,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_running_loop()
        
        try:
            # We run the blocking stream generator in a thread pool
            def process_synthesis():
                # Get the stream generator from the client
                gen = self._plugin._client.infer_stream(
                    self._input_text,
                    voice=self._plugin._voice_data
                )
                
                for audio_chunk in gen:
                    if audio_chunk is None or len(audio_chunk) == 0:
                        continue
                        
                    # VieNeu returns float32 numpy arrays [-1.0, 1.0]
                    # We convert to int16 PCM for LiveKit
                    pcm_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
                    
                    # Push directly to the emitter
                    emitter.push(pcm_chunk)
                    
            # Run the synthesis and emission in a background thread
            await loop.run_in_executor(None, process_synthesis)
            
        except Exception as e:
            logger.exception(f"Unexpected error during VieNeu synthesis: {e}")
        finally:
            # Signal the end of the stream
            emitter.flush()
