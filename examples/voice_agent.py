#!/usr/bin/env python3
"""
Local Voice Agent Example
=========================

A complete example showing how to use the local STT/TTS plugins
with LiveKit Agents. Supports both cloud and local pipelines.

Usage:
    # Start with local pipeline
    USE_LOCAL=true python voice_agent.py dev

    # Start with cloud pipeline (requires API keys)
    python voice_agent.py dev

Environment Variables:
    USE_LOCAL           - Set to "true" for local pipeline
    WHISPER_MODEL       - FasterWhisper model size (default: "medium")
    WHISPER_DEVICE      - "cuda" or "cpu" (default: "cuda")
    PIPER_MODEL_PATH    - Path to Piper .onnx model
    OLLAMA_MODEL        - Ollama model name (default: "llama3.1:8b")
    OLLAMA_BASE_URL     - Ollama API URL (default: "http://localhost:11434/v1")

For cloud pipeline:
    DEEPGRAM_API_KEY    - Deepgram API key
    OPENAI_API_KEY      - OpenAI API key
    CARTESIA_API_KEY    - Cartesia API key
"""

from __future__ import annotations

import logging
import os
import sys
import time

# Add parent directory to path for local development (must be before other imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime
from dotenv import load_dotenv


# Load environment variables (try .env.local first, then .env)
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env.local"))
load_dotenv(os.path.join(_script_dir, ".env"))

# LiveKit imports - plugins must be imported on main thread
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, llm

from livekit.plugins import silero
from livekit.plugins import openai as lk_openai

# Local plugins
from local_livekit_plugins import FasterWhisperSTT, VieNeuTTS, PiperTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("voice-agent")

# =============================================================================
# Configuration
# =============================================================================

USE_LOCAL = os.getenv("USE_LOCAL", "false").lower() == "true"

# Local pipeline settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
STT_PROVIDER = os.getenv("STT_PROVIDER", "faster_whisper").lower()
VLLM_ASR_MODEL = os.getenv("VLLM_ASR_MODEL", "qwen3asr")
VLLM_ASR_BASE_URL = os.getenv("VLLM_ASR_BASE_URL", "http://10.148.180.105:1670/v1")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "vieneu").lower()
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"

VIENEU_API_BASE = os.getenv("VIENEU_API_BASE", "http://localhost:23333/v1")
VIENEU_MODEL_NAME = os.getenv("VIENEU_MODEL_NAME", "pnnbao-ump/VieNeu-TTS")
VIENEU_VOICE_ID = os.getenv("VIENEU_VOICE_ID", "") # e.g. "nu_m_001"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LANGUAGE = os.getenv("LANGUAGE", "vi").lower() # "vi" or "en"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen3_vl")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://10.148.180.105:22003/v1")


# =============================================================================
# Agent Definition
# =============================================================================

from tools import AssistantFnc


class VoiceAssistant(Agent):
    """A simple voice assistant that responds to user queries."""

    def __init__(self, tools: list[llm.FunctionTool] | None = None) -> None:
        now = datetime.now()
        days = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        day_str = days[now.weekday()]
        current_time_str = f"{day_str}, {now.strftime('%d/%m/%Y %H:%M:%S')}"

        from prompts import VIETNAMESE_RESPONSE_SYSTEM_PROMPT, ENGLISH_RESPONSE_SYSTEM_PROMPT
        
        # Select prompt based on desired language
        if LANGUAGE == "en":
             base_prompt = ENGLISH_RESPONSE_SYSTEM_PROMPT
             # For dynamic instructions in English
             now_msg = f"Current time is: {now.strftime('%A, %d/%m/%Y %H:%M:%S')}."
        else:
             base_prompt = VIETNAMESE_RESPONSE_SYSTEM_PROMPT
             now_msg = f"Hiện tại là: {current_time_str}."

        instructions=f"{now_msg}\n\n{base_prompt}"
        
        super().__init__(
            instructions=instructions,
            tools=tools,
        )





# =============================================================================
# Pipeline Factories
# =============================================================================

def create_local_session() -> AgentSession:
    """
    Create an AgentSession using fully local STT/LLM/TTS.

    Requirements:
        - Ollama running with desired model
        - Piper voice model downloaded
        - CUDA toolkit (optional, for GPU acceleration)
    """
    logger.info("=" * 60)
    logger.info("STARTING LOCAL PIPELINE")
    logger.info("=" * 60)
    if STT_PROVIDER == "vllm":
        logger.info(f"  STT: vLLM ({VLLM_ASR_MODEL})")
    else:
        logger.info(f"  STT: FasterWhisper ({WHISPER_MODEL} on {WHISPER_DEVICE})")
    if LLM_PROVIDER == "vllm":
        logger.info(f"  LLM: vLLM ({VLLM_MODEL})")
    else:
        logger.info(f"  LLM: Ollama ({OLLAMA_MODEL})")
    
    if TTS_PROVIDER == "piper":
        logger.info(f"  TTS: Piper (CUDA: {PIPER_USE_CUDA})")
        if not PIPER_MODEL_PATH:
            raise ValueError(
                "PIPER_MODEL_PATH not set. Download a voice model from:\n"
                "https://huggingface.co/rhasspy/piper-voices"
            )
    else:
        logger.info(f"  TTS: VieNeu ({VIENEU_MODEL_NAME} at {VIENEU_API_BASE})")
    
    logger.info("=" * 60)

    return AgentSession(
        stt=lk_openai.STT(
            model=VLLM_ASR_MODEL,
            base_url=VLLM_ASR_BASE_URL,
            detect_language=True,
        ) if STT_PROVIDER == "vllm" else FasterWhisperSTT(
            model_size=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
            language=None,  # Enable auto-detection
        ),
        llm=lk_openai.LLM(
            model=VLLM_MODEL,
            base_url=VLLM_BASE_URL,
        ) if LLM_PROVIDER == "vllm" else lk_openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
        ),
        tts=PiperTTS(
            model_path=PIPER_MODEL_PATH,
            use_cuda=PIPER_USE_CUDA,
        ) if TTS_PROVIDER == "piper" else VieNeuTTS(
            api_base=VIENEU_API_BASE,
            model_name=VIENEU_MODEL_NAME,
            voice_id=VIENEU_VOICE_ID if VIENEU_VOICE_ID else None,
        ),
        vad=silero.VAD.load(),
    )


def create_cloud_session() -> AgentSession:
    """
    Create an AgentSession using cloud STT/LLM/TTS services.

    Requirements:
        - DEEPGRAM_API_KEY
        - OPENAI_API_KEY
        - CARTESIA_API_KEY
    """
    logger.info("=" * 60)
    logger.info("STARTING CLOUD PIPELINE")
    logger.info("=" * 60)
    logger.info("  STT: Deepgram Nova-2")
    logger.info("  LLM: OpenAI GPT-4o-mini")
    logger.info("  TTS: Cartesia Sonic")
    logger.info("=" * 60)

    return AgentSession(
        stt="deepgram/nova-2",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic",
        vad=silero.VAD.load(),
    )


# =============================================================================
# Agent Entrypoint
# =============================================================================

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for the voice agent."""

    logger.info(f"Joining room: {ctx.room.name}")
    await ctx.connect()

    # Create session based on configuration
    session = create_local_session() if USE_LOCAL else create_cloud_session()

    # ==========================================================================
    # Round-trip latency tracking
    # ==========================================================================
    # Measures time from user speech transcribed to agent starting to speak.
    # This captures: LLM processing + TTS first byte (STT already done).

    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

    # ==========================================================================

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(), # Removed tools for now due to vLLM server limitations
        room_input_options=RoomInputOptions(),
    )




    # Send initial greeting
    await session.generate_reply(
        instructions="Greet the user and let them know you're ready to help."
    )

    logger.info("Agent ready - listening for speech...")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
