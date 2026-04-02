from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(".env")

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, llm, RunContext
from livekit.plugins import silero
from livekit.plugins import openai as lk_openai
from livekit.agents.llm import function_tool
from livekit.agents.llm.tool_context import StopResponse
from livekit.plugins import noise_cancellation
# livekit-plugins-noise-cancellation 
from livekit.agents import mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("voice-agent")


from scripts.config import (
    USE_LOCAL, WHISPER_MODEL, WHISPER_DEVICE, STT_PROVIDER,
    VLLM_ASR_MODEL, VLLM_ASR_BASE_URL, TTS_PROVIDER,
    PIPER_MODEL_PATH, PIPER_USE_CUDA, VIENEU_API_BASE,
    VIENEU_MODEL_NAME, VIENEU_VOICE_ID, LLM_PROVIDER,
    LANGUAGE, OLLAMA_MODEL, OLLAMA_BASE_URL,
    VLLM_MODEL, VLLM_BASE_URL
)

from scripts.tools import AssistantFnc
from scripts.prompts import VIETNAMESE_RESPONSE_SYSTEM_PROMPT, ENGLISH_RESPONSE_SYSTEM_PROMPT
from local_livekit_plugins import FasterWhisperSTT, VieNeuTTS, PiperTTS
from scripts.generic_agent import GenericAgent

class VoiceAssistant(GenericAgent):
    """A simple voice assistant that responds to user queries."""
    def __init__(self) -> None:
        # Select prompt based on desired language
        if LANGUAGE == "en":
             base_prompt = ENGLISH_RESPONSE_SYSTEM_PROMPT
        else:
             base_prompt = VIETNAMESE_RESPONSE_SYSTEM_PROMPT
        now_msg = ""

        instructions=f"{now_msg}\n\n{base_prompt}"
        
        super().__init__(
            instructions=instructions,
        )

        self.airbnbs = {
            "san francisco": [
                {
                    "id": "sf001",
                    "name": "Loft ấm cúng tại trung tâm",
                    "address": "123 Market Street, San Francisco, CA",
                    "price": 150,
                    "amenities": ["WiFi", "Nhà bếp", "Không gian làm việc"],
                },
                {
                    "id": "sf002",
                    "name": "Nhà kiểu Victorian với tầm nhìn ra vịnh",
                    "address": "456 Castro Street, San Francisco, CA",
                    "price": 220,
                    "amenities": ["WiFi", "Chỗ đậu xe", "Máy giặt/Máy sấy", "Tầm nhìn ra vịnh"],
                },
                {
                    "id": "sf003",
                    "name": "Studio hiện đại gần Cầu Cổng Vàng",
                    "address": "789 Presidio Avenue, San Francisco, CA",
                    "price": 180,
                    "amenities": ["WiFi", "Nhà bếp", "Cho phép mang thú cưng"],
                },
            ],
            "new york": [
                {
                    "id": "ny001",
                    "name": "Căn hộ gạch nâu (Brownstone) tại Brooklyn",
                    "address": "321 Bedford Avenue, Brooklyn, NY",
                    "price": 175,
                    "amenities": ["WiFi", "Nhà bếp", "Lối vào sân sau"],
                },
                {
                    "id": "ny002",
                    "name": "Penthouse ngắm cảnh trời Manhattan",
                    "address": "555 Fifth Avenue, Manhattan, NY",
                    "price": 350,
                    "amenities": ["WiFi", "Phòng tập gym", "Lễ tân", "Tầm nhìn ra thành phố"],
                },
                {
                    "id": "ny003",
                    "name": "Loft nghệ thuật tại East Village",
                    "address": "88 Avenue A, Manhattan, NY",
                    "price": 195,
                    "amenities": ["WiFi", "Máy giặt", "Tường gạch trần"],
                },
            ],
            "los angeles": [
                {
                    "id": "la001",
                    "name": "Bungalow tại bãi biển Venice",
                    "address": "234 Ocean Front Walk, Venice, CA",
                    "price": 200,
                    "amenities": ["WiFi", "Lối ra bãi biển", "Sân trong"],
                },
                {
                    "id": "la002",
                    "name": "Biệt thự đồi Hollywood",
                    "address": "777 Mulholland Drive, Los Angeles, CA",
                    "price": 400,
                    "amenities": ["WiFi", "Hồ bơi", "Tầm nhìn ra thành phố", "Bồn tắm nước nóng"],
                },
            ],
        }

        # Track bookings
        self.bookings = []

    @function_tool
    async def search_airbnbs(self, context: RunContext, city: str) -> str:
        """Search for available Airbnbs in a city.

        Args:
            city: The city name to search for Airbnbs (e.g., 'San Francisco', 'New York', 'Los Angeles')
        """
        city_lower = city.lower()

        if city_lower not in self.airbnbs:
            return f"Sorry, I don't have any Airbnb listings for {city} at the moment. Available cities are: San Francisco, New York, and Los Angeles."

        listings = self.airbnbs[city_lower]
        result = f"Found {len(listings)} Airbnbs in {city}:\n\n"

        for listing in listings:
            result += f"• {listing['name']}\n"
            result += f"  Address: {listing['address']}\n"
            result += f"  Price: ${listing['price']} per night\n"
            result += f"  Amenities: {', '.join(listing['amenities'])}\n"
            result += f"  ID: {listing['id']}\n\n"

        return result

    @function_tool
    async def book_airbnb(self, context: RunContext, airbnb_id: str, guest_name: str, check_in_date: str, check_out_date: str) -> str:
        """Book an Airbnb.

        Args:
            airbnb_id: The ID of the Airbnb to book (e.g., 'sf001')
            guest_name: Name of the guest making the booking
            check_in_date: Check-in date (e.g., 'January 15, 2025')
            check_out_date: Check-out date (e.g., 'January 20, 2025')
        """
        # Find the Airbnb
        airbnb = None
        for city_listings in self.airbnbs.values():
            for listing in city_listings:
                if listing['id'] == airbnb_id:
                    airbnb = listing
                    break
            if airbnb:
                break

        if not airbnb:
            return f"Sorry, I couldn't find an Airbnb with ID {airbnb_id}. Please search for available listings first."

        # Create booking
        booking = {
            "confirmation_number": f"BK{len(self.bookings) + 1001}",
            "airbnb_name": airbnb['name'],
            "address": airbnb['address'],
            "guest_name": guest_name,
            "check_in": check_in_date,
            "check_out": check_out_date,
            "total_price": airbnb['price'],
        }

        self.bookings.append(booking)

        result = f"✓ Booking confirmed!\n\n"
        result += f"Confirmation Number: {booking['confirmation_number']}\n"
        result += f"Property: {booking['airbnb_name']}\n"
        result += f"Address: {booking['address']}\n"
        result += f"Guest: {booking['guest_name']}\n"
        result += f"Check-in: {booking['check_in']}\n"
        result += f"Check-out: {booking['check_out']}\n"
        result += f"Nightly Rate: ${booking['total_price']}\n\n"
        result += f"You'll receive a confirmation email shortly. Have a great stay!"

        return result        

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Trả về ngày giờ hiện tại"""
        now = datetime.now()
        # Format: Thứ Năm, 26 tháng 03 năm 2026, 16:00:39
        # In Vietnamese
       
        days = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        day_str = days[now.weekday()]
        return f"{day_str}, {now.strftime('%d/%m/%Y %H:%M:%S')}"

    # @function_tool
    # async def observe_scene(self, context: RunContext) -> str:
    #     """Sử dụng camera để quan sát và nhận dạng các đối tượng trong môi trường. 
    #     Hàm này trả về cả danh sách đối tượng và mô tả chi tiết từ YOLO.
    #     """
    #     logger.info("Agent is observing the scene...")
        
    #     # We need to call the MCP tool 'get_camera_frame'
    #     # Since MCP tools are attached to the session, we can call them via the session's tool call logic
    #     # OR we can just use the tool context.
        
    #     try:
    #         # Look for the MCP tool in the session
    #         # Note: In a real scenario, we might want to call the tool by name
    #         # For simplicity, we assume the agent will call get_camera_frame directly if needed,
    #         # but providing this wrapper helps enforce the "give to LLM" logic.
            
    #         # Actually, the best way for the agent to "see" is to use the existing prompt logic
    #         # but provide a better summary of what it sees.
            
    #         # If the user specifically wants the agent to "detect objects", 
    #         # we can use this tool to trigger the behavior.
    #         return "Please call `get_camera_frame` to receive the visual data and then provide a concise description in Vietnamese as per your instructions."
    #     except Exception as e:
    #         logger.error(f"Error in observe_scene: {e}")
    #         return f"Lỗi khi quan sát: {e}"

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        """Filter out background noise when the agent is speaking."""
        # Use simple heuristic: if agent is speaking, ignore everything except stop keywords
        if self._activity and self._activity.current_speech:
            transcript = " ".join(new_message.content) if isinstance(new_message.content, list) else str(new_message.content)
            transcript_lower = transcript.lower().strip()
            
            stop_keywords = ["stop", "dừng lại", "thôi", "im đi", "dừng", "đợi đã", "wait", "hold on"]
            if not any(kw in transcript_lower for kw in stop_keywords):
                logger.info(f"Ignoring background speech while agent is speaking: {transcript}")
                raise StopResponse()





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
            detect_language=["Vietnamese", "English"],
        ) if STT_PROVIDER == "vllm" else FasterWhisperSTT(
            model_size=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
            language="None",  # Enable auto-detection
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
        allow_interruptions=False,
        discard_audio_if_uninterruptible=False,
        tts_text_transforms=["filter_markdown", "filter_emoji", "filter_thoughts"],


        # MCP servers
        mcp_servers=[
            mcp.MCPServerStdio(
                command="/home/vrh3/workspace/projects/voice-agent/mcp/YOLO-MCP-Server/.venv/bin/python",
                args=["/home/vrh3/workspace/projects/voice-agent/mcp/YOLO-MCP-Server/server.py"],
                env={
                    "PYTHONPATH": "/home/vrh3/workspace/projects/voice-agent/mcp/YOLO-MCP-Server",
                    "DISPLAY": os.environ.get("DISPLAY", ":1"),
                    "XAUTHORITY": os.environ.get("XAUTHORITY", ""),
                    "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
                    "QT_X11_NO_MITSHM": "1"
                },
                client_session_timeout_seconds=20
            )
        ],
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
        logger.info(f"User said: {ev.transcript}")
        
        # Custom interruption logic
        stop_keywords = ["stop", "dừng lại", "thôi", "im đi", "dừng", "đợi đã", "wait", "hold on"]
        transcript_lower = ev.transcript.lower().strip()
        if any(kw in transcript_lower for kw in stop_keywords):
            logger.info("STOP keyword detected, interrupting agent...")
            session.interrupt(force=True)

    @session.on("agent_answer_transcribed")
    def on_agent_answer_transcribed(ev) -> None:
        logger.info(f"LLM Answer: {ev.transcript}")

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
        agent=VoiceAssistant(), 
        room_input_options=RoomInputOptions(
            # Enable noise cancellation
            noise_cancellation=noise_cancellation.BVC(),
            # For telephony, use: noise_cancellation.BVCTelephony()
        ),
    )


    # Handle session events
    @session.on("agent_state_changed")
    def on_state_changed(ev):
        """Log agent state changes."""
        logger.info(f"State: {ev.old_state} -> {ev.new_state}")
    
    @session.on("user_started_speaking")
    def on_user_speaking():
        """Track when user starts speaking."""
        logger.debug("User started speaking")
    
    @session.on("user_stopped_speaking")
    def on_user_stopped():
        """Track when user stops speaking."""
        logger.debug("User stopped speaking")

    
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
