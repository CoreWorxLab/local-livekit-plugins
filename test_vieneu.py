import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import asyncio
import local_livekit_plugins
print(f"📦 Imported local_livekit_plugins from: {local_livekit_plugins.__file__}")

from local_livekit_plugins import VieNeuTTS

logging.basicConfig(level=logging.INFO)

async def test_init():
    try:
        print("🚀 Testing VieNeuTTS initialization...")
        # Note: VieNeuTTS will add models/VieNeu-TTS/src to sys.path itself
        tts = VieNeuTTS(
            api_base="http://localhost:23333/v1",
            model_name="pnnbao-ump/VieNeu-TTS"
        )
        print("✅ VieNeuTTS initialized successfully!")
        
        # Test synthesis capability (without streaming to room)
        print("🗣️ Testing synthesis stream...")
        stream = tts.synthesize("Xin chào, tôi là trợ lý ảo.")
        print(f"✅ Stream created for: {stream.input_text}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_init())
