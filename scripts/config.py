import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    USE_LOCAL: bool = Field(default_factory=lambda: os.getenv("USE_LOCAL", "false").lower() == "true")
    
    # Local pipeline settings
    WHISPER_MODEL: str = Field(default_factory=lambda: os.getenv("WHISPER_MODEL", "medium"))
    WHISPER_DEVICE: str = Field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    STT_PROVIDER: str = Field(default_factory=lambda: os.getenv("STT_PROVIDER", "faster_whisper").lower())
    VLLM_ASR_MODEL: str = Field(default_factory=lambda: os.getenv("VLLM_ASR_MODEL", "qwen3asr"))
    VLLM_ASR_BASE_URL: str = Field(default_factory=lambda: os.getenv("VLLM_ASR_BASE_URL", "http://10.148.180.105:1670/v1"))
    TTS_PROVIDER: str = Field(default_factory=lambda: os.getenv("TTS_PROVIDER", "vieneu").lower())
    PIPER_MODEL_PATH: str = Field(default_factory=lambda: os.getenv("PIPER_MODEL_PATH", ""))
    PIPER_USE_CUDA: bool = Field(default_factory=lambda: os.getenv("PIPER_USE_CUDA", "false").lower() == "true")

    VIENEU_API_BASE: str = Field(default_factory=lambda: os.getenv("VIENEU_API_BASE", "http://localhost:23333/v1"))
    VIENEU_MODEL_NAME: str = Field(default_factory=lambda: os.getenv("VIENEU_MODEL_NAME", "pnnbao-ump/VieNeu-TTS"))
    VIENEU_VOICE_ID: str = Field(default_factory=lambda: os.getenv("VIENEU_VOICE_ID", "Doan")) # e.g. "nu_m_001"

    LLM_PROVIDER: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    LANGUAGE: str = Field(default_factory=lambda: os.getenv("LANGUAGE", "vi").lower()) # "vi" or "en"
    OLLAMA_MODEL: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    OLLAMA_BASE_URL: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    VLLM_MODEL: str = Field(default_factory=lambda: os.getenv("VLLM_MODEL", "qwen3_vl"))
    VLLM_BASE_URL: str = Field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://10.148.180.105:22003/v1"))

# Create a singleton instance
settings = Settings()

# Export variables for backward compatibility
USE_LOCAL = settings.USE_LOCAL
WHISPER_MODEL = settings.WHISPER_MODEL
WHISPER_DEVICE = settings.WHISPER_DEVICE
STT_PROVIDER = settings.STT_PROVIDER
VLLM_ASR_MODEL = settings.VLLM_ASR_MODEL
VLLM_ASR_BASE_URL = settings.VLLM_ASR_BASE_URL
TTS_PROVIDER = settings.TTS_PROVIDER
PIPER_MODEL_PATH = settings.PIPER_MODEL_PATH
PIPER_USE_CUDA = settings.PIPER_USE_CUDA
VIENEU_API_BASE = settings.VIENEU_API_BASE
VIENEU_MODEL_NAME = settings.VIENEU_MODEL_NAME
VIENEU_VOICE_ID = settings.VIENEU_VOICE_ID
LLM_PROVIDER = settings.LLM_PROVIDER
LANGUAGE = settings.LANGUAGE
OLLAMA_MODEL = settings.OLLAMA_MODEL
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
VLLM_MODEL = settings.VLLM_MODEL
VLLM_BASE_URL = settings.VLLM_BASE_URL
