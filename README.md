# Local LiveKit Plugins

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents-purple.svg)](https://docs.livekit.io/agents/)

**Run LiveKit voice agents with fully local STT and TTS - no cloud APIs required.**

Custom plugins for [LiveKit Agents](https://docs.livekit.io/agents/) that enable completely local speech processing using [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) for STT and [Piper](https://github.com/rhasspy/piper) for TTS.

## Tested On

- **OS:** Linux (Arch 6.17.7)
- **GPU:** NVIDIA RTX 3060 (12GB VRAM)
- **CUDA:** 12.x
- **Python:** 3.10+

**Windows/Mac users:** Not yet tested. Community contributions welcome! Please report issues on GitHub.

## Why Local?

| | Cloud | Local |
|---|---|---|
| **Quality** | Better | Good |
| **Latency** | ~2.1s total | ~2.8s total |
| **Cost** | ~$150/year* | ~$20/year |
| **Privacy** | Data sent externally | Stays on your network |
| **Control** | Vendor dependent | Full ownership |

*Based on 100 hours/year: Deepgram Nova-2 ($0.35/hr) + Cartesia Sonic ($50/1M chars).

## Features

- **FasterWhisperSTT** - GPU-accelerated speech-to-text
  - Multiple model sizes (tiny → large-v3)
  - ~230-460ms processing time on RTX 3060
  - Configurable beam search and VAD

- **PiperTTS** - Fast local text-to-speech
  - Multiple voice models available
  - ~9ms per character (~300ms for short responses)
  - Configurable speed, volume, pitch

## Quick Start

### Prerequisites

**Required:**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Python 3.10+
- Docker (for LiveKit server)
- [Ollama](https://ollama.ai/) (for local LLM) - Must be running: `ollama serve`
- ffmpeg or libavcodec (for audio processing)

**For GPU Acceleration (recommended):**
- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended for larger Whisper models)
- NVIDIA drivers with CUDA 11.8+ support
- Note: PyTorch (~2GB download) includes bundled CUDA libraries

### 1. Clone and Install

```bash
git clone https://github.com/CoreWorxLab/local-livekit-plugins.git
cd local-livekit-plugins

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Download a Piper Voice Model

```bash
mkdir -p models/piper && cd models/piper

# Download Ryan (male US English, high quality)
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json
```

More voices: [Piper Voices on HuggingFace](https://huggingface.co/rhasspy/piper-voices/tree/main)

### 3. Start LiveKit Server

```bash
docker compose up -d

# Verify it's running
curl http://localhost:7880
```

### 4. Configure and Run

```bash
# Setup environment file
cp examples/.env.local.example examples/.env.local
# Edit examples/.env.local with your model paths

# Run with local pipeline (console mode for testing)
uv run examples/voice_agent.py console

# Or run with cloud pipeline (requires API keys)
USE_LOCAL=false uv run examples/voice_agent.py console

# For dev mode (connects to LiveKit playground)
uv run examples/voice_agent.py dev
```

## Using the Plugins in Your Own Project

### Install from GitHub

```bash
uv add git+https://github.com/CoreWorxLab/local-livekit-plugins.git
```

### Use in Your Agent

```python
from local_livekit_plugins import FasterWhisperSTT, PiperTTS
from livekit.agents import AgentSession
from livekit.plugins import silero, openai as lk_openai

# Create local STT
stt = FasterWhisperSTT(
    model_size="medium",      # tiny, base, small, medium, large-v3
    device="cuda",            # cuda or cpu
    compute_type="float16",   # float16, int8
)

# Create local TTS
tts = PiperTTS(
    model_path="/path/to/en_US-ryan-high.onnx",
    use_cuda=False,           # CPU recommended for compatibility
    speed=1.0,
)

# Create session with local LLM (Ollama)
session = AgentSession(
    stt=stt,
    llm=lk_openai.LLM.with_ollama(model="llama3.1:8b"),
    tts=tts,
    vad=silero.VAD.load(),
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                       │
├─────────────────────────────────────────────────────────────┤
│                    LiveKit Agents SDK                       │
├───────────────────┬──────────────────┬──────────────────────┤
│  FasterWhisperSTT │      LLM        │      PiperTTS         │
│  (this package)   │   (Ollama)      │   (this package)      │
├───────────────────┼─────────────────┼───────────────────────┤
│  faster-whisper   │   ollama        │    piper-tts          │
│     + CUDA        │                 │   + onnxruntime       │
└───────────────────┴─────────────────┴───────────────────────┘
```

## Configuration Reference

### FasterWhisperSTT

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | "medium" | Model size: tiny, base, small, medium, large-v3 |
| `device` | str | "cuda" | Processing device: cuda, cpu, auto |
| `compute_type` | str | "float16" | Quantization: float16, int8, int8_float16 |
| `language` | str | "en" | Language hint for recognition |
| `beam_size` | int | 5 | Beam search width (1-10) |
| `vad_filter` | bool | True | Enable voice activity detection |

### PiperTTS

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to .onnx voice model |
| `use_cuda` | bool | False | Enable GPU (has CUDA version constraints) |
| `speed` | float | 1.0 | Speech rate (0.5-2.0) |
| `volume` | float | 1.0 | Volume level |
| `noise_scale` | float | 0.667 | Phoneme variation |
| `noise_w` | float | 0.8 | Phoneme width variation |

## Development

```bash
# Clone the repo
git clone https://github.com/CoreWorxLab/local-livekit-plugins.git
cd local-livekit-plugins

# Install with dev dependencies
uv sync --all-extras

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest
```

## Performance

Tested on RTX 3060 12GB:

| Component | Metric | Value |
|-----------|--------|-------|
| STT (whisper-medium) | Processing time | ~230-460ms |
| STT (whisper-medium) | End-to-end (transcript_delay) | ~760ms avg |
| TTS (piper ryan-high) | Per character | ~9ms |
| TTS (piper ryan-high) | Short response (30 chars) | ~270ms |
| TTS (piper ryan-high) | Long response (130 chars) | ~1200ms |

**Note:** End-to-end latency includes VAD buffering. Local STT uses batch processing (waits for speech to end), while cloud STT streams in real-time.

## Known Issues

### Installation & Platform

1. **Large Download Size**: PyTorch with CUDA support is ~2GB. First install may take a while depending on your connection.

2. **Windows/Mac Untested**: This has only been tested on Linux. You may encounter:
   - Path handling issues (especially Windows)
   - Platform-specific audio library requirements
   - Different PyTorch wheel availability
   - **Help wanted!** If you get it working, please share your setup in a GitHub issue or PR.

3. **Whisper Model Downloads**: Models download automatically on first run (1-5GB depending on size). This happens silently - check logs if first startup seems slow.

### Technical Limitations

4. **Piper GPU Support**: Piper has CUDA version constraints. If you're on CUDA 12+, you may need to:
   - Use CPU mode (`use_cuda=False`) - often faster anyway for short utterances
   - Containerize with a compatible CUDA version
   - See: [Piper CUDA 12 Discussion](https://github.com/rhasspy/piper/discussions/544)

5. **No Streaming STT**: FasterWhisper uses batch processing - it waits for speech to end before transcribing. Cloud services like Deepgram stream audio in real-time, giving them a ~300ms latency advantage. This is a fundamental limitation of Whisper-based solutions.

6. **Quality vs Cloud**: Local models are good but not as polished as cloud services like Deepgram or ElevenLabs.

### Getting Help

**Community Support:** This project is community-supported. For issues:
- Check [existing GitHub issues](https://github.com/CoreWorxLab/local-livekit-plugins/issues)
- Search for similar problems (especially platform-specific)
- Open a new issue with your system info (OS, GPU, Python version, error logs)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
  - PyTorch is included as a dependency (~850MB) and provides bundled CUDA 12 libraries
  - No separate CUDA toolkit installation required
- ~2-4GB VRAM for Whisper medium model
- ~500MB disk per Piper voice model

## Related Projects

- [LiveKit Agents](https://github.com/livekit/agents) - The framework these plugins integrate with
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 Whisper implementation
- [Piper](https://github.com/rhasspy/piper) - Fast local neural TTS
- [Ollama](https://ollama.ai/) - Local LLM server
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

## Contributing

Contributions are welcome! This project is in active development and we'd love your help.

**High Priority:**
- **Platform testing**: Windows and Mac users - try it out and report what works/breaks
- **GPU compatibility**: Test with different CUDA versions, AMD GPUs, or CPU-only setups
- **Documentation**: Improve setup instructions, add troubleshooting tips

**Also Welcome:**
- Bug fixes and performance improvements
- New features (open an issue first to discuss)
- Better error messages and logging
- Example projects and use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, or just open an issue to start a discussion!

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with [Claude Code](https://claude.ai/claude-code)**
