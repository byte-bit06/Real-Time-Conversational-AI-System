# Real-Time Conversational AI System 🗣️

Experience natural, low-latency human-AI interaction with multimodal understanding, vision capabilities, and responsive task automation.

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/byte-bit06/Real-Time-Conversational-AI-System)
[![License](https://img.shields.io/badge/license-None-lightgrey)](https://github.com/byte-bit06/Real-Time-Conversational-AI-System)
![Stars](https://img.shields.io/github/stars/byte-bit06/Real-Time-Conversational-AI-System?style=social)
![Forks](https://img.shields.io/github/forks/byte-bit06/Real-Time-Conversational-AI-System?style=social)

## ✨ Key Features

This system provides seamless, intuitive human-AI collaboration with advanced multimodal capabilities.

*   **⚡ Real-time Streaming Responses:** Token-by-token streaming inference for immediate, natural conversation flow without noticeable delays.

*   **👁️ Vision-Language Understanding:** Powered by Qwen2-VL-7B-Instruct model, capable of analyzing screenshots and visual content alongside text.

*   **🎤 Live Speech-to-Text:** Real-time audio transcription using OpenAI Whisper with voice activity detection for hands-free interaction.

*   **🔊 Text-to-Speech:** Natural voice output using Microsoft Edge TTS with multiple voice fallbacks for reliable audio generation.

*   **📺 Screen Sharing & Analysis:** Capture and share your screen with the AI for visual context-aware assistance.

*   **⌨️ Keyboard Shortcuts:** Global hotkeys (press 'S' key) for quick screen capture and sharing.

*   **🔄 Kafka Integration:** Message streaming infrastructure for real-time agent responses and event handling.

*   **🧠 Context-Aware Conversations:** Maintains conversation history for coherent multi-turn dialogues.

## 🚀 Installation Guide

### Prerequisites

*   Python 3.8+
*   CUDA-capable GPU (recommended) or CPU support
*   Docker & Docker Compose (for Kafka services)
*   Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/byte-bit06/Real-Time-Conversational-AI-System.git
cd Real-Time-Conversational-AI-System
```

### Step 2: Set Up Kafka (Required for Message Streaming)

Start Kafka and Zookeeper services using Docker Compose:

```bash
docker-compose up -d
```

This will start Kafka on `localhost:9092` for message streaming.

### Step 3: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
```

### Step 4: Install Dependencies

```bash
# Install PyTorch (choose based on your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# OR for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# OR for CPU-only:
pip install torch torchvision torchaudio

# Install core dependencies
pip install transformers accelerate bitsandbytes
pip install pyaudio openai-whisper
pip install edge-tts sounddevice soundfile
pip install pillow
pip install kafka-python
pip install keyboard  # Optional: for global hotkeys
```

**Note:** On Windows, installing `pyaudio` may require:
```bash
pip install pipwin
pipwin install pyaudio
```

### Step 5: Download Models

The system will automatically download models on first run:
- **Qwen2-VL-7B-Instruct** (Vision-Language model) - ~14GB
- **Whisper Base** (Speech recognition) - ~150MB

Models are cached in `~/.cache/huggingface/` after first download.

## 💡 Usage

### Starting the System

1. **Ensure Kafka is running:**
   ```bash
   docker-compose up -d
   ```

2. **Run the main script:**
   ```bash
   python main.py
   ```

### First Run Setup

On first launch, the system will:
1. Check GPU availability and display diagnostics
2. Download required models (this may take several minutes)
3. Ask for screen access permission
4. Load Whisper model for speech recognition

### Interaction Modes

#### Audio Mode (Default)
- **Speak naturally** - The system listens continuously
- **Voice Activity Detection** - Automatically detects when you start/stop speaking
- **Auto-transcription** - Speech is transcribed when you pause (~1.5 seconds of silence)

#### Text Mode
- Type `text` to switch to text input mode
- Enter your message and press Enter
- Type `audio` to switch back to voice mode

### Screen Sharing

1. **Enable Screen Sharing:**
   - Type `screenshot` or `share screen`
   - Or press `S` key (if keyboard library is installed)
   - A window will open showing what the AI can see

2. **Visual Queries:**
   - Ask questions about what's on your screen
   - The AI will analyze screenshots automatically
   - Example: "What's on my screen?" or "Help me with this code"

### Common Commands

- `screenshot` / `share screen` - Toggle screen sharing
- `text` - Switch to text input mode
- `audio` - Switch to audio input mode
- `quit` / `exit` / `q` - Exit the application

### Example Interactions

```
👤 You: What's the weather like today?
🤖 AI: I don't have access to real-time weather data, but I can help you find a weather service or check online.

👤 You: [Shares screen] What's on my screen?
🤖 AI: I can see you have a code editor open with Python code. Would you like help with anything specific?

👤 You: Set a timer for 5 minutes
🤖 AI: I can't directly set timers, but I can help you write a script to do that or suggest timer apps.
```

## 🛠️ Technical Details

### Architecture

- **Vision Model:** Qwen/Qwen2-VL-7B-Instruct (4-bit quantized on GPU)
- **Speech Recognition:** OpenAI Whisper (Base model)
- **Text-to-Speech:** Microsoft Edge TTS (cloud-based, free)
- **Message Streaming:** Apache Kafka
- **Streaming Inference:** Transformers TextIteratorStreamer

### Performance

- **GPU Recommended:** CUDA-enabled GPU with 8GB+ VRAM for optimal performance
- **CPU Support:** Available but significantly slower
- **Model Quantization:** 4-bit quantization reduces VRAM usage by ~75%

### System Requirements

- **Minimum:** 16GB RAM, CPU-only mode
- **Recommended:** 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM, CUDA 11.8+

## 🛣️ Project Roadmap

*   **Version 1.1 - Enhanced Vision:**
    *   `👁️` Improved screen analysis accuracy
    *   `🖼️` Support for multiple monitor setups
    *   `📸` Better screenshot quality and compression

*   **Version 1.2 - Advanced Features:**
    *   `🧠` Long-term memory and context persistence
    *   `🔗` API integrations (calendar, email, smart home)
    *   `⚙️` Plugin architecture for extensibility

*   **Version 1.3 - Performance & Security:**
    *   `⚡` Further latency optimizations
    *   `🔒` Enhanced privacy controls for screen sharing
    *   `📈` Performance benchmarking tools

## 🤝 Contribution Guidelines

We welcome contributions! Please follow these guidelines:

1. **Fork the Repository:** Start by forking the project to your GitHub account.

2. **Create a New Branch:**
    *   For features: `git checkout -b feature/your-feature-name`
    *   For bug fixes: `git checkout -b bugfix/issue-description`

3. **Code Style:**
    *   Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
    *   Use descriptive variable names and add comments for complex logic
    *   Maintain the existing code structure and patterns

4. **Commit Messages:** Write clear, concise commit messages:
    *   Example: `feat: Add voice activity detection threshold configuration`
    *   Example: `fix: Resolve TTS voice fallback issue`

5. **Testing:**
    *   Test your changes with both GPU and CPU modes if possible
    *   Verify audio input/output functionality
    *   Test screen sharing features

6. **Pull Request Process:**
    *   Open a Pull Request to the `main` branch
    *   Provide a detailed description of changes
    *   Include screenshots or examples if applicable
    *   Be responsive to feedback during review

## ⚠️ Known Issues & Troubleshooting

### Audio Input Not Working
- **Windows:** Install `pipwin` and use `pipwin install pyaudio`
- **Linux:** Install `portaudio19-dev`: `sudo apt-get install portaudio19-dev`
- **macOS:** Install via Homebrew: `brew install portaudio`

### GPU Not Detected
- Verify CUDA installation: `nvidia-smi`
- Install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)
- Check CUDA version compatibility

### Kafka Connection Errors
- Ensure Docker containers are running: `docker-compose ps`
- Check Kafka is accessible: `docker-compose logs kafka`
- Verify port 9092 is not in use by another service

### Model Download Issues
- Check internet connection
- Verify sufficient disk space (~15GB for models)
- Try manual download from Hugging Face

## 📄 License Information

This project currently does not have a formal license. All rights are reserved by the main contributor.

*   **Copyright © 2025 byte-bit06.** All rights reserved.
*   **Usage Restrictions:** Without an explicit license, redistribution, modification, or commercial use of this software is generally restricted. Please contact the main contributor for specific permissions.

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen2-VL vision-language model
- **OpenAI** for Whisper speech recognition
- **Microsoft** for Edge TTS service
- **Hugging Face** for transformers library and model hosting

