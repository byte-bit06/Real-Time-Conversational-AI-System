"""
Conversational AI Demo: Real-Time Audio/Text Chat with Vision, TTS, and Screen Sharing

This script enables:
- Conversational AI interaction (audio and text)
- Live speech-to-text transcription
- Text-to-speech using Edge TTS
- Vision-Language model support (Qwen/Qwen2-VL)
- Real-time screen sharing and capture
- Kafka integration for agent message streaming

Author: [YOUR NAME or LEAVE BLANK]
Instructions for employers: Run as a demo for real-time AI chat, or review as a candidate's showcase of model integration, live input, and system design skills.
"""

import json
import torch
import threading
import time
import queue
import sys
import asyncio

# Import transformers for language models and streaming support
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

# ========== Optional Library Imports (feature toggles) ==========
# Keyboard events for hotkeys
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

# Vision-Language specialized imports (Qwen2-VL)
try:
    from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
    QWEN2VL_AVAILABLE = True
except ImportError:
    QWEN2VL_AVAILABLE = False
    Qwen2VLProcessor = None
    Qwen2VLForConditionalGeneration = None

# Audio (Speech-to-Text)
try:
    import pyaudio
    import numpy as np
    import whisper
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ Audio libraries not installed. No live voice input. (pip install pyaudio openai-whisper)")

# Text-to-Speech (Edge TTS)
TTS_AVAILABLE = False
edge_tts = None
try:
    import edge_tts
    import sounddevice as sd
    import numpy as np
    TTS_AVAILABLE = True
    print("✅ Text-to-Speech available (edge-tts)")
except ImportError as e:
    print(f"⚠️ TTS not available: {e} (pip install edge-tts sounddevice)")

# Screen capture/screenshot and GUI
SCREEN_CAPTURE_AVAILABLE = False
try:
    from PIL import ImageGrab, Image, ImageTk
    import base64, io
    SCREEN_CAPTURE_AVAILABLE = True
    print("✅ Screen capture (Pillow) available")
except ImportError as e:
    print(f"⚠️ Screen capture unavailable: {e} (pip install pillow)")
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk
    GUI_AVAILABLE = True
except ImportError:
    pass

# ========== Kafka Integration for Message Event Streaming ==========
from kafka import KafkaConsumer, KafkaProducer
consumer = KafkaConsumer(
    'agent-responses',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    consumer_timeout_ms=1000
)
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# ========== Conversation and Global App State ==========
conversation_history = []
generation_active = threading.Event()
generation_thread = None
current_streamer = None
current_screenshot = None
audio_playing = threading.Event()       # Prevent listen-while-TTS
screen_sharing_active = False
screen_sharing_window = None
screen_access_permission = False        # User permission for screen access

# ========== System Prompt: Task-Oriented, SPECIFIC, Short ==========
SYSTEM_PROMPT = (
    "Your name is Apollo. You are an intelligent, specific, ultra-brief AI assistant (max 50 tokens per response). "
    "NEVER give generic advice. Always be specific, solution-focused, and direct. "
    "Do not regurgitate user input. Be concise (max 2 sentences). "
    "Samples: 'I can help with that. What should the script do?' "
    "'Post on Twitter: Check out my new project!'"
    "Ask for clarifications only if absolutely required. "
)

# ========== Model Setup ==========
model_name = "Qwen/Qwen2-VL-7B-Instruct"
is_vision_model = True

def print_diagnostics():
    print(f"🔍 PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}, Mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("🖥️  No GPU - using CPU")

print("Loading model... This may take a few minutes on first run.")
print_diagnostics()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load processor (for vision) and tokenizer
processor, tokenizer, model = None, None, None
try:
    if is_vision_model and QWEN2VL_AVAILABLE:
        print("Loading Vision-Language model...")
        processor = Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if torch.cuda.is_available():
            print("Using GPU 4-bit quantization for Qwen2-VL.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quantization_config, device_map="auto",
                trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            print("Vision-Language model using CPU - will be slow.")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, device_map="cpu", trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True)
        print("Vision-Language model loaded.")
    else:
        raise Exception("Qwen2-VL model/class not available, falling back to text-only.")
except Exception as e:
    print(f"⚠️ Vision model load failed ({e}), reverting to text")
    is_vision_model = False
    QWEN2VL_AVAILABLE = False

if not is_vision_model or not QWEN2VL_AVAILABLE:
    print("Loading text-only model for fallback.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True)
    print("Text-only model loaded.")

# ========== UTILS (Speech, Audio Input, Vision) ==========

async def generate_speech_async(text: str, voice: str = "en-US-JennyNeural"):
    """
    Generate speech audio for given text (using edge-tts).
    Returns (audio_array, sample_rate).
    """
    if edge_tts is None:
        raise ImportError("edge_tts not available - cannot generate speech")
    
    # Validate input text
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for speech generation")
    
    text = text.strip()
    
    import tempfile, os
    tmp_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_path = tmp_file.name
        
        # Fallback voices to try in order
        voices_to_try = [
            voice,  # Try requested voice first
            "en-US-JennyNeural",  # Default fallback
            "en-US-AriaNeural",
            "en-US-GuyNeural",
            "en-US-DavisNeural",
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        voices_to_try = [v for v in voices_to_try if not (v in seen or seen.add(v))]
        
        last_error = None
        voice_used = None
        
        # Try voices in order
        for test_voice in voices_to_try:
            try:
                # Simple usage pattern as shown in example
                communicate = edge_tts.Communicate(text, test_voice)
                await communicate.save(tmp_path)
                voice_used = test_voice
                if test_voice != voice:
                    print(f"ℹ️ Using fallback voice: {voice_used}")
                break
            except Exception as e:
                last_error = e
                if test_voice == voice:
                    # Only show error for primary voice, not every fallback attempt
                    print(f"⚠️ Voice '{voice}' failed, trying alternatives...")
                continue
        
        if voice_used is None:
            raise Exception(f"TTS generation failed with all voices. Last error: {last_error}")
        
        # Verify file was created and has content
        if not os.path.exists(tmp_path):
            raise Exception("TTS output file was not created")
        
        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            raise Exception("TTS output file is empty - no audio was received")
        
        # Read audio file - soundfile handles both WAV and MP3
        audio_array = None
        sample_rate = None
        
        try:
            # soundfile can read both WAV and MP3 files
            import soundfile as sf
            audio_array, sample_rate = sf.read(tmp_path)
        except ImportError:
            # Fallback: try pydub for MP3 support, or wave for WAV
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(tmp_path)
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                if audio_segment.channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                sample_rate = audio_segment.frame_rate
                # Normalize to [-1, 1]
                audio_array = audio_array / (2**15)
            except ImportError:
                # Last resort: try wave (WAV only)
                import wave
                with wave.open(tmp_path, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(-1)
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Validate audio data
        if audio_array is None or len(audio_array) == 0:
            raise Exception("No audio data was extracted from TTS output file")
        
        # Convert stereo to mono for consistent playback
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize audio if needed
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val
        
        return audio_array, sample_rate
    except Exception as e:
        raise Exception(f"TTS error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                import time
                time.sleep(0.1)  # Brief delay to ensure file is closed
                os.remove(tmp_path)
            except Exception:
                pass

def speak_text(text: str):
    """Convert text to speech and play it (Edge TTS)."""
    if not TTS_AVAILABLE:
        print("⚠️ No TTS available.")
        return
    if not text or not text.strip():
        print("⚠️ Empty text, cannot generate speech")
        return
    try:
        print(f"🔊 (TTS) {text[:55] + ('...' if len(text) > 55 else '')}")
        audio_data, sample_rate = asyncio.run(generate_speech_async(text.strip()))
        if audio_data is not None and sample_rate is not None and len(audio_data) > 0:
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
        else:
            print("⚠️ No valid audio data generated")
    except ValueError as e:
        print(f"⚠️ TTS validation error: {e}")
    except Exception as e:
        error_msg = str(e)
        if "No audio was received" in error_msg or "empty" in error_msg.lower():
            print(f"⚠️ TTS service returned no audio (may be a network issue)")
        else:
            print(f"⚠️ TTS/playback error: {e}")

def transcribe_audio_chunk(audio_data, whisper_model, sample_rate: int = 16000) -> str:
    """Use Whisper to transcribe audio chunk."""
    try:
        result = whisper_model.transcribe(audio_data, language="en", task="transcribe")
        return result["text"].strip()
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""

def detect_voice_activity(audio_chunk, threshold: float = 0.02) -> bool:
    """Simple amplitude-based voice activity detection."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    peak = np.abs(audio_chunk).max()
    return rms > threshold and peak > (threshold * 2)

def record_and_transcribe_live(whisper_model, sample_rate: int = 16000, chunk_size: int = 1024, silence_threshold: float = 0.01, silence_duration: float = 1.0) -> str:
    """Live audio: Record until user stops speaking, then transcribe."""
    if not AUDIO_AVAILABLE:
        raise ImportError("Audio libraries not available")
    CHUNK = chunk_size
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sample_rate
    print("🎙️ Speak now (listening)...")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    audio_buffer = []
    last_voice_time = None
    required_silence_chunks = int(silence_duration * (RATE / CHUNK))
    consecutive_silence_chunks = 0
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if detect_voice_activity(audio_chunk, threshold=silence_threshold):
                audio_buffer.append(audio_chunk)
                last_voice_time = time.time()
                consecutive_silence_chunks = 0
                print(".", end="", flush=True)
            else:
                consecutive_silence_chunks += 1
                if audio_buffer and last_voice_time and (time.time() - last_voice_time > silence_duration) and consecutive_silence_chunks >= required_silence_chunks:
                    print("\n🔄 Transcribing...")
                    full_audio = np.concatenate(audio_buffer)
                    transcription = transcribe_audio_chunk(full_audio, whisper_model, sample_rate)
                    print(f"📝 You said: {transcription}")
                    return transcription
    except KeyboardInterrupt:
        print("\n[CTRL+C] Audio capture aborted.")
        return ""
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def stop_generation():
    """Stop any in-progress AI text generation."""
    global generation_active, generation_thread, current_streamer
    if generation_active.is_set():
        print("\n🛑 Stopping current AI response...")
        generation_active.clear()
        if generation_thread and generation_thread.is_alive():
            generation_thread.join(timeout=1.0)
        current_streamer = None

# ========== Main Generation: Prompt → Model → Stream Output ==========
def generate_response(user_msg: str, interrupt_event: threading.Event = None, image: 'Image.Image' = None):
    """Generate an AI response (streaming), optionally with screen image (if enabled)."""
    global conversation_history
    # Build prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history[-10:])
    
    # Include image if available and vision model is enabled
    if is_vision_model and image is not None and processor is not None:
        print("👁️ Vision model: Including screen image in context")
        messages.append({"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_msg}]})
    else:
        if image is not None:
            print("⚠️ Image provided but vision model not available - using text only")
        messages.append({"role": "user", "content": user_msg})
    # Prepare model input
    try:
        if is_vision_model and image is not None and processor is not None:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            input_ids = inputs.input_ids
            attention_mask = getattr(inputs, 'attention_mask', None)
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
    except Exception as e:
        print(f"⚠️ Vision input processing failed: {e}")
        return "", None, None

    print(f"🔧 Model input: {input_ids.shape} on {input_ids.device}")
    # Increase timeout and add better error handling
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
    
    # Gen kwargs - check if inputs exists for vision model
    generation_kwargs = None
    if is_vision_model and image is not None and processor is not None:
        # Check if inputs was created successfully
        if 'inputs' in locals() and inputs is not None:
            generation_kwargs = {**inputs, "max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9, "do_sample": True, "streamer": streamer, "pad_token_id": tokenizer.pad_token_id}
        else:
            print("⚠️ Vision inputs not properly created, falling back to text-only")
            generation_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9, "do_sample": True, "streamer": streamer, "pad_token_id": tokenizer.pad_token_id}
    else:
        generation_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9, "do_sample": True, "streamer": streamer, "pad_token_id": tokenizer.pad_token_id}
    
    # Threaded streaming with better error handling
    generation_error = [None]  # Use list to allow modification in nested function
    
    def generate_with_error_handling():
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            error_msg = str(e)
            generation_error[0] = error_msg
            import traceback
            print(f"\n❌ Generation error in thread: {error_msg}")
            print(f"   Full traceback:")
            traceback.print_exc()

    global generation_active, generation_thread, current_streamer
    generation_active.set()
    current_streamer = streamer
    generation_thread = threading.Thread(target=generate_with_error_handling, daemon=True)
    generation_thread.start()
    print("🤖 AI:", end=" ", flush=True)
    response = ""
    token_count = 0
    try:
        # Wait a bit longer for generation to start
        time.sleep(0.5)
        # Check if thread is still alive after brief wait
        if not generation_thread.is_alive():
            if generation_error[0]:
                print(f"\n⚠️ Generation thread died with error: {generation_error[0]}")
            else:
                print("\n⚠️ Generation thread died unexpectedly - no error reported")
        else:
            # Try to get tokens from streamer with timeout handling
            try:
                for token in streamer:
                    if interrupt_event is not None and interrupt_event.is_set():
                        print("\n⚠️ AI response interrupted by user input")
                        generation_active.clear()
                        interrupt_event.clear()
                        break
                    if not generation_active.is_set():
                        print("\n⚠️ Generation stopped by system")
                        break
                    if generation_error[0] is not None:
                        print(f"\n⚠️ Generation error detected: {generation_error[0]}")
                        break
                    token_count += 1
                    response += token
                    print(token, end="", flush=True)
                    producer.send("agent-responses", {"text": response})
                    producer.flush()
            except queue.Empty:
                # This is expected if streamer times out
                print("\n⚠️ Streamer timeout - no tokens received within timeout period")
                # Check if thread is still running
                if generation_thread.is_alive():
                    print("   Generation thread still running, waiting a bit longer...")
                    time.sleep(2.0)
                    # Try one more time
                    try:
                        for token in streamer:
                            token_count += 1
                            response += token
                            print(token, end="", flush=True)
                            producer.send("agent-responses", {"text": response})
                            producer.flush()
                    except queue.Empty:
                        print("\n   Still no tokens - generation may be stuck")
            except Exception as e:
                # Catch other exceptions (not queue.Empty)
                import traceback
                print(f"\n⚠️ Streaming error: {e}")
                print(f"   Error type: {type(e).__name__}")
                traceback.print_exc()
    except queue.Empty:
        # Handle queue.Empty at outer level too
        print("\n⚠️ Streamer timeout at outer level - no tokens received")
    except Exception as e:
        import traceback
        print(f"\n⚠️ Unexpected error in streaming loop: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
    finally:
        generation_thread.join(timeout=5.0)
        print()  # Newline after streaming
        
        # If no tokens received, try fallback generation
        if not response or token_count == 0:
            if generation_error[0]:
                print(f"⚠️ Generation failed: {generation_error[0]}")
            print("🔄 Attempting fallback non-streaming generation...")
            try:
                with torch.no_grad():
                    # Create fallback kwargs without streamer
                    fallback_kwargs = {}
                    for k, v in generation_kwargs.items():
                        if k != "streamer":
                            fallback_kwargs[k] = v
                    
                    print(f"   Fallback kwargs keys: {list(fallback_kwargs.keys())}")
                    print(f"   Input shape: {input_ids.shape}, Device: {input_ids.device}")
                    print(f"   Model device: {next(model.parameters()).device}")
                    
                    outputs = model.generate(**fallback_kwargs)
                    
                    # Decode only new tokens
                    if outputs is not None and len(outputs) > 0:
                        new_tokens = outputs[0][input_ids.shape[1]:]
                        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        if response:
                            print(f"🤖 AI (fallback): {response}")
                        else:
                            print("⚠️ Fallback generation produced empty decoded response")
                            print(f"   Raw tokens length: {len(new_tokens)}")
                    else:
                        print("⚠️ Fallback generation returned None or empty output")
            except Exception as fallback_error:
                print(f"⚠️ Fallback generation failed: {fallback_error}")
                import traceback
                traceback.print_exc()
                # Check for common issues
                if "device" in str(fallback_error).lower():
                    print("   💡 Possible device mismatch - check model and input device")
                if "memory" in str(fallback_error).lower() or "cuda" in str(fallback_error).lower():
                    print("   💡 Possible memory issue - try reducing max_new_tokens or using CPU")

    # Play synthesized audio (complete response)
    audio_data = sample_rate = None
    if TTS_AVAILABLE and response.strip():
        try:
            print("\n🔊 Generating speech for AI response...")
            response_text = response.strip()
            # Skip very short responses to avoid TTS issues
            if len(response_text) < 3:
                print("⚠️ Response too short for TTS, skipping audio")
            else:
                audio_data, sample_rate = asyncio.run(generate_speech_async(response_text))
                if audio_data is not None and sample_rate is not None and len(audio_data) > 0:
                    print("🔊 Playing audio...")
                    audio_playing.set()
                    sd.play(audio_data, samplerate=sample_rate)
                    sd.wait()
                    audio_playing.clear()
                    print("✅ Audio playback complete")
                else:
                    print("⚠️ No valid audio data generated, skipping playback")
        except ValueError as e:
            print(f"⚠️ TTS validation error: {e}")
            audio_playing.clear()
        except Exception as e:
            error_msg = str(e)
            if "No audio was received" in error_msg or "empty" in error_msg.lower():
                print(f"⚠️ TTS service returned no audio. This may be a temporary network issue.")
                print(f"   Response text: '{response_text[:50]}...' (continuing without audio)")
            else:
                print(f"⚠️ Audio playback failed: {e}")
            audio_playing.clear()
    # Ensure we have a response
    if not response or not response.strip():
        response = "I apologize, but I'm having trouble generating a response right now. Please try again."
        print("⚠️ Warning: Empty response - using fallback message")
    
    # Only add to history if we got a real response
    if response and response.strip() and not response.startswith("I apologize"):
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": response.strip()})
        # Maintain last 10 messages only
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
    
    return response.strip(), audio_data, sample_rate

# ========== UI: Screen Sharing ==========
def create_screen_sharing_window():
    """Create (or focus) a window that shows your screen in real-time."""
    global screen_sharing_window, screen_sharing_active, current_screenshot
    if not (GUI_AVAILABLE and SCREEN_CAPTURE_AVAILABLE):
        print("⚠️ GUI or screenshot dependencies missing. No screen sharing window.")
        return None
    if screen_sharing_window is not None:
        try:
            screen_sharing_window.lift()
            screen_sharing_window.focus_force()
            return screen_sharing_window
        except:
            pass
    def update_screen():
        global current_screenshot, screen_sharing_active
        if not screen_sharing_active:
            return
        try:
            screenshot = ImageGrab.grab(all_screens=True)
            current_screenshot = screenshot
            max_width, max_height = 800, 600
            screenshot.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(screenshot)
            label.config(image=photo)
            label.image = photo
            if screen_sharing_active:
                root.after(500, update_screen)
        except Exception as e:
            print(f"⚠️ Screen update error: {e}")
            if screen_sharing_active:
                root.after(1000, update_screen)
    def on_closing():
        global screen_sharing_active, screen_sharing_window
        screen_sharing_active = False
        screen_sharing_window = None
        root.destroy()
    root = tk.Tk()
    root.title("Screen Sharing - AI Can See This")
    root.geometry("800x600")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    label = tk.Label(root)
    label.pack(fill=tk.BOTH, expand=True)
    status_label = tk.Label(root, text="🟢 Screen sharing active - AI can see your screen", bg="green", fg="white", font=("Arial", 10, "bold"))
    status_label.pack(fill=tk.X)
    screen_sharing_active = True
    screen_sharing_window = root
    update_screen()
    threading.Thread(target=root.mainloop, daemon=True).start()
    return root

def toggle_screen_sharing():
    """Toggle the visibility of the screen sharing window."""
    global screen_sharing_active, screen_sharing_window, screen_access_permission
    if not screen_access_permission:
        print("⚠️ Screen access permission not granted. Cannot enable screen sharing.")
        return False
    if screen_sharing_active and screen_sharing_window is not None:
        try:
            screen_sharing_window.destroy()
        except:
            pass
        screen_sharing_window = None
        screen_sharing_active = False
        print("📴 Screen sharing stopped")
        return False
    else:
        create_screen_sharing_window()
        print("📺 Screen sharing started")
        return True

def capture_and_share_screen():
    """Capture screen and provide as image (for vision model)."""
    if not SCREEN_CAPTURE_AVAILABLE:
        return False, "Screen capture unavailable (pip install pillow)", None
    try:
        print("📸 Capturing screen...")
        screenshot = ImageGrab.grab(all_screens=True)
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        import os
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, f"screenshot_{int(time.time())}.png")
        screenshot.save(screenshot_path)
        print(f"✅ Screenshot saved: {screenshot_path}")
        msg = "I've shared my screen with you. Please analyze what you see and help me with any tasks."
        return True, msg, screenshot
    except Exception as e:
        print(f"⚠️ Error capturing screen: {e}")
        return False, str(e), None

# ============================== MAIN INTERACTIVE LOOP ==============================
def main():
    global screen_access_permission
    
    print("="*60)
    print("🚀 Real-Time Conversational AI Demo")
    print("="*60)
    
    # Ask for screen access permission before starting
    if SCREEN_CAPTURE_AVAILABLE:
        print("\n📺 Screen Access Permission")
        print("-" * 60)
        print("The AI can analyze your screen to provide visual context.")
        print("This allows the AI to see what's on your screen when you share it.")
        print("-" * 60)
        while True:
            response = input("Allow the AI to access your screen? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                screen_access_permission = True
                print("✅ Screen access enabled.")
                print("   💡 To enable screen sharing: Type 'screenshot' or press 'S' key")
                print("   💡 Screenshots will be automatically included in queries when enabled")
                break
            elif response in ['no', 'n']:
                screen_access_permission = False
                print("❌ Screen access disabled. Screen sharing features will be unavailable.")
                break
            else:
                print("⚠️ Please enter 'yes' or 'no'.")
        print()
    
    print("Features:")
    print(" - Voice: Just start speaking")
    if screen_access_permission:
        print(" - Screenshot (visual context): Say or type 'screenshot', or press 'S' (with keyboard lib)")
    print(" - Switch input: Type 'text' (text only) or 'audio' (voice)")
    print(" - Exit: Type 'quit'/'exit'/'q'")
    print("="*60)
    print(f"Text-to-Speech: {'✅ available' if TTS_AVAILABLE else '❌ not available'}\n")

    # Whisper Speech-to-Text load
    whisper_model = None
    audio_available = AUDIO_AVAILABLE
    if audio_available:
        try:
            print("Loading Whisper model (for speech recognition)...")
            whisper_model = whisper.load_model("base")
            print("Whisper model loaded.")
        except Exception as e:
            print(f"⚠️ Whisper model failed to load: {e}")
            audio_available = False
    audio_enabled = audio_available and whisper_model is not None
    
    screenshot_triggered = threading.Event()
    if KEYBOARD_AVAILABLE and screen_access_permission:
        def on_screenshot_key():
            screenshot_triggered.set()
        keyboard.on_press_key('s', lambda _: on_screenshot_key())
        print("⌨️  Hotkey enabled: Press 'S' to share screen")
    elif KEYBOARD_AVAILABLE:
        print("⌨️  Hotkey disabled: Screen access not granted")
    else:
        print("(Install 'keyboard' lib for global 'S' screenshot hotkey)")

    try:
        global screen_sharing_active, current_screenshot
        while True:
            user_input = None
            interrupt_event = threading.Event()
            if audio_enabled:
                if audio_playing.is_set():
                    while audio_playing.is_set():
                        time.sleep(0.1)
                    time.sleep(0.3)
                try:
                    user_input = record_and_transcribe_live(whisper_model, sample_rate=16000, silence_duration=1.5, silence_threshold=0.02)
                    if not user_input:
                        continue
                    if generation_active.is_set():
                        print("🛑 Interrupting AI response...")
                        stop_generation()
                        interrupt_event.set()
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    print("[Audio recording cancelled.]")
                    continue
                except Exception as e:
                    print(f"⚠️ Audio input error: {e}")
                    continue
            else:
                if KEYBOARD_AVAILABLE and screenshot_triggered.is_set():
                    screenshot_triggered.clear()
                    if not screen_access_permission:
                        print("⚠️ Screen access was denied at startup. Cannot enable screen sharing.")
                        continue
                    is_active = toggle_screen_sharing()
                    if is_active:
                        print("📺 Screen sharing ON via keypress")
                    else:
                        print("📴 Screen sharing turned off via keypress")
                    continue
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() == 'text':
                    print("📝 Switched to text input mode")
                    continue
                elif user_input.lower() == 'audio':
                    if audio_available and whisper_model:
                        audio_enabled = True
                        print("🎤 Switched to audio input mode")
                        continue
                    else:
                        print("⚠️ Audio not available")
                        continue

            # Exit?
            if not user_input:
                print("⚠️ Please enter a message.")
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye, thanks for trying the demo!")
                break

            # Screen sharing and screenshot?
            if user_input.lower() in ['screenshot', 'share screen', 'screen', 'capture screen', 'share', 'toggle screen']:
                if not screen_access_permission:
                    print("⚠️ Screen access was denied at startup. Please restart the application to enable screen sharing.")
                    continue
                is_active = toggle_screen_sharing()
                if is_active:
                    print("📺 Screen sharing ON - AI can now see your screen!")
                    print("   All future queries will include your screen automatically.")
                else:
                    print("📴 Screen sharing stopped - AI will no longer see your screen")
                    current_screenshot = None
                continue

            # Send message (for testing with agent consumers)
            producer.send("chat-messages", {"text": user_input})
            producer.flush()
            if interrupt_event:
                interrupt_event.clear()

            # Prepare vision context (only if permission granted)
            screenshot_to_use = None
            if screen_access_permission:
                # If screen sharing window is active, use its screenshot
                if screen_sharing_active and current_screenshot is not None:
                    try:
                        fresh_screenshot = ImageGrab.grab(all_screens=True)
                        screenshot_to_use = fresh_screenshot
                        current_screenshot = fresh_screenshot
                        print("📸 Including current screen in AI context...")
                    except Exception as e:
                        print(f"⚠️ Failed to capture fresh screenshot: {e}, using cached")
                        screenshot_to_use = current_screenshot
                # If screen sharing is not explicitly enabled but permission is granted,
                # automatically capture screenshot for this query
                elif not screen_sharing_active:
                    try:
                        screenshot_to_use = ImageGrab.grab(all_screens=True)
                        print("📸 Capturing screen for AI context...")
                    except Exception as e:
                        print(f"⚠️ Failed to capture screenshot: {e}")
                        screenshot_to_use = None

            try:
                response, audio_data, sample_rate = generate_response(user_input, interrupt_event=interrupt_event, image=screenshot_to_use)
                if audio_data is not None and sample_rate is not None:
                    time.sleep(0.3)
                if not response:
                    print("⚠️ No AI response generated.")
            except KeyboardInterrupt:
                print("\n[Generation cancelled by user]")
                break
            except Exception as e:
                print(f"\n❌ Error during response: {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye, exiting.")
        stop_generation()
    except Exception as e:
        print(f"\n❌ Uncaught error: {e}")
        stop_generation()

if __name__ == "__main__":
    main()
