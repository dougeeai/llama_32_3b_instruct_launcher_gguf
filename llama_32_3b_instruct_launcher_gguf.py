# %% [0.0] Launcher Script Info
# Script metadata and documentation for version tracking
# Llama 3.2 3B Instruct GGUF Launcher
# Description: Optimized launcher for Llama-3.2-3B-Instruct GGUF models
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11
# Optimized for: Python 3.13 + CUDA 13.0

# %% [0.1] Model Card & Summary
# Quick reference for model capabilities and requirements
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)

# %% [1.0] Core Imports
# Essential Python libraries required for GGUF operation

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GGUF support
from llama_cpp import Llama, LlamaGrammar

# %% [1.1] Utility Imports
# Supporting libraries for hardware monitoring and performance metrics

import time
import psutil
import platform
from datetime import datetime

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
except:
    NVIDIA_GPU_AVAILABLE = False

# %% [2.0] Base Directory Configuration
# Set base AI directory - all paths will be relative to this location

# Set your base directory (change this for your system)
BASE_DIR = r"E:\ai"  # <-- CHANGE THIS to the AI folder location

# Directory structure (automatically created from BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
HF_DOWNLOADS_DIR = os.path.join(MODELS_DIR, "huggingface_downloads")  # For HF downloads
HF_CACHE_DIR = os.path.join(HF_DOWNLOADS_DIR, "cache")  # HF cache

# Create directories if they don't exist
for dir_path in [MODELS_DIR, HF_DOWNLOADS_DIR, HF_CACHE_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# %% [2.1] Model Source Configuration
# Configure where to load the model from - local file or HuggingFace download

# Choose model source: "local" or "huggingface"
MODEL_SOURCE = "local"  # Options: "local" or "huggingface"

# Model identification
MODEL_NAME = "llama_32_3b_instruct_gguf"  # Folder name for this model
MODEL_FILENAME = "llama_32_3b_instruct_q8_0.gguf"  # Actual GGUF filename

# Local file configuration (for manually downloaded models)
# Local models go directly in: BASE_DIR/models/MODEL_NAME/
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME, MODEL_FILENAME)

# HuggingFace configuration
HF_REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
HF_FILENAME = "Llama-3.2-3B-Instruct-Q8_0.gguf"  # Filename on HuggingFace
# HuggingFace models will be saved to: models/huggingface_downloads/MODEL_NAME/
HF_MODEL_DIR = os.path.join(HF_DOWNLOADS_DIR, MODEL_NAME)

# Resolve actual model path based on source
if MODEL_SOURCE == "huggingface":
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model from HuggingFace: {HF_REPO_ID}/{HF_FILENAME}")
        print(f"Download location: {HF_MODEL_DIR}")
        print(f"Cache location: {HF_CACHE_DIR}")
        
        MODEL_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=HF_CACHE_DIR,
            local_dir=HF_MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to: {MODEL_PATH}")
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        print("Falling back to local path...")
        MODEL_PATH = LOCAL_MODEL_PATH
    except Exception as e:
        print(f"ERROR downloading from HuggingFace: {e}")
        print("Falling back to local path...")
        MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH
    if not Path(MODEL_PATH).exists():
        print(f"WARNING: Local model not found at: {MODEL_PATH}")
        print(f"Expected location: {LOCAL_MODEL_PATH}")
        print(f"To download from HuggingFace, set MODEL_SOURCE = 'huggingface'")

# %% [2.2] User Configuration - All Settings
# Central location for all user-modifiable model and generation settings

# Hardware Configuration
CPU_ONLY = False         # Set True to force CPU-only mode
GPU_LAYERS = 28          # Number of layers to offload to GPU (28 = all for 3B)
THREADS = 8              # CPU threads to use
BATCH_SIZE = 512         # Batch size for processing
CONTEXT_LENGTH = 131072  # Context window size (131072 = full 128K)

# Memory Configuration
USE_MMAP = True          # Memory mapping for faster loading
USE_MLOCK = False        # Lock model in RAM (needs 32GB+ RAM)
USE_F16_KV = True        # Use 16-bit key/value cache

# Performance Optimizations
USE_FLASH_ATTN = True    # Flash Attention (for 30xx/40xx GPUs)
USE_CUDA_GRAPHS = False  # CUDA graphs (experimental)
OFFLOAD_KQV = False      # Keep KV cache on GPU

# Generation Configuration
TEMPERATURE = 0.7        # Randomness (0.0 = deterministic, 2.0 = very random)
TOP_P = 0.9             # Nucleus sampling threshold
TOP_K = 40              # Top-k sampling
REPEAT_PENALTY = 1.1    # Penalty for repetition
MAX_TOKENS = 2048       # Maximum tokens to generate
SEED = -1               # Random seed (-1 = random)

# Chat Configuration
SYSTEM_MESSAGE = "You are a helpful AI assistant."
CHAT_FORMAT = "llama-3"  # Chat template format

# Output Format Configuration
USE_JSON_MODE = False    # Force JSON output
USE_GRAMMAR = False      # Use grammar constraints

# Debug Configuration
VERBOSE = False          # Show detailed model loading info

# Generation Presets (alternative to manual settings above)
GENERATION_PRESETS = {
    "precise": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.1
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1
    },
    "creative": {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 100,
        "repeat_penalty": 1.0
    }
}

# Select a preset (None = use manual settings above)
USE_PRESET = None  # Options: None, "precise", "balanced", "creative"

# %% [2.3] Model Configuration Dataclass
# Structured container for passing configuration to model loader

@dataclass
class ModelConfig:
    """Configuration container - populated from user settings above"""
    # Model paths
    model_path: str
    
    # Hardware settings
    cpu_only: bool = CPU_ONLY
    n_gpu_layers: int = 0 if CPU_ONLY else GPU_LAYERS
    n_threads: int = THREADS
    n_batch: int = BATCH_SIZE
    n_ctx: int = CONTEXT_LENGTH
    
    # Memory settings
    use_mmap: bool = USE_MMAP
    use_mlock: bool = USE_MLOCK
    f16_kv: bool = USE_F16_KV
    
    # Generation settings
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repeat_penalty: float = REPEAT_PENALTY
    max_tokens: int = MAX_TOKENS
    seed: int = SEED
    
    # Advanced settings
    verbose: bool = VERBOSE
    use_flash_attention: bool = USE_FLASH_ATTN
    offload_kqv: bool = OFFLOAD_KQV
    
    # Output format
    use_json_mode: bool = USE_JSON_MODE
    grammar: Optional[str] = None
    
    # Chat settings
    chat_format: str = CHAT_FORMAT
    system_message: str = SYSTEM_MESSAGE

# %% [3.0] Hardware Auto-Detection
# Automatically determine optimal settings based on available hardware

def get_optimal_settings() -> Dict[str, Any]:
    """Auto-configure based on available hardware"""
    settings = {}
    
    # CPU settings
    cpu_count = psutil.cpu_count()
    settings['n_threads'] = min(8, cpu_count // 2)
    
    # Memory settings
    ram_gb = psutil.virtual_memory().total / (1024**3)
    settings['use_mlock'] = ram_gb >= 32
    
    # GPU settings for 3B model
    if NVIDIA_GPU_AVAILABLE and not CPU_ONLY:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        vram_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        vram_gb = vram_bytes / (1024**3)
        
        # Layer allocation for 3B model
        if vram_gb >= 24:
            settings['n_gpu_layers'] = 28  # All layers
        elif vram_gb >= 12:
            settings['n_gpu_layers'] = 24
        elif vram_gb >= 8:
            settings['n_gpu_layers'] = 18
        else:
            settings['n_gpu_layers'] = 12
    else:
        settings['n_gpu_layers'] = 0  # CPU only
    
    return settings

# %% [3.1] Hardware Detection
# Gather detailed hardware information for optimization decisions

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware capabilities"""
    info = {
        "platform": platform.system(),
        "cpu_name": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "gpu_available": False,
        "gpu_name": None,
        "vram_total_gb": 0,
        "vram_available_gb": 0
    }
    
    if NVIDIA_GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total = mem_info.total
            vram_free = mem_info.free
            
            info["gpu_available"] = True
            info["gpu_name"] = gpu_name
            info["vram_total_gb"] = round(vram_total / (1024**3), 1)
            info["vram_available_gb"] = round(vram_free / (1024**3), 1)
        except Exception as e:
            print(f"DEBUG: Error getting GPU info: {e}")
    
    return info

# %% [3.2] Environment Validation
# Verify Python version and required packages before proceeding

def validate_environment() -> bool:
    """Validate Python and CUDA environment"""
    valid = True
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or py_version.minor < 10:
        print(f"WARNING: Python {py_version.major}.{py_version.minor} detected. Python 3.10+ recommended.")
    
    # Check llama-cpp-python
    try:
        import llama_cpp
        print(f"OK: llama-cpp-python version {llama_cpp.__version__}")
    except ImportError:
        print("ERROR: llama-cpp-python not installed")
        valid = False
    
    # Check CUDA availability
    if NVIDIA_GPU_AVAILABLE:
        print("OK: NVIDIA GPU detected")
    else:
        print("INFO: Running in CPU mode (pynvml not available or no GPU)")
    
    return valid

# %% [4.0] Model Loader
# Class to handle GGUF model loading with optimal settings

class GGUFModelLoader:
    """Simple GGUF model loader"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
    
    def load(self) -> Llama:
        """Load the GGUF model with optimized settings"""
        
        print(f"Loading model from: {self.config.model_path}")
        print(f"GPU Layers: {self.config.n_gpu_layers}")
        print(f"Context Length: {self.config.n_ctx}")
        
        start_time = time.time()
        
        self.model = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            f16_kv=self.config.f16_kv,
            verbose=self.config.verbose,
            seed=self.config.seed if self.config.seed != -1 else None,
            flash_attn=self.config.use_flash_attention,
            offload_kqv=self.config.offload_kqv,
            chat_format=self.config.chat_format
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        return self.model

# %% [4.1] Model Validation
# Verify GGUF file integrity before attempting to load

def validate_model_file(path: str) -> bool:
    """Basic validation of GGUF file"""
    path = Path(path)
    
    if not path.exists():
        print(f"ERROR: Model file not found: {path}")
        return False
    
    # Check file size
    size_gb = path.stat().st_size / (1024**3)
    print(f"Model size: {size_gb:.2f} GB")
    
    if size_gb < 1:
        print("WARNING: Model seems too small for a 3B model")
    
    # Check GGUF header (magic number)
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                print("ERROR: Not a valid GGUF file")
                return False
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return False
    
    print("OK: Model validation passed")
    return True

# %% [5.0] Model Initialization
# Create and configure model instance with optional preset support

def initialize_model(config: Optional[ModelConfig] = None) -> Llama:
    """Initialize model with config"""
    
    if config is None:
        # Apply preset if selected
        gen_settings = {}
        if USE_PRESET and USE_PRESET in GENERATION_PRESETS:
            gen_settings = GENERATION_PRESETS[USE_PRESET]
        
        config = ModelConfig(
            model_path=MODEL_PATH,
            temperature=gen_settings.get('temperature', TEMPERATURE),
            top_p=gen_settings.get('top_p', TOP_P),
            top_k=gen_settings.get('top_k', TOP_K),
            repeat_penalty=gen_settings.get('repeat_penalty', REPEAT_PENALTY)
        )
    
    loader = GGUFModelLoader(config)
    return loader.load()

# %% [6.0] Inference Test
# Quick test to verify model works and measure performance

def test_inference(model: Llama, prompt: str = "Hello! How are you?") -> str:
    """Quick test to verify model works"""
    print("\n--- Running inference test ---")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=128
    )
    
    inference_time = time.time() - start_time
    
    result = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    tokens_per_second = total_tokens / inference_time
    
    print(f"Response: {result}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Tokens/second: {tokens_per_second:.1f}")
    
    return result

# %% [6.1] Terminal Chat Interface
# Interactive chat loop with conversation history and streaming responses

def chat_loop(model: Llama, config: ModelConfig):
    """Simple terminal chat interface"""
    print("\n--- Chat Interface ---")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("-" * 40)
    
    messages = [
        {"role": "system", "content": config.system_message}
    ]
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                messages = [{"role": "system", "content": config.system_message}]
                print("Conversation cleared")
                continue
            elif not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            response = model.create_chat_completion(
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                max_tokens=config.max_tokens,
                stream=True  # Stream for better UX
            )
            
            # Stream response
            full_response = ""
            for chunk in response:
                if 'content' in chunk['choices'][0]['delta']:
                    content = chunk['choices'][0]['delta']['content']
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # New line after response
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})
            
            # Keep conversation history manageable
            if len(messages) > 20:
                # Keep system message and last 18 messages
                messages = [messages[0]] + messages[-18:]
                
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")

# %% [7.0] Optional Features
# Additional capabilities like JSON mode and grammar constraints

def create_json_grammar(schema: dict) -> LlamaGrammar:
    """Create grammar for JSON output"""
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    return grammar

# Uncomment to use JSON generation
# def generate_json(model: Llama, prompt: str, schema: dict) -> dict:
#     grammar = create_json_grammar(schema)
#     
#     response = model(
#         prompt,
#         grammar=grammar,
#         max_tokens=500,
#         temperature=0.1  # Low temp for structured output
#     )
#     
#     return json.loads(response['choices'][0]['text'])

# %% [8.0] Main Entry Point
# Orchestrate the entire launch sequence from validation to chat interface

def main():
    """Main execution function"""
    
    print("=" * 50)
    print("Llama 3.2 3B Instruct GGUF Launcher")
    print("=" * 50)
    
    # Step 1: Validate environment
    if not validate_environment():
        print("ERROR: Environment validation failed")
        return 1
    
    # Step 2: Detect hardware
    hw_info = detect_hardware()
    print(f"\nCPU: {hw_info['cpu_name']}")
    print(f"Cores: {hw_info['cpu_count']}, RAM: {hw_info['ram_total_gb']}GB total, {hw_info['ram_available_gb']}GB available")
    if hw_info['gpu_available']:
        print(f"GPU: {hw_info['gpu_name']}")
        print(f"VRAM: {hw_info['vram_total_gb']}GB total, {hw_info['vram_available_gb']}GB available")
    
    # Step 3: Validate model file
    if not validate_model_file(MODEL_PATH):
        return 1
    
    # Step 4: Load model
    print("\nInitializing model...")
    model = initialize_model()
    
    # Step 5: Run test
    test_inference(model)
    
    # Step 6: Start chat
    config = ModelConfig(model_path=MODEL_PATH)
    
    print("\nStarting chat interface...")
    chat_loop(model, config)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
