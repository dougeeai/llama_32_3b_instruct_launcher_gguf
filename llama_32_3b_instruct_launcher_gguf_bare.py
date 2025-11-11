# %% [0.0] Launcher Script Info
# Minimal launcher script metadata
# Llama 3.2 3B Instruct GGUF Launcher - BARE BONES VERSION
# Description: Minimal launcher for Llama-3.2-3B-Instruct GGUF models
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11

# %% [0.1] Model Card & Summary
# Bare bones version - minimal code, no checks
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)

# %% [1.0] Core Imports
# Essential imports only
import os
from pathlib import Path
from llama_cpp import Llama

# %% [1.1] Utility Imports
# Bare Version: Utility imports skipped

# %% [2.0] Base Directory Configuration
# Set base directory for portability
BASE_DIR = r"E:\ai"  # <-- CHANGE THIS to your AI folder location
MODELS_DIR = os.path.join(BASE_DIR, "models")

# %% [2.1] Model Source Configuration
# Simplified model path configuration
MODEL_NAME = "llama_32_3b_instruct_gguf"
MODEL_FILENAME = "llama_32_3b_instruct_q8_0.gguf"  # Change for different quants
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME, MODEL_FILENAME)

# %% [2.2] User Configuration - All Settings
# Core settings for model operation
GPU_LAYERS = 28
CONTEXT_LENGTH = 131072
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1
MAX_TOKENS = 2048
SYSTEM_MESSAGE = "You are a helpful AI assistant."

# %% [2.3] Model Configuration Dataclass
# Bare Version: Dataclass skipped - using direct variables

# %% [3.0] Hardware Auto-Detection
# Bare Version: Auto-detection skipped

# %% [3.1] Hardware Detection
# Bare Version: Hardware detection skipped

# %% [3.2] Environment Validation
# Bare Version: Environment validation skipped

# %% [4.0] Model Loader
# Direct model loading function
def load_model():
    """Load the GGUF model - bare bones"""
    return Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=GPU_LAYERS,
        n_ctx=CONTEXT_LENGTH,
        chat_format="llama-3"
    )

# %% [4.1] Model Validation
# Bare Version: Model validation skipped

# %% [5.0] Model Initialization
# Bare Version: Using direct load_model() instead

# %% [6.0] Inference Test
# Bare Version: Inference test skipped

# %% [6.1] Terminal Chat Interface
# Minimal chat loop with streaming
def chat_loop(model):
    """Minimal chat interface"""
    print("Chat Interface - Type 'quit' to exit")
    print("-" * 40)
    
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="", flush=True)
        
        response = model.create_chat_completion(
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPEAT_PENALTY,
            max_tokens=MAX_TOKENS,
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if 'content' in chunk['choices'][0]['delta']:
                content = chunk['choices'][0]['delta']['content']
                print(content, end="", flush=True)
                full_response += content
        
        print()
        messages.append({"role": "assistant", "content": full_response})
        
        # Keep history manageable
        if len(messages) > 20:
            messages = [messages[0]] + messages[-18:]

# %% [7.0] Optional Features
# Bare Version: Optional features skipped

# %% [8.0] Main Entry Point
# Simple main function - load and chat
def main():
    """Bare bones main - just load and chat"""
    print("Loading model...")
    model = load_model()
    print("Model loaded. Starting chat...")
    chat_loop(model)

if __name__ == "__main__":
    main()
