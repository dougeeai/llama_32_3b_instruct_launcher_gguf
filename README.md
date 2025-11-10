# Llama 3.2 3B Instruct GGUF Launcher

Quick and dirty launcher scripts for running Llama-3.2-3B-Instruct GGUF models with llama-cpp-python.

## Description

Simple Python scripts to load and chat with Llama 3.2 3B Instruct GGUF quantized models. Includes a full version with hardware detection and optimizations, and a bare-bones version for minimal setup.

## Requirements

- Python 3.13
- llama-cpp-python with CUDA support (see installation below)
- 4-5GB VRAM for Q8_0 quantization
- GGUF model file (Q8_0 recommended for best quality)

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate llama_32_3b_instruct_launcher_gguf
```

2. Install llama-cpp-python wheel:
```bash
# Option A: Download pre-built wheel from https://github.com/dougeeai/llama-cpp-python-wheels
pip install llama_cpp_python-X.X.X-cpXXX-cpXXX-win_amd64.whl

# Option B: Build from source
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/13.0
```

3. **IMPORTANT: Update model path in the script**
```python
MODEL_PATH = r"E:\ai\models\llama_32_3b_instruct_gguf\llama_32_3b_instruct_q8_0.gguf"  # Change to your model location
```

4. Run:
```bash
python llama_32_3b_instruct_launcher_gguf.py  # Full version
# or
python llama_32_3b_instruct_launcher_gguf_bare.py  # Minimal version
```

## Files

- `llama_32_3b_instruct_launcher_gguf.py` - Full launcher with hardware detection, GPU layer optimization, streaming
- `llama_32_3b_instruct_launcher_gguf_bare.py` - Bare minimum code to load and chat
- `environment.yml` - Conda environment specification

## Configuration

Key settings in the script:
- `GPU_LAYERS` - Number of layers to offload to GPU (28 = all layers for 3B model)
- `CONTEXT_LENGTH` - Context window (131072 = full 128K context)
- `THREADS` - CPU threads for computation

## Usage

Type messages at the prompt. Commands:
- `quit` - Exit the chat
- `clear` - Reset conversation history (full version only)

## Notes

- GGUF models are more memory efficient than safetensors
- Q8_0 provides best quality/size ratio at ~3.2GB
- Bare version skips all validation and optimization features
- Default uses 28 GPU layers (full GPU offload)
